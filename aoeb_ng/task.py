import heapq
import io
import json
import logging
import os
from pathlib import Path

from dataclasses import dataclass
from enum import Enum
from typing import Sequence

import pytrec_eval
import pandas as pd
import datasets
import torch

from .data import IndexedMmapTarFile
from .model import EmbedderProtocol, MultimodalEmbedderProtocol

logger = logging.getLogger(__name__)


class Modal(Enum):
    LABEL = 0
    TEXT = 1
    IMAGE = 2
    SEQ = 3
    MIXED = 9


@dataclass
class RetrievalTask:
    name: str
    data: dict
    main_metric: str = "ndcg_at_10"
    k_values: Sequence[int] = (1, 3, 5, 10, 20, 100, 1000)
    abstask_prompt = "Retrieve text based on user query."
    ignore_identical_ids: bool = False
    chunk_size: int = 100000

    # data holders
    qrels: dict = None
    query_ds = None
    collection_ds = None
    packages: dict = None

    # for datasets
    keep_in_memory: bool = True

    @classmethod
    def from_config(cls, config_path: str):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return cls(**config)

    def load_data(self, load_all: bool = False):
        if load_all:
            if self.query_ds is None:
                self.query_ds = self._load_one(**self.data['query'])
            if self.collection_ds is None:
                self.collection_ds = self._load_one(**self.data['collection'])
        if self.qrels is None:
            label_ds = self._load_one(**self.data['label'])
            self.qrels = dict()
            for item in label_ds:
                self.qrels[item['id']] = json.loads(item['data'])
        return self

    def _load_one(
        self,
        path: str,
        modal: str,
        with_prompt: bool = False,
        load_kwargs: dict = None,
        xjx_style: bool = False,
    ) -> datasets.Dataset:
        load_kwargs = load_kwargs or dict()
        if 'split' not in load_kwargs:
            load_kwargs['split'] = 'train'
        if 'keep_in_memory' not in load_kwargs:
            load_kwargs['keep_in_memory'] = self.keep_in_memory

        if xjx_style:
            return self._load_one_xjx_style(
                path=path,
                modal=modal,
                load_kwargs=load_kwargs,
            )

        features = {'id': datasets.Value('string'), 'data': datasets.Value('string')}
        if with_prompt:
            features['prompt'] = datasets.Value('string')
        if modal == Modal.MIXED.name:
            features['modal'] = datasets.Value('string')

        ds = datasets.load_dataset(path, features=datasets.Features(features), **load_kwargs)

        if modal not in (Modal.LABEL.name, Modal.TEXT.name):
            self._load_packages()
        return ds

    def _load_packages(self):
        if self.packages is None:
            self.packages = dict()
        for k, v in self.data['packages'].items():
            if k not in self.packages:
                self.packages[k] = IndexedMmapTarFile(**v)
        return self.packages

    def _get_transform(self, ds_modal: str):
        from PIL import Image

        default_package = next(self.data['packages'])

        def load_image(path: str, package: str = None) -> Image.Image:
            package_name = package or default_package
            data_bytes = self.packages[package_name][path]
            img = Image.open(io.BytesIO(data_bytes)).convert('RGB')
            return img

        def decode_image(example):
            example['data'] = load_image(**example['data'])
            return example

        def decode_seq(example):
            seq_data = list()
            for item in example['data']:
                if isinstance(item, dict):
                    # TODO: support other modalities in seq
                    seq_data.append(load_image(**item))
                else:
                    seq_data.append(item)
            example['data'] = seq_data
            return example

        if ds_modal == Modal.IMAGE.name:
            return decode_image
        elif ds_modal == Modal.SEQ.name:
            return decode_seq
        else:
            raise ValueError(f"Unsupported modal type: {ds_modal}")

    def _load_one_xjx_style(
        self,
        path: str,
        modal: str,
        load_kwargs: dict = None,
    ) -> datasets.Dataset:
        if modal == Modal.LABEL.name:
            data = dict()
            with open(path) as file:
                for line in file:
                    qid, docid, label = line.strip().split('\t')
                    if qid not in data:
                        data[qid] = dict()
                    data[qid][docid] = int(label)
            data = [{'id': k, 'data': json.dumps(v)} for k, v in data.items()]
            ds = datasets.Dataset.from_list(data)
        else:
            ds = datasets.load_dataset(path, **load_kwargs)
            if modal == Modal.TEXT.name:
                ds = ds.rename_column('text', 'data')
        return ds

    def evaluate(self, model: EmbedderProtocol, output_dir: str, encode_kwargs=None) -> dict:
        task_dir = os.path.join(output_dir, self.name)
        path_metric = task_dir + '.json'
        if os.path.exists(path_metric):
            logger.warning(f"Task result {path_metric} exists, skip evaluation.")
            with open(path_metric) as file:
                metrics = json.load(file)
            return metrics

        os.makedirs(task_dir, exist_ok=True)
        path_result = Path(task_dir, 'result.json')
        if path_result.exists():
            with path_result.open() as file:
                results = json.load(file)
            logger.info(f"Found existing search results in {path_result}, skip encoding.")
        else:
            encode_kwargs = encode_kwargs or dict()
            encode_kwargs.update(task_name=self.name)
            query = self.encode(model, 'query', task_dir, encode_kwargs, no_chunk=True)
            collection = self.encode(model, 'collection', task_dir, encode_kwargs, no_chunk=True)
            results = self.search(query, collection)
            with path_result.open('w') as file:
                json.dump(results, file)
            logger.info(f"Task {self.name} search results saved to {path_result}")
        if self.qrels is None:
            self.load_data(load_all=False)
        metrics = self.compute_metrics(results, self.qrels)
        with open(path_metric, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Task {self.name} evaluation results saved to {path_metric}")
        logger.info(f"Metrics: {json.dumps(metrics, indent=2)}")
        return metrics

    def encode(
        self,
        model: EmbedderProtocol,
        input_type: str,
        task_dir: str,
        encode_kwargs: dict,
        no_chunk=False
    ) -> tuple:
        assert no_chunk
        save_path = Path(task_dir, f'{input_type}.pth')
        if no_chunk:
            if save_path.exists():
                encoding = torch.load(save_path)
                return encoding
        else:
            chunk_files = Path(task_dir).glob(f'{input_type}*.pth')
            if len(list(chunk_files)) > 0:
                encoding_iter = iter(torch.load(p) for p in chunk_files)
                return encoding_iter

        encode_kwargs.update(input_type=input_type)
        config = self.data[input_type]
        ds = self._load_one(**config)
        if config['modal'] == Modal.TEXT.name:
            ds = ds.sort('data')
        elif config['modal'] != Modal.MIXED.name:
            data_transform = self._get_transform(config['modal'])
            ds = ds.with_transform(data_transform)

        if config['modal'] != Modal.MIXED.name:
            encoding = self._encode(ds, config['modal'], model, encode_kwargs)
        else:
            modalities = (Modal.TEXT.name, Modal.IMAGE.name, Modal.SEQ.name)
            groups = list()
            for modal in modalities:
                ds_m = ds.filter(lambda x: x['modal'] == modal)
                if len(ds_m) > 0:
                    if modal != Modal.TEXT.name:
                        data_transform = self._get_transform(modal)
                        ds_m = ds_m.with_transform(data_transform)
                    groups.append(self._encode(ds_m, modal, model, encode_kwargs))
            ids = sum((g[0] for g in groups), start=list())
            vectors = torch.cat([g[1] for g in groups], dim=0)
            encoding = (ids, vectors)
        torch.save(encoding, save_path)
        return encoding

    def _encode(self, chunk, modal: Modal, model: EmbedderProtocol, encode_kwargs: dict) -> tuple:
        with torch.inference_mode():
            if isinstance(model, MultimodalEmbedderProtocol):
                if modal == Modal.TEXT.name:
                    vectors = model.embed_text(chunk, **encode_kwargs)
                if modal == Modal.IMAGE.name:
                    vectors = model.embed_image(chunk, **encode_kwargs)
                elif modal == Modal.SEQ.name:
                    vectors = model.embed_multimodal(chunk, **encode_kwargs)
                else:
                    raise ValueError(f"Unsupported modal type: {modal}")
            else:
                assert modal == Modal.TEXT.name
                vectors = model.embed(chunk, **encode_kwargs)
        ids = list(chunk['id'])
        return ids, vectors

    def search(self, query_encoding: tuple, chunks: list) -> dict:
        top_k = max(self.k_values)
        result_heaps = dict()
        query_ids, query_embed = query_encoding
        # TODO: Optimize chunking
        for doc_ids, doc_embed in chunks:
            if torch.cuda.is_available():
                query_embed = query_embed.cuda()
                doc_embed = doc_embed.cuda()

            scores = cos_sim(query_embed.float(), doc_embed.float())
            scores[torch.isnan(scores)] = -1
            kth = min(top_k + 1, scores.size(1))
            top_k_values, top_k_idx = scores.topk(kth, dim=1, largest=True)
            top_k_idx, top_k_values = top_k_idx.cpu().tolist(), top_k_values.cpu().tolist()

            for i, query_id in enumerate(query_ids):
                if query_id not in result_heaps:
                    result_heaps[query_id] = list()
                for j, score in zip(top_k_idx[i], top_k_values[i]):
                    doc_id = doc_ids[j]
                    if len(result_heaps[query_id]) < top_k:
                        # Push item on the heap
                        heapq.heappush(result_heaps[query_id], (score, doc_id))
                    else:
                        # If item is larger than the smallest in the heap,
                        # push it on the heap then pop the smallest element.
                        # Tuples are compared by each item.
                        heapq.heappushpop(result_heaps[query_id], (score, doc_id))

        result = dict()
        for query_id, heap in result_heaps.items():
            result[query_id] = {doc_id: score for score, doc_id in heap}
        return result

    def compute_metrics(self, results: dict, qrels: dict):
        # https://github.com/beir-cellar/beir/blob/main/beir/retrieval/custom_metrics.py
        def mrr(qrels: dict[str, dict[str, int]],
                results: dict[str, dict[str, float]],
                k_values: list[int]
            ) -> dict[str, float]:

            MRR = {}

            for k in k_values:
                MRR[f"mrr_at_{k}"] = 0.0

            k_max, top_hits = max(k_values), {}

            for query_id, doc_scores in results.items():
                # top_hits[query_id] = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[0:k_max]
                top_hits[query_id] = list(doc_scores.items())[0:k_max]

            for query_id, hits in top_hits.items():
                relevant_docs = set([doc_id for doc_id, score in qrels[query_id].items() if score > 0])
                for k in k_values:
                    for rank, hit in enumerate(hits[0:k]):
                        if hit[0] in relevant_docs:
                            MRR[f"mrr_at_{k}"] += 1.0 / (rank + 1)
                            break

            for k in k_values:
                MRR[f"mrr_at_{k}"] = MRR[f"mrr_at_{k}"] / len(qrels)

            return MRR

        map_string = "map_cut." + ",".join([str(k) for k in self.k_values])
        ndcg_string = "ndcg_cut." + ",".join([str(k) for k in self.k_values])
        recall_string = "recall." + ",".join([str(k) for k in self.k_values])
        precision_string = "P." + ",".join([str(k) for k in self.k_values])
        # qrels = {str(qid): {str(docid): s for docid, s in v.items()} for qid, v in qrels.items()}
        # results = {str(qid): {str(docid): s for s, docid in v} for qid, v in result_heaps.items()}
        evaluator = pytrec_eval.RelevanceEvaluator(
            qrels, {map_string, ndcg_string, recall_string, precision_string}
        )
        scores_by_query = evaluator.evaluate(results)
        scores = pd.DataFrame.from_dict(scores_by_query.values()).mean()
        # metrics = mrr(qrels, results, k_values)
        metrics = dict()  # TODO
        for prefix in ('map_cut', 'ndcg_cut', 'recall', 'P'):
            name = 'precision' if prefix == 'P' else prefix.split('_')[0]
            for k in self.k_values:
                metrics[f'{name}_at_{k}'] = scores[f'{prefix}_{k}']

        task_metrics = self.compute_task_metrics(results, qrels)
        if task_metrics:
            metrics.update(task_metrics)
        return metrics

    def compute_task_metrics(self, results: dict, qrels: dict):
        return None


def cos_sim(a: torch.Tensor, b: torch.Tensor, do_norm=True):
    """
    Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
    :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
    """
    if not isinstance(a, torch.Tensor):
        a = torch.tensor(a)

    if not isinstance(b, torch.Tensor):
        b = torch.tensor(b)

    if len(a.shape) == 1:
        a = a.unsqueeze(0)

    if len(b.shape) == 1:
        b = b.unsqueeze(0)

    if do_norm:
        a = torch.nn.functional.normalize(a, p=2, dim=1)
        b = torch.nn.functional.normalize(b, p=2, dim=1)

    return torch.mm(a, b.transpose(0, 1))
