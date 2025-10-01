# 由于 memory 任务的特殊性，需要重写一下 evaluate 部分对逻辑
# 原 LocalAbsTaskRetrievalV3.py
import os
import csv
import json
import numpy as np
from typing import List, Any
from time import time
from mteb.abstasks.TaskMetadata import TaskMetadata
from mteb.abstasks.AbsTaskRetrieval import HFDataLoader, AbsTaskRetrieval
from mteb.abstasks.MultilingualTask import MultilingualTask
from aoeb.abstasks.LocalRetrieval import LocalRetrieval
from mteb.abstasks.TaskMetadata import HFSubset
from mteb.evaluation.evaluators import RetrievalEvaluator
from mteb.load_results.task_results import ScoresDict

PREFIX = os.environ.get("LOCAL_DATA_PREFIX", "local-data")


class SubsetRetrieval(LocalRetrieval):
    metadata = TaskMetadata(
        name="UselessTaskName",
        description=(
            "Instruction retrieval benchmark: queries include an instruction + query, "
            "corpus contains tool documentation passages."
        ),
        reference="https://example.com/toolret",
        type="Retrieval",
        category="s2p",  # sentence-to-passage retrieval
        modalities=["text"],
        eval_splits=["test"],
        eval_langs={
            "web": ["eng-Latn"],
            "code": ["eng-Latn"],
            "customized": ["eng-Latn"]
        },
        main_score="ndcg_at_10",
        date=("2024-01-01", "2024-12-31"),
        domains=["Programming", "Web"],
        task_subtypes=[],
        license="cc-by-nc-sa-4.0",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation="""@inproceedings{your_citation_2024,
title={ToolRet: A Benchmark for Tool Retrieval},
author={...},
booktitle={...},
year={2024}
}""",
        dataset={
            "path": "../datasets/mangopy_toolret1",   # 根目录，下面有 web/code/customized
            "revision": "1.0"
        },
        
    )

    # 修改 evaluate 函数逻辑
    # 1. 调整 topk
    # 2. 重新处理results，只保留 query 对应的 会话，再算分
    def evaluate(
        self,
        model,
        split: str = "test",
        subsets_to_run: list[HFSubset] | None = None,
        *,
        encode_kwargs: dict[str, Any] = {},
        **kwargs,
    ) -> dict[HFSubset, ScoresDict]:
        # retriever = RetrievalEvaluator(
        #     retriever=model,
        #     task_name=self.metadata.name,
        #     encode_kwargs=encode_kwargs,
        #     # 手动修改 topk, longmemeval 需要的topk是[5, 10, corpus.length]
        #     k_values=self.k_values,
        #     **kwargs,
        # )

        scores = {}
        hf_subsets = list(self.hf_subsets) if self.is_multilingual else ["default"]
        if subsets_to_run is not None:
            hf_subsets = [s for s in hf_subsets if s in subsets_to_run]

        for hf_subset in hf_subsets:
            # logger.info(f"Subset: {hf_subset}")
            print(f"Subset: {hf_subset}")

            if hf_subset == "default":
                corpus, queries, relevant_docs = (
                    self.corpus[split],
                    self.queries[split],
                    self.relevant_docs[split],
                )
            else:
                corpus, queries, relevant_docs = (
                    self.corpus[hf_subset][split],
                    self.queries[hf_subset][split],
                    self.relevant_docs[hf_subset][split],
                )
            
            self.k_values.append(len(corpus))
            retriever = RetrievalEvaluator(
                retriever=model,
                task_name=self.metadata.name,
                encode_kwargs=encode_kwargs,
                # 手动修改 topk, longmemeval 需要的topk是[5, 10, corpus.length]
                k_values=self.k_values,
                **kwargs,
            )
            
            scores[hf_subset] = self._evaluate_subset(
                retriever, corpus, queries, relevant_docs, hf_subset, **kwargs
            )
            self.k_values.pop()
        return scores

    def _evaluate_subset(
        self, retriever, corpus, queries, relevant_docs, hf_subset: str, **kwargs
    ) -> ScoresDict:
        start_time = time()
        print(f"size of corpus: {len(corpus)}")
        print(f"size of queries: {len(queries)}")
        results = retriever(corpus, queries)
        end_time = time()
        # logger.info(f"Time taken to retrieve: {end_time - start_time:.2f} seconds")
        print(f"Time taken to retrieve: {end_time - start_time:.2f} seconds")

        # 加工 results
        # results 是 dict(str, dict(str, float))
        # results[qid][docid] = score
        # 建立 {qid: list(docid)} 的映射
        # ""
        qid_docids_map = {}
        if self.metadata_dict["dataset"]["origin_data_file"].endswith(".jsonl"):
            origin_data_file = os.path.join(PREFIX, self.metadata_dict["dataset"]["path"], hf_subset, self.metadata_dict["dataset"]["origin_data_file"])
            with open(origin_data_file, "r", encoding="utf-8") as f:
                for line in f:
                    data_point = json.loads(line)
                    qid_docids_map[data_point["query_id"]] = data_point["candidate_doc_ids"]
        else:
            origin_data_file = os.path.join(PREFIX, self.metadata_dict["dataset"]["path"], self.metadata_dict["dataset"]["origin_data_file"])
            with open(origin_data_file, "r", encoding="utf-8") as f:
                origin_data = json.load(f)
                for data_point in origin_data:
                    qid_docids_map[data_point["question_id"]] = data_point["haystack_session_ids"]
        
        
        # 过滤 results
        for qid, doc_scores in results.items():
            docids = qid_docids_map[qid]
            if not docids:
                print(f"qid: {qid} has empty docids")
            results[qid] = {docid: score for docid, score in doc_scores.items() if docid in docids}

        # debug 检查results里有没有空列表
        for qid, doc_scores in results.items():
            if not doc_scores:
                print(f"qid: {qid} has empty doc_scores")

        # qids_qrels = set(relevant_docs.keys())
        # qids_results = set(results.keys())

        # # 打印基本情况
        # print("[DBG] |relevant_docs|:", len(qids_qrels))
        # print("[DBG] |results|:", len(qids_results))

        # # 查找差异
        # only_in_results = qids_results - qids_qrels
        # only_in_qrels = qids_qrels - qids_results
        # print("[DBG] only_in_results:", len(only_in_results))
        # print("[DBG] only_in_qrels:", len(only_in_qrels))

        # # 检查哪些 qrels 没有正样本
        # zero_pos = [
        #     qid for qid, rels in relevant_docs.items()
        #     if not any(r > 0 for r in rels.values())
        # ]
        # print("[DBG] qids with zero positives:", len(zero_pos))
        
        ndcg, _map, recall, precision, naucs = retriever.evaluate(
            relevant_docs,
            results,
            # 计算指标的时候不需要那么大的 topk
            retriever.k_values[:-1],
            ignore_identical_ids=self.ignore_identical_ids,
            
        )
        mrr, naucs_mrr = retriever.evaluate_custom(
            relevant_docs, results, retriever.k_values, "mrr"
        )
        scores = {
            **{f"ndcg_at_{k.split('@')[1]}": v for (k, v) in ndcg.items()},
            **{f"map_at_{k.split('@')[1]}": v for (k, v) in _map.items()},
            **{f"recall_at_{k.split('@')[1]}": v for (k, v) in recall.items()},
            **{f"precision_at_{k.split('@')[1]}": v for (k, v) in precision.items()},
            **{f"mrr_at_{k.split('@')[1]}": v for (k, v) in mrr.items()},
            **{
                k.replace("@", "_at_").replace("_P", "_precision").lower(): v
                for k, v in naucs.items()
            },
            **{
                k.replace("@", "_at_").replace("_P", "_precision").lower(): v
                for k, v in naucs_mrr.items()
            },
        }
        self._add_main_score(scores)


        return scores
    


