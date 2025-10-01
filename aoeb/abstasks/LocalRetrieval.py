# 原 LocalAbsTaskRetrievalV2.py
import os
import csv
import numpy as np
from typing import List
from mteb.abstasks.TaskMetadata import TaskMetadata
from mteb.abstasks.AbsTaskRetrieval import HFDataLoader, AbsTaskRetrieval
from mteb.abstasks.MultilingualTask import MultilingualTask

from datasets import load_dataset

PREFIX = os.environ.get("LOCAL_DATA_PREFIX", "local-data")


class LocalRetrieval(AbsTaskRetrieval, MultilingualTask):
    """
    Retrieval task for local ToolRet dataset with multiple subsets (web, code, customized).
    """
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

    def load_qrels_from_tsv(self, qrels_path: str):
        qrels_dict = {}
        with open(qrels_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t", fieldnames=["query-id","corpus-id","score"])
            for row in reader:
                qid = row["query-id"]
                doc_id = row["corpus-id"]
                score = int(row["score"])
                qrels_dict.setdefault(qid, {})[doc_id] = score
        return qrels_dict

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        # corpus = {"web": {}, "code": {}, "customized": {}}
        self.corpus, self.queries, self.relevant_docs = {}, {}, {}
        eval_splits = kwargs.get("eval_splits", ["test"])

        
        subsets = list(self.metadata.eval_langs.keys())

        for hf_subset in subsets:
            self.corpus[hf_subset] = {}
            self.queries[hf_subset] = {}
            self.relevant_docs[hf_subset] = {}

            for split in eval_splits:
                data_folder = os.path.join(PREFIX, self.metadata.dataset["path"], hf_subset)

                queries_path = os.path.join(data_folder, self.metadata.dataset["query_file_name"])
                corpus_path = os.path.join(data_folder, self.metadata.dataset["corpus_file_name"])
                qrels_path = os.path.join(data_folder, self.metadata.dataset["qrels_dir"], self.metadata.dataset["qrels_file_name"])

                # 直接用 datasets.load_dataset 读 jsonl
                corpus_ds = load_dataset("json", data_files=corpus_path, split="train")
                queries_ds = load_dataset("json", data_files=queries_path, split="train")
                print(f"111Loaded {len(corpus_ds)} documents and {len(queries_ds)} queries for {hf_subset} {split} split.")

                # 保险起见，移除 _id
                if "_id" in corpus_ds.column_names:
                    corpus_ds = corpus_ds.remove_columns("_id")
                if "_id" in queries_ds.column_names:
                    queries_ds = queries_ds.remove_columns("_id")

                queries = {q[self.metadata.dataset["query_id_field"]]: q[self.metadata.dataset["query_text_field"]] for q in queries_ds}
                corpus = {d[self.metadata.dataset["corpus_id_field"]]: d.get(self.metadata.dataset["corpus_title_field"], "") + " " + d.get(self.metadata.dataset["corpus_text_field"], "") for d in corpus_ds}
                
                qrels_ds = self.load_qrels_from_tsv(qrels_path)


                self.corpus[hf_subset][split] = corpus
                self.queries[hf_subset][split] = queries
                self.relevant_docs[hf_subset][split] = qrels_ds
                print(f"Loaded {len(corpus)} documents and {len(queries)} queries for {hf_subset} {split} split.")
        self.data_loaded = True