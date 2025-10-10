# ToolRetV4.py
from aoeb.abstasks.LocalRetrieval import LocalRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class CosQAPlus(LocalRetrieval):
    metadata = TaskMetadata(
        name="CosQAPlus",
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
            # 暂时必须得有一个“子集”
            "nl2code": ["eng-Latn"],
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
            "path": "vec-ai/cosqa-plus",   
            "revision": "1.0",
            # 文件字段名声明
            "query_file_name": "queries.jsonl",
            "query_id_field": "id",
            "query_text_field": "text",
            "corpus_file_name": "corpus.jsonl",
            "corpus_id_field": "id",
            "corpus_title_field": "title",
            "corpus_text_field": "text",
            "qrels_dir": "qrels",
            "qrels_file_name": "qrels.tsv",
            # "origin_data_file": "candidates.jsonl"
        },
    )
