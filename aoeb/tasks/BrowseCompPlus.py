from aoeb.abstasks.LocalRetrieval import LocalRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class BrowseCompPlus(LocalRetrieval):
    metadata = TaskMetadata(
        name="BrowseCompPlus",
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
            "gold": ["eng-Latn"]
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
            "path": "vec-ai/browsecomp-plus",   # 根目录，下面有 web/code/customized
            "revision": "1.0",
            # 文件字段名声明
            "query_file_name": "decrypted.jsonl",
            "query_id_field": "query_id",
            "query_text_field": "query",
            "corpus_file_name": "mini_corpus.jsonl",
            "corpus_id_field": "docid",
            "corpus_title_field": "title",
            "corpus_text_field": "text",
            "qrels_dir": "qrels",
            "qrels_file_name": "evidence_qrels.tsv",
        },
    )
