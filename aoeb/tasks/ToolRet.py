from aoeb.abstasks.LocalRetrieval import LocalRetrieval
from mteb.abstasks.TaskMetadata import TaskMetadata


class ToolRet(LocalRetrieval):
    metadata = TaskMetadata(

        name="ToolRet",
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
            "prefix": "local-data",
            "path": "vec-ai/mangopy_toolret",   # 根目录，下面有 web/code/customized
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
            "qrels_file_name": "test.tsv",
            
        },     
    )
