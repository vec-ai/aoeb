import argparse
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["HF_HOME"] = "/data8/zhangxin/aoeb/hf_cache"
os.environ["HF_HUB_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import mteb
from mteb.tasks.Retrieval import ArguAna
from sentence_transformers import SentenceTransformer
from aoeb.tasks.BrowseCompPlus import BrowseCompPlus
from aoeb.tasks.ToolRet import ToolRet
from aoeb.tasks.LongMemEval import LongMemEval
from aoeb.tasks.LoCoMo import LoCoMo

def get_local_task_instance(local_task_dict, task_name, **kwargs):
    # 下划线改为 -
    # 全部字母小写
    task_name = task_name.lower().replace("_", "-")
    task_cls = local_task_dict.get(task_name)
    if task_cls is None:
        raise ValueError(f"Task {task_name} not defined.")
    return task_cls(**kwargs)

def get_parser():
    parser = argparse.ArgumentParser(description="Run MTEB-style benchmark on multiple tasks for a single model.")
    parser.add_argument("--model_path", type=str, default="./models/origin-lychee-rerank")
    parser.add_argument("--tasks", type=str, default="")
    parser.add_argument("--output_folder", required=True, type=str)
    return parser

def main():
    # 指定 model
    # tasks 支持一次评测多任务
    # output_folder 以模型为单位存储评测结果
    parser = get_parser()
    args = parser.parse_args()

    model = SentenceTransformer(args.model_path)
    tasks = args.tasks.split(",")
    print(f"Running tasks: {tasks}")
    output_folder = os.path.join(args.output_folder, args.model_path.split("/")[-1])

    # tasks: list[AbsTask] or list[str] 对应本地任务类和 mteb 已支持的任务类
    # 本地任务模式
    local_tasks_dict = {
        "browsecomp-plus": BrowseCompPlus,
        "toolret": ToolRet,
        "longmemeval": LongMemEval,
        "locomo": LoCoMo,
    }
    tasks = [get_local_task_instance(local_tasks_dict, task) for task in tasks]
    evaluation = mteb.MTEB(tasks=tasks)
    # evaluation = mteb.MTEB(tasks=[BrowseCompPlus(), ToolRet(), LongMemEval()])
    results = evaluation.run(model, output_folder=output_folder)
    print(results) 

    # # mteb 模式
    # tasks = mteb.get_tasks(
    #     tasks = [
    #         "AppsRetrieval",
    #         "CosQA",
    #     ]
    # )
    # evaluation_mteb = mteb.MTEB(tasks=tasks)
    # results = evaluation_mteb.run(model, output_folder=output_folder)
    # print(results)

if __name__ == "__main__":
    main()


# local-data 应该要关联一下远程仓库
