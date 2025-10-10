import argparse
import os
import torch
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["HF_HOME"] = "/data8/zhangxin/aoeb/hf_cache"
os.environ["HF_HUB_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# os.environ["LOCAL_DATA_PREFIX"] = os.environ.get("LOCAL_DATA_PREFIX", "/data/workspace/aoeb/local-data")

import mteb
from sentence_transformers import SentenceTransformer

import aoeb


def get_parser():
    parser = argparse.ArgumentParser(description="Run MTEB-style benchmark on multiple tasks for a single model.")
    parser.add_argument("--model_path", type=str, default="/data/workspace/models/Qwen/Qwen3-Embedding-0.6B")
    parser.add_argument("--tasks", type=str, default="ToolRet")
    parser.add_argument("--output_folder", type=str, default="results/")
    parser.add_argument("--batch_size", type=int, default=32)
    return parser


def main():
    # 指定 model
    # tasks 支持一次评测多任务
    # output_folder 以模型为单位存储评测结果
    parser = get_parser()
    args = parser.parse_args()

    model = SentenceTransformer(args.model_path, model_kwargs={"torch_dtype": torch.float16})
    model.max_seq_length = min(8*1024, model.max_seq_length)
    tasks = args.tasks.split(",")
    print(f"Running tasks: {tasks}")
    output_folder = os.path.join(args.output_folder, args.model_path.split("/")[-1], "20251009")

    # 验证任务是否已注册
    print(f"Available custom tasks: {list(aoeb.LOCAL_REGISTRY.keys())}")
    
    # tasks: list[AbsTask] or list[str] 对应本地任务类和 mteb 已支持的任务类
    tasks = mteb.get_tasks(tasks=tasks)
    evaluation = mteb.MTEB(tasks=tasks)
    encode_kwargs = {
        "batch_size": args.batch_size,
        "show_progress_bar": True,
        "normalize_embeddings": True,
    }
    results = evaluation.run(
        model, 
        output_folder=output_folder, 
        encode_kwargs=encode_kwargs,
    )
    print(results)
    return


if __name__ == "__main__":
    main()


# local-data 应该要关联一下远程仓库
