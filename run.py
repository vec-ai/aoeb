import argparse
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '7'
os.environ['HF_HOME'] = '/data8/zhangxin/aoeb/hf_cache'
os.environ['HF_HUB_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import mteb
from mteb.tasks.Retrieval import ArguAna
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('/data8/xjx/pretrained_models/e5-small-v2')

# tasks = mteb.get_tasks(tasks=['ArguAna'])
# tasks[0].metadata_dict["dataset"]["path"] = "../../datasets/mangopy_ToolRet_Training_20w"
evaluation = mteb.MTEB(tasks=[ArguAna()])
results = evaluation.run(model, output_folder="results/")


def main():
    parser = get_parser()
    args = parser.parse_args()

    # model
    # tasks
    # output_folder


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./models/origin-lychee-rerank')
    return parser


if __name__ == "__main__":
    main()
