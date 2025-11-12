import logging
import os
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = '6'

# Setup logging
logging.basicConfig(
    format="%(asctime)s|%(name)s:%(lineno)s|%(levelname)s - %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)

import torch

from aoeb_ng.task import RetrievalTask
from aoeb_ng.model import SentenceTransformerEmbedder
from aoeb_ng.models.jinav4 import JinaV4Wrapper



def main():
    # task_config = '/data8/zhangxin/aoeb/aoeb_ng/tasks/FreshStackDocAngular.json'
    task_config = '/data8/zhangxin/aoeb/aoeb_ng/tasks/MMRC.json'
    task = RetrievalTask.from_config(task_config)
    # device = 'cuda' if torch.cuda.device_count() == 1 else 'cpu'
    # model = SentenceTransformerEmbedder("/data8/xjx/pretrained_models/bge-large-en-v1.5", device=device)
    # model.model = model.model.half()
    model = JinaV4Wrapper("/data8/xjx/pretrained_models/jina-embeddings-v4")
    task.evaluate(model, "results/bge-m3", encode_kwargs={'batch_size': 128})
    print("ok.")



if __name__ == "__main__":
    main()
