import json
import logging
import os
import sys
from typing import Optional
from dataclasses import dataclass, field

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# os.environ["HF_HOME"] = "/data8/zhangxin/aoeb/hf_cache"
os.environ["HF_HUB_ENDPOINT"] = "https://hf-mirror.com"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
# os.environ["LOCAL_DATA_PREFIX"] = os.environ.get("LOCAL_DATA_PREFIX", "/data/workspace/aoeb/local-data")

import torch
from transformers import HfArgumentParser, AutoModel
# from transformers.utils.versions import require_version
import mteb
# from mteb.models import model_meta_from_sentence_transformers
# mteb2
from mteb.models.model_meta import ModelMeta
from mteb.models.get_model_meta import _model_meta_from_sentence_transformers
from mteb.cache import ResultCache
from aoeb.st_wrapper import STWrapper
from mteb.models.model_implementations.jina_models import JinaV4Wrapper
from mteb.models.model_implementations.vista_models import vista_loader
from mteb.models.model_implementations.gme_v_models import GmeQwen2VL
from mteb.models.model_implementations.vlm2vec_models import VLM2VecWrapper
from mteb.models.sentence_transformer_wrapper import SentenceTransformerMultimodalEncoderWrapper


logging.basicConfig(
    format="%(levelname)s|%(asctime)s|%(name)s#%(lineno)s: %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger('run_mteb.py')
# require_version("mteb>=1.35.0", "To fix: pip install --upgrade mteb")


@dataclass
class EvalArguments:
    """
    Arguments.
    """
    model_path: Optional[str] = field(
        default='/data8/xjx/pretrained_models/e5-base-v2',
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    model_kwargs: Optional[str] = field(
        default=None,
        metadata={"help": "The specific model kwargs, json string."},
    )
    mteb_model: Optional[bool] = field(
        default=False, metadata={"help": "If `True`, use mteb native models."}
    )
    encode_kwargs: Optional[str] = field(
        default=None,
        metadata={"help": "The specific encode kwargs, json string."},
    )
    run_kwargs: Optional[str] = field(
        default=None,
        metadata={"help": "The specific kwargs for `MTEB.run()`, json string."},
    )

    output_dir: Optional[str] = field(default="results", metadata={"help": "output dir of results"})
    # benchmark: Optional[str] = field(default=None, metadata={"help": "Benchmark name"})
    tasks: Optional[str] = field(default="ToolRet", metadata={"help": "',' seprated"})
    # langs: Optional[str] = field(default=None, metadata={"help": "',' seprated"})
    only_load: bool = field(default=False, metadata={"help": ""})
    load_model: bool = field(default=False, metadata={"help": "when only_load"})

    # fast_retrieval: bool = field(default=True, metadata={"help": ""})
    # corpus_chunk_size: Optional[int] = field(default=None, metadata={"help": "corpus_chunk for eval"})
    batch_size: int = field(default=32, metadata={"help": "Will be set to `encode_kwargs`"})
    precision: str = field(default='fp16', metadata={"help": "amp_fp16,amp_bf16,fp16,bf16,fp32"})

    is_multimodal: bool = field(default=False, metadata={"help": "If True, use multimodal model wrapper."})

    def __post_init__(self):
        if isinstance(self.tasks, str):
            self.tasks = [s for s in self.tasks.split(',') if s]
        # if isinstance(self.langs, str):
        #     self.langs = [s for s in self.langs.split(',') if s]
        for name in ('model', 'encode', 'run'):
            name = name + '_kwargs'
            attr = getattr(self, name)
            if attr is None:
                setattr(self, name, dict())
            elif isinstance(attr, str):
                setattr(self, name, json.loads(attr))


def run_eval(model, tasks: list, args: EvalArguments, **kwargs):
    if not tasks:
        raise RuntimeError("No task selected")

    encode_kwargs = args.encode_kwargs or dict()

    _num_gpus, _started = torch.cuda.device_count(), False
    if _num_gpus > 1 and not _started and hasattr(model, 'start'):
        model.start()
        _started = True

    for t in tasks:
        # mteb 2.1.0
        results = mteb.evaluate(
            model,
            [t],
            cache = ResultCache(cache_path="."), #会自动在cache_path下创建一个results目录
            encode_kwargs=encode_kwargs,
            **kwargs
        )

    if model is not None and _started and hasattr(model, 'stop'):
        model.stop()
    return results


def main():
    parser = HfArgumentParser(EvalArguments)
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        with open(os.path.abspath(sys.argv[1])) as f:
            config = json.load(f)
        logger.warning(f"Json config {f.name} : \n{json.dumps(config, indent=2)}")
        args, *_ = parser.parse_dict(config)
        del config, f
    else:
        args, *_ = parser.parse_args_into_dataclasses()
        logger.warning(f"Args {args}")
    del parser

    tasks = mteb.get_tasks(tasks=args.tasks)
    logger.warning(f"Selected {len(tasks)} tasks:\n" + '\n'.join(str(t) for t in tasks))
    # for task in tasks:
    #     if "TempReason" in task.metadata.name:
    #         if task.metadata.dataset["config_name"] is None:
    #             task.metadata.dataset["config_name"] = "queries"
    if args.only_load:
        for t in tasks:
            logger.warning(f"Loading {t}")
            t.load_data()
        if not args.load_model:
            return

    if args.precision == 'fp16':
        args.model_kwargs.update({"torch_dtype": torch.float16})
    elif args.precision == 'bf16':
        args.model_kwargs.update({"torch_dtype": torch.bfloat16})
    # device = 'cuda' if torch.cuda.device_count() == 1 else 'cpu'
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.is_multimodal:
        # jina-embeddings-v4
        if 'jina' in args.model_path and 'v4' in args.model_path:
            model = JinaV4Wrapper(args.model_path, device=device, model_kwargs=args.model_kwargs)
            
        # VISTA model
        elif 'bge' in args.model_path:
            if 'bge-base' in args.model_path:
                model_weight = "/data8/xjx/pretrained_models/bge-visualized/Visualized_base_en_v1.5.pth"
                image_tokens_num = 196
            elif 'bge-m3' in args.model_path:
                model_weight = "/data8/xjx/pretrained_models/bge-visualized/Visualized_m3.pth"
                image_tokens_num = 256
            
            # 合并 loader_kwargs
            loader_kwargs = {
                "model_weight": model_weight,
                "image_tokens_num": image_tokens_num,
            }
            loader_kwargs.update(args.model_kwargs)
            
            model = vista_loader(args.model_path, **loader_kwargs)
        
        elif 'mme5' in args.model_path.lower():
            model = SentenceTransformerMultimodalEncoderWrapper(
                model=args.model_path,
                model_kwargs=args.model_kwargs
            )
        elif 'gme' in args.model_path.lower():
            # model = AutoModel.from_pretrained(args.model_path, **args.model_kwargs)
            model = GmeQwen2VL(
                model_name=args.model_path,
                revision="latest",
                device=device,
                **args.model_kwargs
            )
        elif 'vlm2vec' in args.model_path.lower():
            model = VLM2VecWrapper(
                model_name=args.model_path,
                device=device,
                **args.model_kwargs
            )
        elif 'mm-embed' in args.model_path.lower():
            args.model_kwargs["device_map"] = "auto"
            model = AutoModel.from_pretrained(args.model_path, **args.model_kwargs)
            # model = AutoModel.from_pretrained(args.model_path, **args.model_kwargs).to(device)
            
        # 模型元数据
        meta_data = {
            "name": f"multimodal/{args.model_path.split('/')[-1]}",
            "loader": lambda: None,
            "revision": "latest",
            "languages": ["eng-Latn"],
            "modalities": ["text", "image"],
            # 添加所有其他必需字段的默认值
            "release_date": "2024-01-01",
            "n_parameters": 11000000000,
            "memory_usage_mb": 20000,
            "max_tokens": 2048,
            "embed_dim": 1024,
            "license": "apache-2.0",  # 必须是预定义的许可证之一
            "open_weights": True,
            "public_training_code": "https://example.com",  # 应该是URL字符串
            "public_training_data": False,
            "framework": ["PyTorch"],  # 应该是列表
            "similarity_fn_name": "cosine",
            "use_instructions": True,
            "training_datasets": set()  # 应该是字典
        }
        if 'jina' in args.model_path and 'v4' in args.model_path:
            from mteb.models.model_implementations.jina_models import jina_embeddings_v4
            model.mteb_model_meta = jina_embeddings_v4
        elif 'mme5' in args.model_path.lower():
            from mteb.models.model_implementations.mme5_models import mme5_mllama
            model.mteb_model_meta = mme5_mllama
        else:   
            model.mteb_model_meta = ModelMeta(
                    **meta_data
                )


    else:
        device = 'cuda' if torch.cuda.device_count() == 1 else 'cpu'
        if "gme" in args.model_path or "mme5" in args.model_path.lower():
            args.model_kwargs.pop("torch_dtype", None)
            model = STWrapper(args.model_path, device=device, model_kwargs=args.model_kwargs)
        elif "gte-multilingual" in args.model_path.lower() or "nomic-embed-text" in args.model_path.lower():
            model = STWrapper(args.model_path, device=device, model_kwargs=args.model_kwargs, trust_remote_code=True)
        elif "jina" in args.model_path.lower():
            # args.model_kwargs.pop("torch_dtype", None)
            # model = STWrapper(args.model_path, device=device, **args.model_kwargs)
            model = JinaV4Wrapper(args.model_path, device=device, model_kwargs=args.model_kwargs)
        elif "nv-embed" in args.model_path.lower():
            args.model_kwargs.pop("return_dict", None)
            model = STWrapper(args.model_path, device=device, model_kwargs=args.model_kwargs, trust_remote_code=True)
        else:
            model = STWrapper(args.model_path, device=device, model_kwargs=args.model_kwargs)
        
        if "jina" in args.model_path.lower():
            from mteb.models.model_implementations.jina_models import jina_embeddings_v4
            model.mteb_model_meta = jina_embeddings_v4
            # 修改一下模态信息
            model.mteb_model_meta.modalities = ["text"]
        elif "mme5" in args.model_path.lower():
            model.model.max_seq_length = min(2048, model.model.max_seq_length) if model.model.max_seq_length is not None else 2048
            print(model.model.max_seq_length)
            model.mteb_model_meta = _model_meta_from_sentence_transformers(model.model)
    
        else:
            model.model.max_seq_length = min(8192, model.model.max_seq_length) if model.model.max_seq_length is not None else 8192
            model.mteb_model_meta = _model_meta_from_sentence_transformers(model.model)
    
    if model.mteb_model_meta.name is None:
        print("model name is None")
        model.mteb_model_meta.name = f"local/{os.path.basename(args.model_path)}"
    if args.only_load:
        return

    args.encode_kwargs.update(batch_size=args.batch_size)
    run_eval(model, tasks, args, **args.run_kwargs)
    logger.warning(f"Done {len(tasks)} tasks.")
    return


if __name__ == "__main__":
    main()
