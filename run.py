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
from transformers import HfArgumentParser
# from transformers.utils.versions import require_version
import mteb
from mteb.models import model_meta_from_sentence_transformers

from aoeb.st_wrapper import STWrapper


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
        evaluation = mteb.MTEB(tasks=[t])
        eval_splits = evaluation.tasks[0].metadata.eval_splits
        if len(eval_splits) > 1:
            eval_splits = ['test']
        results = evaluation.run(
            model,
            output_folder=args.output_dir,
            encode_kwargs=encode_kwargs,
            eval_splits=eval_splits,
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
    device = 'cuda' if torch.cuda.device_count() == 1 else 'cpu'
    model = STWrapper(args.model_path, device=device, model_kwargs=args.model_kwargs)
    model.model.max_seq_length = min(8192, model.model.max_seq_length)
    if args.only_load:
        return
    model.mteb_model_meta = model_meta_from_sentence_transformers(model.model)
    model.mteb_model_meta.name = os.path.basename(args.model_path)

    args.encode_kwargs.update(batch_size=args.batch_size)
    run_eval(model, tasks, args, **args.run_kwargs)
    logger.warning(f"Done {len(tasks)} tasks.")
    return


if __name__ == "__main__":
    main()
