import logging
import math
import queue
from contextlib import nullcontext

from tqdm.autonotebook import tqdm
import numpy as np
import torch
from mteb.encoder_interface import PromptType
from mteb.models.sentence_transformer_wrapper import SentenceTransformerWrapper

logger = logging.getLogger(__name__)

AMP_DTYPE = None


def _encode_worker(target_device: str, model, input_queue, results_queue):
    device = torch.device(target_device)
    with torch.autocast(device_type=device.type, dtype=AMP_DTYPE) if AMP_DTYPE is not None else nullcontext():
        while True:
            try:
                chunk_id, sentences, kwargs = input_queue.get()
                embeddings = model.encode(
                    sentences,
                    device=device,
                    show_progress_bar=False,
                    **kwargs,
                ).cpu()
                results_queue.put([chunk_id, embeddings])
            except queue.Empty:
                break


def _encode_multi_process(mp_pool, sentences: list[str], chunk_size: int | None = None, **kwargs):
    if chunk_size is None:
        chunk_size = min(math.ceil(len(sentences) / len(mp_pool["processes"]) / 10), 5000)

    num_chunks = math.ceil(len(sentences) / chunk_size)
    logger.info(f"Chunk data into {num_chunks} packages of size {chunk_size}")

    def _receive(oq, timeout=0.00125):
        try:
            n, embed = oq.get(timeout=timeout)
            result_dict[n] = embed
            pbar.update(1)
        except queue.Empty:
            pass

    show_progress_bar = kwargs.pop('show_progress_bar', True)
    input_queue, output_queue = mp_pool["input"], mp_pool["output"]
    result_dict = dict()
    pbar = tqdm(
        total=num_chunks, disable=not show_progress_bar, mininterval=1, miniters=10, desc='encode_multi_process'
    )
    for n, i in enumerate(range(0, len(sentences), chunk_size)):
        chunk = sentences[i: i + chunk_size]
        input_queue.put((n, chunk, kwargs))
        _receive(output_queue)
    while len(result_dict) < num_chunks:
        _receive(output_queue)
    pbar.close()

    embeddings = torch.cat([result_dict[i] for i in range(num_chunks)])
    return embeddings


### https://github.com/embeddings-benchmark/mteb/blob/main/mteb/models/sentence_transformer_wrapper.py
from collections.abc import Sequence
from typing import Any


class STWrapper(SentenceTransformerWrapper):
    def encode(
        self,
        sentences: Sequence[str],
        *,
        task_name: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Encodes the given sentences using the encoder.

        Args:
            sentences: The sentences to encode.
            task_name: The name of the task. Sentence-transformers uses this to
                determine which prompt to use from a specified dictionary.
            prompt_type: The name type of prompt. (query or passage)
            **kwargs: Additional arguments to pass to the encoder.

            The order of priorities for prompt selection are:
                1. Composed prompt of task name + prompt type (query or passage)
                2. Specific task prompt
                3. Composed prompt of task type + prompt type (query or passage)
                4. Specific task type prompt
                5. Specific prompt type (query or passage)


        Returns:
            The encoded sentences.
        """
        prompt = None
        prompt_name = None
        if self.model_prompts is not None:
            prompt_name = self.get_prompt_name(
                self.model_prompts, task_name, prompt_type
            )
            prompt = self.model_prompts.get(prompt_name, None)
        if prompt_name:
            logger.info(
                f"Using {prompt_name=} for task={task_name} {prompt_type=} with {prompt=}"
            )
        else:
            logger.info(f"No model prompts found for task={task_name} {prompt_type=}")
        logger.info(f"Encoding {len(sentences)} sentences.")

        kwargs.update(convert_to_tensor=True)

        mp_pool = getattr(self, 'mp_pool', None)
        if mp_pool is None:
            with torch.autocast(
                device_type=self.device.type, dtype=AMP_DTYPE
            ) if AMP_DTYPE is not None else nullcontext():
                embeddings = self.model.encode(sentences, prompt=prompt, **kwargs).cpu().float()
        else:
            # kwargs.pop('output_value', 0)
            # kwargs.pop('device', 0)
            embeddings = _encode_multi_process(mp_pool, sentences, **kwargs).float()

        if isinstance(embeddings, torch.Tensor):
            # sometimes in kwargs can be return_tensors=True
            embeddings = embeddings.cpu().detach().float().numpy()
        return embeddings

    def _predict(
        self,
        sentences: Sequence[str],
        **kwargs: Any,
    ) -> np.ndarray:
        raise NotImplementedError("TODO")

    def start(self):
        from sentence_transformers import SentenceTransformer

        SentenceTransformer._encode_multi_process_worker = _encode_worker
        pool = self.model.start_multi_process_pool()
        setattr(self, 'mp_pool', pool)
        logger.info("Pool started")

    def stop(self):
        if pool := getattr(self, 'mp_pool', None):
            self.model.stop_multi_process_pool(pool)
