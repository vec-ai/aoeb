
from typing import Any

import torch

from ..model import MultimodalEmbedderProtocol


class JinaV4Wrapper(MultimodalEmbedderProtocol):
    """following the hf model card documentation."""
    model: Any
    prompts_dict: dict[str, str] | None = None

    def __init__(
        self,
        model: str,
        revision: str | None = None,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code: bool = True,
        model_prompts: dict[str, str] | None = None,
        **kwargs,
    ) -> None:
        # requires_package(
        #     self,
        #     "flash_attn",
        #     model,
        #     "pip install 'mteb[flash_attention]'",
        # )
        # requires_package(self, "peft", model, "pip install 'mteb[jina-v4]'")
        # requires_package(self, "torchvision", model, "pip install 'mteb[jina-v4]'")
        import flash_attn  # noqa: F401
        import peft  # noqa: F401
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained(
            model,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation,
            revision=revision,
        ).eval()
        self.model_prompts = model_prompts or {}
        self.vector_type = "single_vector"  # default vector type

    def embed(self, **kwargs):
        pass

    def embed_multimodal(
        self,
        inputs,
        task_name=None, input_type=None,
        **kwargs: Any,
    ):
        from PIL.Image import Image

        if 'batch_size' not in kwargs:
            kwargs['batch_size'] = self.batch_size
        texts, images = list(), list()
        for seq in inputs['data']:
            text = ''
            image = None
            for i in seq:
                if isinstance(i, str):
                    text += (i + ' ')
                elif isinstance(i, Image):
                    if image is None:
                        image = i
            texts.append(text)
            images.append(image)

        text_embeddings = self.get_text_embeddings(
            texts,
            task_name=task_name,
            input_type=input_type,
            **kwargs,
        )
        image_embeddings = self.get_image_embeddings(
            images,
            task_name=task_name,
            input_type=input_type,
            **kwargs,
        )

        if len(text_embeddings) != len(image_embeddings):
            raise ValueError(
                "The number of texts and images must have the same length"
            )
        fused_embeddings = text_embeddings + image_embeddings
        return fused_embeddings

    def embed_text(
        self,
        inputs,
        task_name=None, input_type=None,
        batch_size: int = 32,
        return_numpy=False,
        **kwargs: Any,
    ):
        sentences = list(inputs['data'])
        with torch.no_grad():
            return self.model.encode_text(
                texts=sentences,
                batch_size=batch_size,
                return_multivector=self.vector_type == "multi_vector",
                return_numpy=return_numpy,
            )

    def embed_image(
        self,
        inputs,
        task_name=None, input_type=None,
        max_pixels: int = 37788800,
        return_numpy=False,
        **kwargs: Any,
    ):
        all_images = list(inputs['data'])
        batch_size = 1
        return self.model.encode_image(
            images=all_images,
            batch_size=batch_size,
            max_pixels=max_pixels,
            return_multivector=self.vector_type == "multi_vector",
            return_numpy=return_numpy,
        )
