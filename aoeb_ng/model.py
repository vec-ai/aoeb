from abc import ABC, abstractmethod

import torch


class EmbedderProtocol(ABC):
    batch_size: int = 32

    @abstractmethod
    def embed(
        self, inputs, prompt=None, input_type=None, task_name=None, **kwargs
    ) -> torch.Tensor:
        raise NotImplementedError

    # @abstractmethod
    # def start(self):
    #     raise NotImplementedError


class MultimodalEmbedderProtocol(EmbedderProtocol):

    @abstractmethod
    def embed_text(
        self, inputs, prompt=None, input_type=None, task_name=None, **kwargs
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def embed_image(
        self, inputs, prompt=None, input_type=None, task_name=None, **kwargs
    ) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def embed_multimodal(
        self, inputs, prompt=None, input_type=None, task_name=None, **kwargs
    ) -> torch.Tensor:
        raise NotImplementedError


class SentenceTransformerEmbedder(EmbedderProtocol):
    def __init__(self, model_path, **kwargs):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_path, **kwargs)

    def embed(self, inputs, task_name=None, input_type=None, **kwargs) -> torch.Tensor:
        texts = list(inputs['data'])
        if 'batch_size' not in kwargs:
            kwargs['batch_size'] = self.batch_size
        return self.model.encode(texts, convert_to_tensor=True, **kwargs)


class SentenceTransformerMultimodalEmbedder(MultimodalEmbedderProtocol):
    def __init__(self, model_path, **kwargs):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_path, **kwargs)

    def embed_text(
        self, inputs, task_name=None, input_type=None, **kwargs
    ) -> torch.Tensor:
        if 'batch_size' not in kwargs:
            kwargs['batch_size'] = self.batch_size
        texts = list()
        for item in inputs['data']:
            texts.append(dict(text=item))
        return self.model.encode(texts, convert_to_tensor=True, **kwargs)

    def embed_image(
        self, inputs, task_name=None, input_type=None, **kwargs
    ) -> torch.Tensor:
        if 'batch_size' not in kwargs:
            kwargs['batch_size'] = self.batch_size
        images = list()
        for item in inputs['data']:
            images.append(dict(image=item))
        return self.model.encode(images, convert_to_tensor=True, **kwargs)

    def embed_multimodal(
        self, inputs: list[dict], task_name=None, input_type=None, **kwargs
    ) -> torch.Tensor:
        from PIL.Image import Image

        if 'batch_size' not in kwargs:
            kwargs['batch_size'] = self.batch_size
        data = list()
        for seq in inputs['data']:
            ins = dict()
            text = ''
            for i in seq:
                if isinstance(i, str):
                    text += (i + ' ')
                elif isinstance(i, Image):
                    if 'image' not in ins:
                        ins['image'] = i
            data.append(ins)
        return self.model.encode(data, convert_to_tensor=True, **kwargs)
