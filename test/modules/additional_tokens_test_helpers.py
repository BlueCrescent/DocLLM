import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig

from docllm.modules.additional_tokens import PartFreezableEmbedding


class EmbeddingConfig(PretrainedConfig):
    def __init__(
        self,
        num_embeddings: int = 5,
        embedding_dim: int = 11,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim


class EmbeddingModel(PreTrainedModel):
    config_class = EmbeddingConfig

    def __init__(self, config: EmbeddingConfig, **kwargs) -> None:
        super().__init__(config)
        self.emb = nn.Embedding(num_embeddings=config.num_embeddings, embedding_dim=config.embedding_dim, **kwargs)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            module.weight.data.fill_(1.0)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].fill_(1.0)


class PFEmbeddingConfig(EmbeddingConfig):
    def __init__(
        self,
        num_embeddings: int = 5,
        embedding_dim: int = 11,
        num_additional_tokens: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(num_embeddings, embedding_dim, **kwargs)
        self.num_additional_tokens = num_additional_tokens


class PFEmbeddingModel(PreTrainedModel):
    config_class = PFEmbeddingConfig

    def __init__(
        self,
        config: PFEmbeddingConfig,
        **kwargs,
    ) -> None:
        super().__init__(config)
        self.emb = PartFreezableEmbedding(
            num_embeddings=config.num_embeddings,
            embedding_dim=config.embedding_dim,
            num_additional_tokens=config.num_additional_tokens,
            **kwargs,
        )

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            module.weight.data.zero_()
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
