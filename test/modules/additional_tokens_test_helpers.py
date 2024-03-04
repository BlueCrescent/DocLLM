import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel

from docllm.modules.additional_tokens import PartFreezableEmbedding, PartFreezableLinear


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


class LinearConfig(PretrainedConfig):
    def __init__(
        self,
        in_features: int = 5,
        out_features: int = 11,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features


class LinearModel(PreTrainedModel):
    config_class = LinearConfig

    def __init__(self, config: LinearConfig, **kwargs) -> None:
        super().__init__(config)
        self.linear = nn.Linear(in_features=config.in_features, out_features=config.out_features, **kwargs)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.fill_(1.0)
            if module.bias is not None:
                module.bias.data.fill_(1.0)


class PFLinearConfig(LinearConfig):
    def __init__(
        self,
        in_features: int = 5,
        out_features: int = 11,
        num_additional_outputs: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(in_features, out_features, **kwargs)
        self.num_additional_outputs = num_additional_outputs


class PFLinearModel(PreTrainedModel):
    config_class = PFLinearConfig

    def __init__(
        self,
        config: PFLinearConfig,
        **kwargs,
    ) -> None:
        super().__init__(config)
        self.linear = PartFreezableLinear(
            in_features=config.in_features,
            out_features=config.out_features,
            num_additional_outputs=config.num_additional_outputs,
            **kwargs,
        )

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.zero_()
            if module.bias is not None:
                module.bias.data.zero_()
