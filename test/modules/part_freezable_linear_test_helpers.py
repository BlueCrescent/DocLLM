import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel

from docllm.modules.part_freezable_linear import PartFreezableLinear


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
