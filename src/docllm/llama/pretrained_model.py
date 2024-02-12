import torch.nn as nn
from transformers import PreTrainedModel
from transformers.utils import add_start_docstrings

from docllm.llama.config import DocLLMLlamaConfig

DOCLLM_LLAMA_START_DOCSTRING = r"""
    ...

    Parameters:
        config ([`DocLLMLlamaConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare LLaMA DocLLM Model outputting raw hidden-states without any specific head on top.",
    DOCLLM_LLAMA_START_DOCSTRING,
)
class DocLLMLlamaPreTrainedModel(PreTrainedModel):
    config_class = DocLLMLlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["DocLLMLlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values", "spatial_past_key_value"]
    _supports_flash_attn_2 = False
    _supports_sdpa = False
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
