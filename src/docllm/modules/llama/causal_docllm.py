from copy import deepcopy
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import torch
from transformers.cache_utils import Cache
from transformers.modeling_outputs import ModelOutput

from docllm.modules.llama.config import DocLLMLlamaConfig
from docllm.modules.llama.llama_docllm import LlamaDocLLM
from docllm.modules.llama.pretrained_model import DocLLMLlamaPreTrainedModel
from docllm.modules.part_freezable_linear import PartFreezableLinear
from docllm.pretraining_loss import DocLLMCrossEntropyLoss


@dataclass
class CausalDocLLMOutputWithPast(ModelOutput):
    """
    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Language modeling loss (for next-token prediction).
        logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        past_key_values (`transformers.Cache)`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        spatial_past_key_value (`transformers.Cache)`, *optional*, returned when `use_cache=True` is passed or when `config.use_cache=True`):
            Contains pre-computed hidden-states (key and values in the self-attention blocks) that can be used (see
            `past_key_values` input) to speed up sequential decoding.
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Cache] = None
    spatial_past_key_value: Optional[Cache] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class CausalLlamaDocLLM(DocLLMLlamaPreTrainedModel):
    def __init__(self, config: DocLLMLlamaConfig):
        config = deepcopy(config)
        super().__init__(config)
        self.model = LlamaDocLLM(config)
        self.vocab_size = config.vocab_size
        self.lm_head = PartFreezableLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            num_additional_outputs=config.additional_training_vocab_size,
        )

        # Initialize weights and apply final processing
        self.post_init()

    def set_freeze_llama_layers(self, freeze: bool):
        self.lm_head.requires_grad_(not freeze)
        self.model.set_freeze_llama_layers(freeze)

    @torch.no_grad()
    def init_additional_weights(self, init_func: Callable[[torch.Tensor], None] = torch.nn.init.xavier_normal_):
        self.lm_head.init_additional_outputs(init_func)
        self.model.init_additional_weights(init_func)

    @torch.no_grad()
    def fuse_additional_embeddings(self):
        self.lm_head.fuse_additional_outputs()
        self.model.fuse_additional_embeddings()
        self.config.vocab_size = self.lm_head.out_features
        self.config.additional_training_vocab_size = 0

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        input_coordinates: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        loss_mask: Optional[torch.BoolTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalDocLLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            input_coordinates=input_coordinates,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            if loss_mask is None:
                raise ValueError("loss_mask is required when labels are provided")
            loss_fct = DocLLMCrossEntropyLoss()
            labels = labels.to(logits.device)
            loss_mask = loss_mask.to(logits.device) if loss_mask is not None else None
            loss = loss_fct(logits, labels, loss_mask)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalDocLLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            spatial_past_key_value=outputs.spatial_past_key_value,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        input_coordinates: torch.FloatTensor,
        past_key_values: Optional[Cache] = None,
        spatial_past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        **kwargs,
    ):
        if input_ids.shape != input_coordinates.shape[:-1]:
            raise ValueError("input_ids and input_coordinates must have the same batch size and sequence length.")

        if spatial_past_key_values is not None and past_key_values is None:
            raise ValueError("past_key_values must be provided when spatial_past_key_values is provided.")

        if past_key_values is not None:
            if not isinstance(past_key_values, Cache):
                raise ValueError("past_key_values must be an instance of transformers.Cache")
            if spatial_past_key_values is None:
                raise ValueError("spatial_past_key_values must be provided when past_key_values is provided.")
            if not isinstance(spatial_past_key_values, Cache):
                raise ValueError("spatial_past_key_values must be an instance of transformers.Cache")
            cache_length = past_length = past_key_values[0][0].shape[2]
            spatial_cache_length = spatial_past_key_values[0][0].shape[2]
            if cache_length != spatial_cache_length:
                raise ValueError("past_key_values and spatial_past_key_values must have the same cache length.")
            max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusively passed as part of the cache (e.g. when passing input_embeds as
            # input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
                input_coordinates = input_coordinates[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
                input_coordinates = input_coordinates[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_coordinates": input_coordinates}
        else:
            model_inputs = {"input_ids": input_ids, "input_coordinates": input_coordinates}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs
