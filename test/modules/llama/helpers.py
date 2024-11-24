from typing import Tuple

import torch
from pydantic import BaseModel, ConfigDict
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding


class InputSizes(BaseModel):
    batch_size: int
    sequence_length: int


class ModelInputs(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    input_embeddings: torch.Tensor
    spatial_embeddings: torch.Tensor

    def compute_positional_embeddings(self, config: LlamaConfig) -> Tuple[torch.Tensor, torch.Tensor]:
        cache_position = torch.arange(0, self.input_embeddings.shape[1], device=self.input_embeddings.device)
        position_ids = cache_position.unsqueeze(0)
        rotary_emb = LlamaRotaryEmbedding(config=config)
        position_embeddings = rotary_emb(self.input_embeddings, position_ids)
        return position_embeddings
