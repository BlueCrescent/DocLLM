import torch
from pydantic import BaseModel, ConfigDict


class InputSizes(BaseModel):
    batch_size: int
    sequence_length: int


class ModelInputs(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    input_embeddings: torch.Tensor
    spatial_embeddings: torch.Tensor
