from enum import StrEnum

import torch
import torch.nn as nn


class SpatialEmbeddingType(StrEnum):
    PROJECTION = "projection"
    EMBED = "embed"


class SpatialEmbedder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        embedding_type: SpatialEmbeddingType,
        include_width_height: bool,
        max_coord: int = 1000,
    ):
        super().__init__()
        self._embedding_type = embedding_type
        self._include_width_height = include_width_height
        self._num_coords = max_coord + 1

        self._setup(hidden_size)

    def forward(self, input_coordinates: torch.FloatTensor) -> torch.FloatTensor:
        if self._include_width_height:
            input_coordinates = self._add_width_and_height(input_coordinates)

        if self._embedding_type == SpatialEmbeddingType.PROJECTION:
            return self.proj(input_coordinates)

        input_coordinates = self._scale(input_coordinates)

        embedded = [
            self.embed_x(input_coordinates[:, :, 0]),
            self.embed_y(input_coordinates[:, :, 1]),
            self.embed_x(input_coordinates[:, :, 2]),
            self.embed_y(input_coordinates[:, :, 3]),
        ]
        if self._include_width_height:
            embedded += [
                self.embed_h(input_coordinates[:, :, 4]),
                self.embed_w(input_coordinates[:, :, 5]),
            ]
        return torch.cat(embedded, dim=-1)

    def _setup(self, hidden_size: int):
        num_feat = 6 if self._include_width_height else 4

        match self._embedding_type:
            case SpatialEmbeddingType.PROJECTION:
                self.proj = nn.Linear(num_feat, hidden_size, bias=False)
            case SpatialEmbeddingType.EMBED:
                self._setup_dimension_embeddings(hidden_size, num_feat)

    def _setup_dimension_embeddings(self, hidden_size: int, num_feat: int):
        if hidden_size % num_feat != 0:
            raise Exception(f"Hidden size must be divisible by {num_feat} for spatial embeddings.")
        self.embed_x = nn.Embedding(self._num_coords, hidden_size // num_feat)
        self.embed_y = nn.Embedding(self._num_coords, hidden_size // num_feat)
        if self._include_width_height:
            self.embed_h = nn.Embedding(self._num_coords, hidden_size // num_feat)
            self.embed_w = nn.Embedding(self._num_coords, hidden_size // num_feat)

    def _add_width_and_height(self, input_coordinates: torch.Tensor) -> torch.Tensor:
        heights = (input_coordinates[:, :, 3] - input_coordinates[:, :, 1]).unsqueeze(2)
        widths = (input_coordinates[:, :, 2] - input_coordinates[:, :, 0]).unsqueeze(2)
        return torch.cat((input_coordinates, heights, widths), dim=-1)

    def _scale(self, input_coordinates: torch.FloatTensor) -> torch.LongTensor:
        max_coord = self._num_coords - 1
        return torch.round(input_coordinates * max_coord).int()
