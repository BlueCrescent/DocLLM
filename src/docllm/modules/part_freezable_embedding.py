from typing import Callable

import torch
import torch.nn as nn


class PartFreezableEmbedding(nn.Embedding):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int | None = None,
        max_norm: float | None = None,
        norm_type: float = 2,
        scale_grad_by_freq: bool = False,
        sparse: bool = False,
        _weight: torch.Tensor | None = None,
        _freeze: bool = False,
        device=None,
        dtype=None,
        num_additional_tokens: int = 0,
    ) -> None:
        super().__init__(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
            max_norm=max_norm,
            norm_type=norm_type,
            scale_grad_by_freq=scale_grad_by_freq,
            sparse=sparse,
            _weight=_weight,
            _freeze=_freeze,
            device=device,
            dtype=dtype,
        )
        self._num_additional_tokens = num_additional_tokens
        if self._num_additional_tokens > 0:
            self.additional_embeddings = nn.Embedding(
                num_embeddings=self._num_additional_tokens,
                embedding_dim=embedding_dim,
                padding_idx=padding_idx,
                max_norm=max_norm,
                norm_type=norm_type,
                scale_grad_by_freq=scale_grad_by_freq,
                sparse=sparse,
                _weight=_weight,
                _freeze=_freeze,
                device=device,
                dtype=dtype,
            )

    def set_freeze_original_embeddings(self, freeze: bool):
        self.requires_grad_(not freeze)
        if self._num_additional_tokens > 0:
            self.additional_embeddings.requires_grad_(True)

    @torch.no_grad()
    def init_additional_embeddings(self, init_func: Callable[[torch.Tensor], None] = torch.nn.init.xavier_normal_):
        if self._num_additional_tokens > 0:
            init_func(self.additional_embeddings.weight)

    @torch.no_grad()
    def fuse_additional_embeddings(self):
        if self._num_additional_tokens > 0:
            self.weight = nn.Parameter(torch.cat([self.weight, self.additional_embeddings.weight], dim=0))
            self.num_embeddings += self._num_additional_tokens
            self._num_additional_tokens = 0
            del self.additional_embeddings

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = input.clone()
        add_token_mask = input >= self.num_embeddings
        input[add_token_mask] -= self.num_embeddings
        embeddings = super().forward(input)
        if self._num_additional_tokens > 0:
            add_embeddings = self.additional_embeddings(input[add_token_mask])
            embeddings.view(-1, self.embedding_dim)[add_token_mask.view(-1)] = add_embeddings
        return embeddings
