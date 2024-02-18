import logging
import math
from typing import Iterable, List, Tuple

import torch
from torch.utils.data import IterDataPipe, functional_datapipe

from docllm.data.pretraining_data_config import DocLLMPreTrainDataConfig


@functional_datapipe("build_docllm_train_data")
class DocLLMTrainDataPipe(IterDataPipe):
    def __init__(self, source_datapipe: IterDataPipe, config: DocLLMPreTrainDataConfig) -> None:
        self._source_datapipe = source_datapipe
        self._config = config.model_copy()
        self._get_num_masks = self._config.num_masked_blocks_callable
        self._mask_text_token = torch.tensor([self._config.mask_text_token])
        self._mask_bbox_token = torch.tensor([self._config.mask_bbox_token])
        self._block_start_text_token = torch.tensor([self._config.block_start_text_token])
        self._bos_text_token = [torch.tensor([self._config.bos_text_token])] if self._config.bos_text_token else []
        self._bos_bbox_token = [torch.tensor([self._config.bos_bbox_token])] if self._config.bos_bbox_token else []

    def __iter__(self) -> Iterable[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        for input_tensors, bbox_tensors in self._source_datapipe:
            yield from self._try_build_inputs(input_tensors, bbox_tensors)

    def _try_build_inputs(
        self, input_tensors: List[torch.LongTensor], bbox_tensors: List[torch.FloatTensor]
    ) -> Iterable[Tuple[torch.LongTensor, torch.FloatTensor, torch.BoolTensor]]:
        try:
            num_blocks = len(input_tensors)
            num_masks = self._get_valid_nun_masks(num_blocks)
            mask_indices = self._get_mask_indices(num_blocks, num_masks)
            yield self._create_masked_input(input_tensors, bbox_tensors, mask_indices)
        except ValueError as e:
            logging.warning(e)

    def _get_valid_nun_masks(self, num_blocks):
        num_masks = self._get_num_masks(num_blocks)
        if num_masks > (max_num_masks := math.ceil(num_blocks * self._config.max_percentage_masked_blocks)):
            logging.warning(
                f"Number of masks ({num_masks}) cannot exceed maximal allowed number of "
                f"blocks (ceil({num_blocks} * {self._config.max_percentage_masked_blocks}))."
            )
            num_masks = max_num_masks
        if num_masks == 0:
            raise ValueError("Number of masks cannot be zero.")
        return num_masks

    def _get_mask_indices(self, num_blocks: int, num_masks: int) -> List[int]:
        # FIXME: Should we prevent adjacent blocks from being masked?
        mask_indices = torch.multinomial(torch.ones(num_blocks), num_masks, replacement=False)
        return sorted(mask_indices)

    def _create_masked_input(
        self, input_tensors: List[torch.LongTensor], bbox_tensors: List[torch.FloatTensor], mask_indices: List[int]
    ) -> Tuple[torch.LongTensor, torch.FloatTensor, torch.BoolTensor]:
        input_tensors, bbox_tensors, target_tokens = self._extract_masked_tokens(
            input_tensors, bbox_tensors, mask_indices
        )
        num_target_tokens = sum(tensor.shape[0] for tensor in target_tokens)
        text_inputs = torch.cat(self._bos_text_token + input_tensors + target_tokens)
        target_bbox_tokens = [self._mask_bbox_token] * num_target_tokens
        spatial_inputs = torch.cat(self._bos_bbox_token + bbox_tensors + target_bbox_tokens)
        loss_mask = torch.zeros_like(text_inputs, dtype=torch.bool)
        # The first block start token [S] is not used for loss computation.
        loss_mask[-(num_target_tokens - 1) :] = 1
        return self._truncate_to_max_seq_len(text_inputs, spatial_inputs, loss_mask, num_target_tokens)

    def _extract_masked_tokens(
        self, input_tensors: List[torch.LongTensor], bbox_tensors: List[torch.FloatTensor], mask_indices: List[int]
    ) -> Tuple[List[torch.LongTensor], List[torch.FloatTensor], List[torch.BoolTensor]]:
        target_tokens = []
        for i in mask_indices:
            target_tokens.append(self._block_start_text_token)
            target_tokens.append(input_tensors[i])
            input_tensors[i] = self._mask_text_token
            bbox_tensors[i] = self._mask_bbox_token
        return input_tensors, bbox_tensors, target_tokens

    def _truncate_to_max_seq_len(
        self,
        text_inputs: torch.LongTensor,
        spatial_inputs: torch.FloatTensor,
        loss_mask: torch.BoolTensor,
        num_target_tokens: int,
    ) -> Tuple[torch.LongTensor, torch.FloatTensor, torch.BoolTensor]:
        if text_inputs.size(0) <= self._config.max_seq_len:
            return text_inputs, spatial_inputs, loss_mask
        if text_inputs.size(0) - num_target_tokens + 1 >= self._config.max_seq_len:
            raise ValueError(
                f"Number of tokens ({text_inputs.size(0)}) exceeds maximal sequence length."
                " Nothing would be learned."
            )
        return (
            text_inputs[: self._config.max_seq_len],
            spatial_inputs[: self._config.max_seq_len],
            loss_mask[: self._config.max_seq_len],
        )
