from typing import Tuple

import pytest
import torch

from docllm.pretraining_loss import DocLLMCrossEntropyLoss


@pytest.fixture
def batch_size() -> int:
    return 10


@pytest.fixture
def seq_len() -> int:
    return 17


@pytest.fixture
def vocab_size() -> int:
    return 7


@pytest.fixture
def loss_inputs(batch_size: int, seq_len: int, vocab_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = torch.rand(batch_size, seq_len, vocab_size)
    labels = torch.randint(vocab_size, (batch_size, seq_len))
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    return logits, labels, mask


def test_initialization():
    loss = DocLLMCrossEntropyLoss()
    assert isinstance(loss, DocLLMCrossEntropyLoss)


def test_forward_creates_tensor(loss_inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
    loss = DocLLMCrossEntropyLoss()
    logits, labels, mask = loss_inputs
    result = loss(logits, labels, mask)
    assert isinstance(result, torch.Tensor)


def test_forward_output_is_single_value(loss_inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
    loss = DocLLMCrossEntropyLoss()
    logits, labels, mask = loss_inputs
    result = loss(logits, labels, mask)
    assert result.size() == torch.Size([])


def test_with_mask_all_zero_loss_is_zero(loss_inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
    loss = DocLLMCrossEntropyLoss()
    logits, labels, mask = loss_inputs
    mask = torch.zeros_like(mask, dtype=torch.bool)
    result = loss(logits, labels, mask)
    assert result.item() == 0.0


def test_with_mask_all_one_loss_is_not_zero(loss_inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
    loss = DocLLMCrossEntropyLoss()
    logits, labels, mask = loss_inputs
    # Ensure loss is not actually zero.
    logits[0][0].fill_(value=-100000.0)
    logits[0][0][0] = 0.0
    labels[0][0] = 1
    result = loss(logits, labels, mask)
    assert result.item() != pytest.approx(0.0, 0.1)


@pytest.mark.parametrize("correct_index", ((0, 0), (1, 3), (9, 15)))
def test_loss_is_zero_with_mask_only_one_at_correct_prediction(
    loss_inputs: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], correct_index: Tuple[int, int]
):
    batch_idx, seq_idx = correct_index
    loss = DocLLMCrossEntropyLoss()
    logits, labels, mask = loss_inputs
    # Values that cause a loss of zero.
    logits[batch_idx][seq_idx].fill_(value=-10000.0)
    logits[batch_idx][seq_idx][labels[batch_idx][seq_idx]] = 0.0
    # Use mask to filter out all other values.
    mask = torch.zeros_like(mask, dtype=torch.bool)
    mask[batch_idx][seq_idx] = 1
    result = loss(logits, labels, mask)
    assert result.item() == pytest.approx(0.0, 0.00001)
