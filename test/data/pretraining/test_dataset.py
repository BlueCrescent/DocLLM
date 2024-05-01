from typing import List, Tuple
from unittest.mock import MagicMock

import pytest
import torch

from docllm.data.pretraining.config import DocLLMPreTrainDataConfig
from docllm.data.pretraining.dataset import DocLLMPretrainDataset
from docllm.data.pretraining.masker import DocLLMPretrainingMasker


def test_initialization(dataset: DocLLMPretrainDataset):
    assert isinstance(dataset, DocLLMPretrainDataset)


def test_has_expected_length(
    dataset: DocLLMPretrainDataset, multiple_docs_test_data: List[List[Tuple[torch.LongTensor, torch.FloatTensor]]]
):
    assert len(dataset) == len(multiple_docs_test_data)


def test_produces_expected_number_of_tensors(dataset: DocLLMPretrainDataset):
    for i in range(len(dataset)):
        assert len(dataset[i]) == 4


def test_produces_data_with_expected_length(
    dataset: DocLLMPretrainDataset,
    multiple_docs_test_data: List[List[Tuple[torch.LongTensor, torch.FloatTensor]]],
    dummy_masker: DocLLMPretrainingMasker,
):
    dataset._masker = dummy_masker
    for i in range(len(dataset)):
        inputs, bboxes, _, _ = dataset[i]
        expected_inputs, expected_bboxes = zip(*multiple_docs_test_data[i])
        assert len(inputs) == sum(len(block) for block in expected_inputs)
        assert len(bboxes) == sum(len(block) for block in expected_bboxes)


def test_produces_expected_data(
    dataset: DocLLMPretrainDataset,
    multiple_docs_test_data: List[List[Tuple[torch.LongTensor, torch.FloatTensor]]],
    dummy_masker: DocLLMPretrainingMasker,
):
    dataset._masker = dummy_masker
    for i in range(len(dataset)):
        inputs, bboxes, _, _ = dataset[i]
        expected_inputs, expected_bboxes = zip(*multiple_docs_test_data[i])
        assert (inputs == torch.cat(expected_inputs)).all()
        assert (bboxes == torch.cat(expected_bboxes)).all()


def test_input_tokens_have_expected_type(dataset: DocLLMPretrainDataset):
    for i in range(len(dataset)):
        inputs, _, _, _ = dataset[i]
        assert isinstance(inputs, torch.LongTensor)


def test_input_bboxes_have_expected_type(dataset: DocLLMPretrainDataset):
    for i in range(len(dataset)):
        _, bboxes, _, _ = dataset[i]
        assert isinstance(bboxes, torch.FloatTensor)


def test_loss_mask_has_expected_type(dataset: DocLLMPretrainDataset):
    for i in range(len(dataset)):
        _, _, mask, _ = dataset[i]
        assert isinstance(mask, torch.BoolTensor)


def test_labels_have_expected_type(dataset: DocLLMPretrainDataset):
    for i in range(len(dataset)):
        _, _, _, labels = dataset[i]
        assert isinstance(labels, torch.LongTensor)


def test_input_tokens_have_expected_shape(dataset: DocLLMPretrainDataset):
    for i in range(len(dataset)):
        inputs, _, _, _ = dataset[i]
        assert len(inputs.shape) == 1


def test_input_bboxes_have_expected_shape(dataset: DocLLMPretrainDataset):
    for i in range(len(dataset)):
        _, bboxes, _, _ = dataset[i]
        assert len(bboxes.shape) == 2
        assert bboxes.size(1) == 4


def test_loss_mask_has_expected_shape(dataset: DocLLMPretrainDataset):
    for i in range(len(dataset)):
        _, _, mask, _ = dataset[i]
        assert len(mask.shape) == 1


def test_labels_have_expected_shape(dataset: DocLLMPretrainDataset):
    for i in range(len(dataset)):
        _, _, _, labels = dataset[i]
        assert len(labels.shape) == 1


def test_all_sequence_lengths_match(dataset: DocLLMPretrainDataset):
    for i in range(len(dataset)):
        inputs, bboxes, mask, labels = dataset[i]
        assert len(inputs) == len(bboxes) == len(mask) == len(labels)


@pytest.fixture
def dataset(pretraining_config_with_input_dir: DocLLMPreTrainDataConfig) -> DocLLMPretrainDataset:
    return DocLLMPretrainDataset(pretraining_config_with_input_dir)


@pytest.fixture
def dummy_masker() -> DocLLMPretrainingMasker:
    def no_mask(
        input_tensors: List[torch.LongTensor], bbox_tensors: List[torch.FloatTensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.BoolTensor, torch.LongTensor]:
        inputs = torch.cat(input_tensors)
        bboxes = torch.cat(bbox_tensors)
        loss_mask = torch.ones_like(inputs, dtype=torch.bool)
        labels = torch.randint(0, 10, (inputs.size(0),), dtype=torch.long)
        return inputs, bboxes, loss_mask, labels

    masker = MagicMock(spec=DocLLMPretrainingMasker)
    masker.side_effect = no_mask
    return masker
