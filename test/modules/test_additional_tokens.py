import os
from tempfile import TemporaryDirectory
from test.modules.additional_tokens_test_helpers import (
    EmbeddingConfig,
    EmbeddingModel,
    LinearConfig,
    LinearModel,
    PFEmbeddingConfig,
    PFEmbeddingModel,
    PFLinearConfig,
    PFLinearModel,
)

import pytest
import torch

from docllm.modules.additional_tokens import PartFreezableEmbedding, PartFreezableLinear


@pytest.fixture
def num_embeddings() -> int:
    return 5


@pytest.fixture
def embedding_dim() -> int:
    return 11


@pytest.fixture
def num_additional_tokens(request) -> int:
    if not hasattr(request, "param") or request.param is None:
        return 3
    return request.param


@pytest.fixture
def part_freezable_emb(
    num_embeddings,
    embedding_dim,
    num_additional_tokens,
) -> PartFreezableEmbedding:
    return PartFreezableEmbedding(
        num_embeddings=num_embeddings,
        embedding_dim=embedding_dim,
        num_additional_tokens=num_additional_tokens,
    )


def test_part_freezable_embeddings_init(part_freezable_emb: PartFreezableEmbedding):
    assert isinstance(part_freezable_emb, PartFreezableEmbedding)


@pytest.mark.parametrize("num_additional_tokens", (0,), indirect=True)
def test_part_freezable_embeddings_returns_expected_shape_for_original_token_without_add_tokens(
    part_freezable_emb: PartFreezableEmbedding, num_embeddings: int, embedding_dim: int
):
    input = torch.randint(0, num_embeddings, (2, 3))
    token_emb = part_freezable_emb(input)
    expected_size = tuple(input.shape) + (embedding_dim,)
    assert token_emb.shape == expected_size


def test_part_freezable_embeddings_returns_expected_shape_for_original_token_with_add_tokens(
    part_freezable_emb: PartFreezableEmbedding, num_embeddings: int, embedding_dim: int
):
    input = torch.randint(0, num_embeddings, (2, 3))
    token_emb = part_freezable_emb(input)
    expected_size = tuple(input.shape) + (embedding_dim,)
    assert token_emb.shape == expected_size


def test_part_freezable_embeddings_returns_expected_shape_for_new_token(
    part_freezable_emb: PartFreezableEmbedding, num_embeddings: int, embedding_dim: int, num_additional_tokens: int
):
    input = torch.randint(num_embeddings, num_embeddings + num_additional_tokens, (2, 3))
    token_emb = part_freezable_emb(input)
    expected_size = tuple(input.shape) + (embedding_dim,)
    assert token_emb.shape == expected_size


@pytest.mark.parametrize("num_additional_tokens", (0,), indirect=True)
def test_part_freezable_embeddings_backward_for_original_token_without_add_tokens(
    part_freezable_emb: PartFreezableEmbedding, num_embeddings: int
):
    input = torch.randint(0, num_embeddings, (2, 3))
    token_emb = part_freezable_emb(input)
    loss = token_emb.sum()
    loss.backward()


def test_part_freezable_embeddings_backward_for_original_token_with_add_tokens(
    part_freezable_emb: PartFreezableEmbedding, num_embeddings: int
):
    input = torch.randint(0, num_embeddings, (2, 3))
    token_emb = part_freezable_emb(input)
    loss = token_emb.sum()
    loss.backward()


def test_part_freezable_embeddings_backward_for_new_token(
    part_freezable_emb: PartFreezableEmbedding, num_embeddings: int, num_additional_tokens: int
):
    input = torch.randint(num_embeddings, num_embeddings + num_additional_tokens, (2, 3))
    token_emb = part_freezable_emb(input)
    loss = token_emb.sum()
    loss.backward()


def test_after_freezing_original_embs_additional_ones_are_not_frozen(part_freezable_emb: PartFreezableEmbedding):
    part_freezable_emb.set_freeze_original_embeddings(True)
    for name, param in part_freezable_emb.named_parameters(recurse=True):
        assert (not param.requires_grad) ^ ("additional" in name)


def test_after_unfreezing_original_embs_everything_is_not_frozen(part_freezable_emb: PartFreezableEmbedding):
    part_freezable_emb.set_freeze_original_embeddings(True)
    part_freezable_emb.set_freeze_original_embeddings(False)
    for param in part_freezable_emb.parameters(recurse=True):
        assert param.requires_grad


def test_loading_part_freezable_embeddings_loads_normal_embeddings():
    original_model = EmbeddingModel(EmbeddingConfig())
    original_model.init_weights()
    with TemporaryDirectory() as dir:
        model_path = os.path.join(dir, "embeddings")
        original_model.save_pretrained(model_path)
        model = PFEmbeddingModel.from_pretrained(model_path, config=PFEmbeddingConfig(num_additional_tokens=3))
        for name, param in model.named_parameters(recurse=True):
            assert (param == 1.0).all().item() ^ ("additional" in name)


def test_loading_part_freezable_embeddings_does_set_additional_weights():
    original_model = EmbeddingModel(EmbeddingConfig())
    original_model.init_weights()
    with TemporaryDirectory() as dir:
        model_path = os.path.join(dir, "embeddings")
        original_model.save_pretrained(model_path)
        model = PFEmbeddingModel.from_pretrained(model_path, config=PFEmbeddingConfig(num_additional_tokens=3))
        for name, param in model.named_parameters(recurse=True):
            assert (param == 0.0).all().item() ^ ("additional" not in name)


def test_additional_embeddings_are_in_parameters():
    model = PFEmbeddingModel(PFEmbeddingConfig(num_additional_tokens=3))
    assert any("additional" in name for name, _ in model.named_parameters(recurse=True))


def test_additional_tokens_cause_expected_additional_num_trainable_parameter():
    config = PFEmbeddingConfig(num_additional_tokens=3)
    original_model_num_params = EmbeddingModel(config).num_parameters()
    new_num_params = PFEmbeddingModel(config).num_parameters()
    assert new_num_params - original_model_num_params == config.num_additional_tokens * config.embedding_dim


def test_freezing_original_embedding_weights_leaves_remaining_num_additional_trainable_parameter():
    config = PFEmbeddingConfig(num_additional_tokens=3)
    model = PFEmbeddingModel(config)
    model.emb.set_freeze_original_embeddings(True)
    new_num_params = model.num_parameters(only_trainable=True)
    assert new_num_params == config.num_additional_tokens * config.embedding_dim


@pytest.fixture
def in_features() -> int:
    return 11


@pytest.fixture
def out_features() -> int:
    return 5


@pytest.fixture
def num_additional_outputs(request) -> int:
    if not hasattr(request, "param") or request.param is None:
        return 3
    return request.param


@pytest.fixture
def part_freezable_linear(
    in_features,
    out_features,
    num_additional_outputs,
) -> PartFreezableLinear:
    return PartFreezableLinear(
        in_features=in_features,
        out_features=out_features,
        num_additional_outputs=num_additional_outputs,
    )


def test_init_part_freezable_linear(part_freezable_linear: PartFreezableLinear):
    assert isinstance(part_freezable_linear, PartFreezableLinear)


@pytest.mark.parametrize("num_additional_outputs", (0,), indirect=True)
def test_part_freezable_linear_returns_expected_shape_without_additional_outputs(
    part_freezable_linear: PartFreezableLinear, in_features: int, out_features: int
):
    input = torch.rand((2, 3, in_features))
    token_emb = part_freezable_linear(input)
    expected_size = tuple(input.shape[:-1]) + (out_features,)
    assert token_emb.shape == expected_size


def test_part_freezable_linear_returns_expected_shape_with_additional_outputs(
    part_freezable_linear: PartFreezableLinear, in_features: int, out_features: int, num_additional_outputs: int
):
    input = torch.rand((2, 3, in_features))
    token_emb = part_freezable_linear(input)
    expected_size = tuple(input.shape[:-1]) + (out_features + num_additional_outputs,)
    assert token_emb.shape == expected_size


@pytest.mark.parametrize("num_additional_outputs", (0,), indirect=True)
def test_part_freezable_linear_backward_without_additional_outputs(
    part_freezable_linear: PartFreezableLinear, in_features: int
):
    input = torch.rand((2, 3, in_features))
    output = part_freezable_linear(input)
    loss = output.sum()
    loss.backward()


def test_part_freezable_linear_backward_with_additional_outputs(
    part_freezable_linear: PartFreezableLinear, in_features: int
):
    input = torch.rand((2, 3, in_features))
    output = part_freezable_linear(input)
    loss = output.sum()
    loss.backward()


def test_after_freezing_original_linear_layer_additional_outputs_are_not_frozen(
    part_freezable_linear: PartFreezableLinear,
):
    part_freezable_linear.set_freeze_original_outputs(True)
    for name, param in part_freezable_linear.named_parameters(recurse=True):
        assert (not param.requires_grad) ^ ("additional" in name)


def test_after_unfreezing_original_linear_layer_everything_is_not_frozen(part_freezable_linear: PartFreezableLinear):
    part_freezable_linear.set_freeze_original_outputs(True)
    part_freezable_linear.set_freeze_original_outputs(False)
    for param in part_freezable_linear.parameters(recurse=True):
        assert param.requires_grad


def test_loading_part_freezable_linear_loads_normal_embeddings():
    original_model = LinearModel(LinearConfig())
    original_model.init_weights()
    with TemporaryDirectory() as dir:
        model_path = os.path.join(dir, "linear")
        original_model.save_pretrained(model_path)
        model = PFLinearModel.from_pretrained(model_path, config=PFLinearConfig(num_additional_outputs=3))
        for name, param in model.named_parameters(recurse=True):
            assert (param == 1.0).all().item() ^ ("additional" in name)


def test_loading_part_freezable_linear_does_set_additional_weights():
    original_model = LinearModel(LinearConfig())
    original_model.init_weights()
    with TemporaryDirectory() as dir:
        model_path = os.path.join(dir, "linear")
        original_model.save_pretrained(model_path)
        model = PFLinearModel.from_pretrained(model_path, config=PFLinearConfig(num_additional_outputs=3))
        for name, param in model.named_parameters(recurse=True):
            assert (param == 0.0).all().item() ^ ("additional" not in name)


def test_additional_outputs_are_in_parameters():
    model = PFLinearModel(PFLinearConfig(num_additional_outputs=3))
    assert any("additional" in name for name, _ in model.named_parameters(recurse=True))


def test_additional_outputs_cause_expected_additional_num_trainable_parameter():
    config = PFLinearConfig(num_additional_outputs=3)
    original_model_num_params = LinearModel(config).num_parameters()
    new_num_params = PFLinearModel(config).num_parameters()
    assert new_num_params - original_model_num_params == (config.in_features + 1) * config.num_additional_outputs


def test_freezing_original_linear_weights_leaves_remaining_num_additional_trainable_parameter():
    config = PFLinearConfig(num_additional_outputs=3)
    model = PFLinearModel(config)
    model.linear.set_freeze_original_outputs(True)
    new_num_params = model.num_parameters(only_trainable=True)
    assert new_num_params == (config.in_features + 1) * config.num_additional_outputs
