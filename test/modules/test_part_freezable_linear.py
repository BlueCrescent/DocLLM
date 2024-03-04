import os
from tempfile import TemporaryDirectory
from test.modules.part_freezable_linear_test_helpers import LinearConfig, LinearModel, PFLinearConfig, PFLinearModel

import pytest
import torch

from docllm.modules.part_freezable_linear import PartFreezableLinear


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
