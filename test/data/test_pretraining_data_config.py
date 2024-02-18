import math

from docllm.data import DocLLMPreTrainDataConfig
from docllm.data.pretraining_data_config import NumMaskedBlocksType


def test_instantiation(pretraining_config: DocLLMPreTrainDataConfig):
    assert isinstance(pretraining_config, DocLLMPreTrainDataConfig)


def test_num_masked_blocks_callable_callable_produces_values_defined_in_config(
    pretraining_config: DocLLMPreTrainDataConfig, num_blocks: int
):
    num_masked_blocks_callable = pretraining_config.num_masked_blocks_callable
    if isinstance(pretraining_config.num_masked_blocks, int):
        assert num_masked_blocks_callable(num_blocks) == pretraining_config.num_masked_blocks
    elif isinstance(pretraining_config.num_masked_blocks, float):
        assert num_masked_blocks_callable(num_blocks) == math.ceil(num_blocks * pretraining_config.num_masked_blocks)
    elif isinstance(pretraining_config.num_masked_blocks, tuple):
        lower = pretraining_config.num_masked_blocks[0]
        upper = pretraining_config.num_masked_blocks[1] - 1
        if isinstance(pretraining_config.num_masked_blocks[0], float):
            lower = math.ceil(num_blocks * lower)
            upper = math.ceil(num_blocks * (upper + 1))
        for _ in range(10):
            assert lower <= num_masked_blocks_callable(num_blocks) <= upper
    else:
        raise ValueError("Invalid num_masked_blocks type")
