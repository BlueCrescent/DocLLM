from enum import StrEnum

from transformers import LlamaConfig

from docllm.modules.spatial_embedder import SpatialEmbeddingType


class PositionalEmbeddingMode(StrEnum):
    NONE = "none"
    TEXT_ONLY = "text_only"
    TEXT_AND_SPATIAL = "text_and_spatial"


class DocLLMLlamaConfig(LlamaConfig):

    def __init__(
        self,
        vocab_size=32000,
        additional_training_vocab_size=0,
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=None,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=False,
        pad_token_id=None,
        bos_token_id=1,
        eos_token_id=2,
        pretraining_tp=1,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        attention_dropout=0.0,
        lambda_ts: float = 1.0,
        lambda_st: float = 1.0,
        lambda_ss: float = 1.0,
        positional_embedding_mode: PositionalEmbeddingMode = PositionalEmbeddingMode.TEXT_AND_SPATIAL,
        embedding_type: SpatialEmbeddingType = SpatialEmbeddingType.PROJECTION,
        embed_include_width_height: bool = False,
        embed_max_coord: int = 1000,
        _attn_implementation: str = "eager",
        **kwargs,
    ):
        super().__init__(
            vocab_size,
            hidden_size,
            intermediate_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            hidden_act,
            max_position_embeddings,
            initializer_range,
            rms_norm_eps,
            use_cache,
            pad_token_id,
            bos_token_id,
            eos_token_id,
            pretraining_tp,
            tie_word_embeddings,
            rope_theta,
            rope_scaling,
            attention_bias,
            attention_dropout,
            _attn_implementation=_attn_implementation,
            **kwargs,
        )
        self.additional_training_vocab_size = additional_training_vocab_size
        self.lambda_ts = lambda_ts
        self.lambda_st = lambda_st
        self.lambda_ss = lambda_ss
        self.positional_embedding_mode = positional_embedding_mode
        self.embedding_type = embedding_type
        self.embed_include_width_height = embed_include_width_height
        self.embed_max_coord = embed_max_coord

        if self.pretraining_tp > 1:
            raise NotImplementedError()
