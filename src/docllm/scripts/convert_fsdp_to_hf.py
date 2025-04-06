import argparse
import os

import torch.distributed._shard.checkpoint as dist_cp

from docllm.modules.llama.causal_docllm import CausalLlamaDocLLM
from docllm.modules.llama.config import DocLLMLlamaConfig


def convert_fsdp_to_hf(fsdp_checkpoint_dir: str, hf_checkpoint_dir: str, base_model_name: str) -> None:
    model_config: DocLLMLlamaConfig = DocLLMLlamaConfig.from_pretrained(
        base_model_name, additional_training_vocab_size=3, use_cache=False, _attn_implementation="sdpa"
    )
    model: CausalLlamaDocLLM = CausalLlamaDocLLM.from_pretrained(base_model_name, config=model_config)
    model_state_dict = {"model": model.state_dict()}
    dist_cp.load(state_dict=model_state_dict, storage_reader=dist_cp.FileSystemReader(fsdp_checkpoint_dir))
    model_state_dict = model_state_dict["model"]
    model.load_state_dict(model_state_dict)
    model.save_pretrained(os.path.join(hf_checkpoint_dir))


def main():
    args = argparse.ArgumentParser(description="Convert FSDP checkpoint to HF format")
    args.add_argument("--fsdp_checkpoint_dir", type=str, required=True, help="Directory containing the FSDP checkpoint")
    args.add_argument("--hf_checkpoint_dir", type=str, required=True, help="Directory to save the HF checkpoint")
    args.add_argument("--base_model_name", type=str, required=True, help="Base model name for configuration")
    args = args.parse_args()
    fsdp_checkpoint_dir = args.fsdp_checkpoint_dir
    hf_checkpoint_dir = args.hf_checkpoint_dir
    base_model_name = args.base_model_name
    convert_fsdp_to_hf(fsdp_checkpoint_dir, hf_checkpoint_dir, base_model_name)
    print(f"Converted FSDP checkpoint from {fsdp_checkpoint_dir} to HF format in {hf_checkpoint_dir}")


if __name__ == "__main__":
    main()
