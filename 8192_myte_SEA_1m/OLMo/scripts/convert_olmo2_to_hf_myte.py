# Copyright 2024 EleutherAI and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import gc
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml
from tokenizers import Tokenizer
from transformers import (
    AutoTokenizer,
    Olmo2Config,
    Olmo2ForCausalLM,
    PreTrainedTokenizerFast,
)
from transformers.models.gpt2.tokenization_gpt2_fast import GPT2TokenizerFast

"""
Sample usage:

```
python src/transformers/models/olmo2/convert_olmo2_to_hf.py \
    --input_dir /path/to/downloaded/olmo2/weights --output_dir /output/path
```

Thereafter, models can be loaded via:

```py
from transformers import Olmo2ForCausalLM, AutoTokenizer

model = Olmo2ForCausalLM.from_pretrained("/output/path")
tokenizer = AutoTokenizer.from_pretrained("/output/path")
```

Important note: you need to be able to host the whole model in RAM to execute this script (even if the biggest versions
come in several checkpoints they each contain a part of each weight of the model, so we need to load them all in RAM).
"""


def compute_intermediate_size(n, ffn_dim_multiplier=1, multiple_of=256):
    return multiple_of * ((int(ffn_dim_multiplier * int(8 * n / 3)) + multiple_of - 1) // multiple_of)


def read_json(path):
    with open(path, "r") as f:
        return json.load(f)


def write_json(text, path):
    with open(path, "w") as f:
        json.dump(text, f)


def detect_tokenizer_type(tokenizer_config: dict) -> str:
    """
    Detect the tokenizer type from the config.
    Returns one of: 'myte', 'parity_bpe', 'byte_bpe', 'gpt2'
    """
    identifier = tokenizer_config.get("identifier", "")

    # Check if it's a local path with a tokenizer_config.json
    config_path = Path(identifier) / "tokenizer_config.json"
    if config_path.is_file():
        with open(config_path) as f:
            tok_cfg = json.load(f)
        tokenizer_class = tok_cfg.get("tokenizer_class", "")
        auto_map = tok_cfg.get("auto_map", {})

        if "MyT5Tokenizer" in tokenizer_class or "MyT5Tokenizer" in str(auto_map):
            return "myte"
        if "ParityBPE" in tokenizer_class or "parity" in tokenizer_class.lower():
            return "parity_bpe"
        if "ByteLevel" in tokenizer_class or tok_cfg.get("model", {}).get("type") == "BPE":
            model_path = Path(identifier) / "tokenizer.json"
            if model_path.is_file():
                with open(model_path) as f:
                    tok_model = json.load(f)
                if tok_model.get("model", {}).get("type") == "BPE" and \
                   any("ByteLevel" in str(p) for p in tok_model.get("pre_tokenizer", {}).values()):
                    return "byte_bpe"
    return "gpt2"


def write_model(
    model_path,
    input_base_path,
    include_tokenizer=True,
    tokenizer_path=None,
    tokenizer_type=None,
    safe_serialization=True,
    fix_eos_token_id=True,
    tmp_cleanup=True,
):
    os.makedirs(model_path, exist_ok=True)
    tmp_model_path = os.path.join(model_path, "tmp")
    os.makedirs(tmp_model_path, exist_ok=True)

    config_path = Path(input_base_path) / "config.yaml"
    olmo2_config = yaml.safe_load(config_path.read_text())["model"]

    if not olmo2_config.get("attention_layer_norm", False):
        raise RuntimeError("OLMo2 checkpoints must have attention layer norm")
    if not olmo2_config.get("norm_after", False):
        raise RuntimeError("OLMo2 checkpoints must set norm_after to True")

    n_layers = olmo2_config["n_layers"]
    n_heads = olmo2_config["n_heads"]
    dim = olmo2_config["d_model"]
    dims_per_head = dim // n_heads
    base = olmo2_config["rope_theta"]
    inv_freq = 1.0 / (base ** (torch.arange(0, dims_per_head, 2).float() / dims_per_head))
    max_position_embeddings = olmo2_config["max_sequence_length"]

    vocab_size = olmo2_config.get("embedding_size", olmo2_config["vocab_size"])

    if olmo2_config.get("n_kv_heads", None) is not None:
        num_key_value_heads = olmo2_config["n_kv_heads"]
    elif olmo2_config["multi_query_attention"]:
        num_key_value_heads = 1
    else:
        num_key_value_heads = n_heads

    print(f"Fetching all parameters from the checkpoint at {input_base_path}.")
    loaded = torch.load(os.path.join(input_base_path, "model.pt"), map_location="cpu")

    param_count = 0
    index_dict: Dict[str, Any] = {"weight_map": {}}
    for layer_i in range(n_layers):
        filename = f"pytorch_model-{layer_i + 1}-of-{n_layers + 1}.bin"
        fused_dims = [dim, dims_per_head * num_key_value_heads, dims_per_head * num_key_value_heads]
        q_proj_weight, k_proj_weight, v_proj_weight = torch.split(
            loaded[f"transformer.blocks.{layer_i}.att_proj.weight"], fused_dims, dim=0
        )
        up_proj_weight, gate_proj_weight = torch.chunk(
            loaded[f"transformer.blocks.{layer_i}.ff_proj.weight"], 2, dim=0
        )
        state_dict = {
            f"model.layers.{layer_i}.self_attn.q_proj.weight": q_proj_weight,
            f"model.layers.{layer_i}.self_attn.k_proj.weight": k_proj_weight,
            f"model.layers.{layer_i}.self_attn.v_proj.weight": v_proj_weight,
            f"model.layers.{layer_i}.self_attn.o_proj.weight": loaded[
                f"transformer.blocks.{layer_i}.attn_out.weight"
            ],
            f"model.layers.{layer_i}.self_attn.q_norm.weight": loaded[
                f"transformer.blocks.{layer_i}.q_norm.weight"
            ],
            f"model.layers.{layer_i}.self_attn.k_norm.weight": loaded[
                f"transformer.blocks.{layer_i}.k_norm.weight"
            ],
            f"model.layers.{layer_i}.mlp.gate_proj.weight": gate_proj_weight,
            f"model.layers.{layer_i}.mlp.down_proj.weight": loaded[f"transformer.blocks.{layer_i}.ff_out.weight"],
            f"model.layers.{layer_i}.mlp.up_proj.weight": up_proj_weight,
            f"model.layers.{layer_i}.post_attention_layernorm.weight": loaded[
                f"transformer.blocks.{layer_i}.attn_norm.weight"
            ],
            f"model.layers.{layer_i}.post_feedforward_layernorm.weight": loaded[
                f"transformer.blocks.{layer_i}.ff_norm.weight"
            ],
        }
        state_dict[f"model.layers.{layer_i}.self_attn.rotary_emb.inv_freq"] = inv_freq

        for k, v in state_dict.items():
            index_dict["weight_map"][k] = filename
            param_count += v.numel()
        torch.save(state_dict, os.path.join(tmp_model_path, filename))

    filename = f"pytorch_model-{n_layers + 1}-of-{n_layers + 1}.bin"
    state_dict = {
        "model.embed_tokens.weight": loaded["transformer.wte.weight"],
        "model.norm.weight": loaded["transformer.ln_f.weight"],
        "lm_head.weight": loaded["transformer.ff_out.weight"]
        if "transformer.ff_out.weight" in loaded
        else loaded["transformer.wte.weight"],
    }

    for k, v in state_dict.items():
        index_dict["weight_map"][k] = filename
        param_count += v.numel()
    torch.save(state_dict, os.path.join(tmp_model_path, filename))

    index_dict["metadata"] = {"total_size": param_count * 2}
    write_json(index_dict, os.path.join(tmp_model_path, "pytorch_model.bin.index.json"))

    if olmo2_config.get("mlp_hidden_size", None) is not None:
        intermediate_size = olmo2_config["mlp_hidden_size"] // 2
    else:
        intermediate_size = (dim * olmo2_config["mlp_ratio"]) // 2

    if fix_eos_token_id and olmo2_config["eos_token_id"] == 0:
        print("Changing eos_token_id from 0 to 50279.")
        olmo2_config["eos_token_id"] = 50279

    config = Olmo2Config(
        vocab_size=vocab_size,
        hidden_size=dim,
        intermediate_size=intermediate_size,
        num_hidden_layers=n_layers,
        num_attention_heads=n_heads,
        num_key_value_heads=num_key_value_heads,
        max_position_embeddings=max_position_embeddings,
        pad_token_id=olmo2_config["pad_token_id"],
        bos_token_id=None,
        eos_token_id=olmo2_config["eos_token_id"],
        tie_word_embeddings=olmo2_config["weight_tying"],
        rms_norm_eps=olmo2_config["layer_norm_eps"],
        rope_theta=base,
    )
    config.save_pretrained(tmp_model_path)

    del state_dict
    del loaded
    gc.collect()

    if include_tokenizer:
        _write_tokenizer(
            model_path,
            config,
            input_base_path,
            tokenizer_path=tokenizer_path,
            tokenizer_type=tokenizer_type,
        )

    print("Loading the checkpoint in a OLMo2 model.")
    model = Olmo2ForCausalLM.from_pretrained(tmp_model_path, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    del model.config._name_or_path
    print("Saving in the Transformers format.")
    model.save_pretrained(model_path, safe_serialization=safe_serialization)
    if tmp_cleanup:
        shutil.rmtree(tmp_model_path)


def _write_tokenizer_gpt2(
    output_path: Path,
    config: Olmo2Config,
    checkpoint_dir: str,
    input_tokenizer_path: Optional[Path],
) -> None:
    """Write a GPT2-style byte-level BPE tokenizer (original OLMo2 behaviour)."""
    print(f"Saving a {GPT2TokenizerFast.__name__} to {output_path}.")

    if input_tokenizer_path is not None:
        base_tokenizer = Tokenizer.from_file(str(input_tokenizer_path))
    else:
        config_path = Path(checkpoint_dir) / "config.yaml"
        tokenizer_config = yaml.safe_load(config_path.read_text())["tokenizer"]
        if Path(tokenizer_config["identifier"]).is_file():
            base_tokenizer = Tokenizer.from_file(tokenizer_config["identifier"])
        else:
            base_tokenizer = Tokenizer.from_pretrained(tokenizer_config["identifier"])

    eos_token_id = config.eos_token_id if config.eos_token_id is not None else base_tokenizer.get_vocab_size() - 1
    pad_token_id = config.pad_token_id if config.pad_token_id is not None else eos_token_id

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=base_tokenizer,
        eos_token=base_tokenizer.decode([eos_token_id], skip_special_tokens=False),
        pad_token=base_tokenizer.decode([pad_token_id], skip_special_tokens=False),
    )
    tokenizer.save_pretrained(output_path)


def _write_tokenizer_auto(
    output_path: Path,
    config: Olmo2Config,
    checkpoint_dir: str,
    input_tokenizer_path: Optional[Path],
    tokenizer_type: str,
) -> None:
    """
    Write a custom tokenizer (myte, parity_bpe, byte_bpe) that has already been
    converted to AutoTokenizer / HF format via convert_myte_tokenizer.py or equivalent.
    The tokenizer directory must contain tokenizer_config.json (and optionally an
    auto_map entry pointing to a local Python file).
    """
    config_path = Path(checkpoint_dir) / "config.yaml"
    olmo_tokenizer_cfg = yaml.safe_load(config_path.read_text()).get("tokenizer", {})
    identifier = olmo_tokenizer_cfg.get("identifier", "")

    # Allow an explicit override path
    if input_tokenizer_path is not None:
        identifier = str(input_tokenizer_path)

    if not identifier:
        raise ValueError(
            f"Cannot locate tokenizer for type '{tokenizer_type}'. "
            "Pass --tokenizer_json_path pointing to the tokenizer directory."
        )

    print(f"Loading '{tokenizer_type}' tokenizer from {identifier}.")
    tokenizer = AutoTokenizer.from_pretrained(
        identifier,
        trust_remote_code=True,
        local_files_only=Path(identifier).exists(),
    )

    # Patch special tokens from model config if needed
    eos_token_id = config.eos_token_id
    pad_token_id = config.pad_token_id

    if eos_token_id is not None and tokenizer.eos_token_id != eos_token_id:
        print(
            f"  Overriding tokenizer eos_token_id: {tokenizer.eos_token_id} -> {eos_token_id}"
        )
        tokenizer.eos_token = tokenizer.decode([eos_token_id])
        tokenizer.eos_token_id = eos_token_id

    if pad_token_id is not None and tokenizer.pad_token_id != pad_token_id:
        print(
            f"  Overriding tokenizer pad_token_id: {tokenizer.pad_token_id} -> {pad_token_id}"
        )
        tokenizer.pad_token = tokenizer.decode([pad_token_id])
        tokenizer.pad_token_id = pad_token_id

    tokenizer.save_pretrained(output_path)

    # If the tokenizer relies on a local Python file (e.g. myt5_tokenizer.py via auto_map),
    # copy that file into the output directory so the saved model is self-contained.
    src_tok_cfg_path = Path(identifier) / "tokenizer_config.json"
    if src_tok_cfg_path.is_file():
        with open(src_tok_cfg_path) as f:
            src_tok_cfg = json.load(f)
        auto_map = src_tok_cfg.get("auto_map", {})
        for _key, class_ref in auto_map.items():
            # class_ref may be a string like "myt5_tokenizer.MyT5Tokenizer" or a list
            if isinstance(class_ref, list):
                class_ref = class_ref[0]
            if class_ref is None:
                continue
            module_name = class_ref.split(".")[0]
            src_py = Path(identifier) / f"{module_name}.py"
            dst_py = Path(output_path) / f"{module_name}.py"
            if src_py.is_file() and not dst_py.is_file():
                print(f"  Copying {src_py} -> {dst_py}")
                shutil.copy(src_py, dst_py)

    print(f"'{tokenizer_type}' tokenizer saved to {output_path}.")


def _write_tokenizer(
    output_path: Path,
    config: Olmo2Config,
    checkpoint_dir: str,
    tokenizer_path: Optional[Path] = None,
    tokenizer_type: Optional[str] = None,
) -> None:
    """
    Dispatch to the appropriate tokenizer writer based on tokenizer_type.

    If tokenizer_type is None, attempt auto-detection from the checkpoint config.
    """
    if tokenizer_type is None:
        try:
            cfg_path = Path(checkpoint_dir) / "config.yaml"
            tokenizer_config = yaml.safe_load(cfg_path.read_text()).get("tokenizer", {})
            tokenizer_type = detect_tokenizer_type(tokenizer_config)
            print(f"Auto-detected tokenizer type: '{tokenizer_type}'")
        except Exception:
            tokenizer_type = "gpt2"
            print("Could not auto-detect tokenizer type, defaulting to 'gpt2'.")

    if tokenizer_type == "gpt2":
        _write_tokenizer_gpt2(output_path, config, checkpoint_dir, tokenizer_path)
    elif tokenizer_type in ("myte", "parity_bpe", "byte_bpe"):
        _write_tokenizer_auto(output_path, config, checkpoint_dir, tokenizer_path, tokenizer_type)
    else:
        raise ValueError(
            f"Unknown tokenizer_type '{tokenizer_type}'. "
            "Choose from: gpt2, myte, parity_bpe, byte_bpe."
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Location of OLMo2 weights, which contains config.yaml and model.pt.",
    )
    parser.add_argument(
        "--no_tokenizer",
        action="store_false",
        dest="include_tokenizer",
        help="If set, do not convert OLMo tokenizer to HF tokenizer.",
    )
    parser.add_argument(
        "--tokenizer_json_path",
        type=Path,
        default=None,
        help=(
            "For gpt2: path to tokenizer.json file. "
            "For myte/parity_bpe/byte_bpe: path to the pre-converted HF tokenizer directory."
        ),
    )
    parser.add_argument(
        "--tokenizer_type",
        type=str,
        default=None,
        choices=["gpt2", "myte", "parity_bpe", "byte_bpe"],
        help=(
            "Tokenizer type to use. If not set, auto-detection is attempted from the checkpoint config. "
            "myte/parity_bpe/byte_bpe expect the tokenizer to have already been converted to "
            "AutoTokenizer format (e.g. via convert_myte_tokenizer.py)."
        ),
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Location to write HF model and tokenizer",
    )
    parser.add_argument(
        "--no_fix_eos_token_id",
        action="store_false",
        dest="fix_eos_token_id",
        help="If set, does not change eos token id from 0 to 50279 if it is 0.",
    )
    parser.add_argument(
        "--no_tmp_cleanup",
        action="store_false",
        dest="tmp_cleanup",
        help="If passed, don't remove temp dir at end of HF conversion.",
    )
    parser.add_argument(
        "--no_safe_serialization",
        action="store_false",
        dest="safe_serialization",
        help="Whether or not to save using `safetensors`.",
    )
    args = parser.parse_args()
    write_model(
        model_path=args.output_dir,
        input_base_path=args.input_dir,
        safe_serialization=args.safe_serialization,
        include_tokenizer=args.include_tokenizer,
        tokenizer_path=args.tokenizer_json_path,
        tokenizer_type=args.tokenizer_type,
        fix_eos_token_id=args.fix_eos_token_id,
        tmp_cleanup=args.tmp_cleanup,
    )


if __name__ == "__main__":
    main()
