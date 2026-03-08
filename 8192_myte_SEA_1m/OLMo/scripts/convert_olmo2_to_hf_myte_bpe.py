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
"""
Sample usage:

    python convert_olmo2_to_hf_myte_bpe.py \
        --input_dir /path/to/olmo2/weights \
        --output_dir /output/path \
        --tokenizer_type myte

Afterwards, models can be loaded via:

    from transformers import Olmo2ForCausalLM, AutoTokenizer
    model = Olmo2ForCausalLM.from_pretrained("/output/path")
    tokenizer = AutoTokenizer.from_pretrained("/output/path")

Note: the whole model must fit in RAM to run this script.
"""

import argparse
import gc
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import yaml
from transformers import (
    AutoTokenizer,
    Olmo2Config,
    Olmo2ForCausalLM,
    PreTrainedTokenizerFast,
)


# ---------------------------------------------------------------------------
# Per-tokenizer EOS and vocab-size defaults.
# These are applied when the checkpoint config reports eos_token_id == 0
# (a known OLMo bug) and when the config vocab_size disagrees with the
# tokenizer's actual vocabulary.
# ---------------------------------------------------------------------------


TOKENIZER_DEFAULTS: Dict[str, Dict[str, Optional[int]]] = {
    # Morphology-Driven Byte Encoding (MYTE) — 256-byte alphabet re-encoded
    # via Morfessor morpheme mapping; vocab is fixed at 384.
    "myte": {"eos_token_id": 1, "vocab_size": 384},
    # BPE
    "bpe": {"eos_token_id": 90369, "vocab_size": 90372},
}

SUPPORTED_TOKENIZER_TYPES = list(TOKENIZER_DEFAULTS)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def compute_intermediate_size(n, ffn_dim_multiplier=1, multiple_of=256):
    return multiple_of * ((int(ffn_dim_multiplier * int(8 * n / 3)) + multiple_of - 1) // multiple_of)

def read_json(path):
    with open(path, "r") as f:
        return json.load(f)

def write_json(text, path):
    with open(path, "w") as f:
        json.dump(text, f)


# ---------------------------------------------------------------------------
# Tokenizer-type detection
# ---------------------------------------------------------------------------

def detect_tokenizer_type(tokenizer_config: dict) -> str:
    """
    Infer the tokenizer type from the OLMo tokenizer config block.

    Returns one of: ``'myte'``, ``'bpe'``.

    Detection order:
    1. Fast path: name/path heuristics on the identifier string.
    2. Read ``tokenizer_config.json`` for an explicit ``tokenizer_class`` or
       ``auto_map`` entry.

    Raises ``ValueError`` if the type cannot be determined.
    """
    identifier = tokenizer_config.get("identifier", "")

    tok_cfg_path = Path(identifier) / "tokenizer_config.json"
    if not tok_cfg_path.is_file():
        raise ValueError(
            f"Cannot detect tokenizer type: '{tok_cfg_path}' not found. "
            "Pass --tokenizer_type explicitly."
        )

    with open(tok_cfg_path) as f:
        tok_cfg = json.load(f)

    tok_class = tok_cfg.get("tokenizer_class", "")
    auto_map  = tok_cfg.get("auto_map", {})

    if "MyT5Tokenizer" in tok_class or "MyT5Tokenizer" in str(auto_map):
        return "myte"
    if "PreTrainedTokenizerFast" in tok_class or "PreTrainedTokenizerFast" in str(auto_map):
        return "bpe"

    raise ValueError(
        f"Could not determine tokenizer type from '{tok_cfg_path}'. "
        "Pass --tokenizer_type explicitly (choices: myte, bpe)."
    )


# ---------------------------------------------------------------------------
# Main conversion entry point
# ---------------------------------------------------------------------------


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
    full_yaml    = yaml.safe_load(config_path.read_text())
    olmo2_config = full_yaml["model"]

    if not olmo2_config.get("attention_layer_norm", False):
        raise RuntimeError("OLMo2 checkpoints must have attention layer norm")
    if not olmo2_config.get("norm_after", False):
        raise RuntimeError("OLMo2 checkpoints must set norm_after to True")

    # ------------------------------------------------------------------
    # Resolve tokenizer type EARLY so EOS / vocab defaults are available
    # before building Olmo2Config.
    # ------------------------------------------------------------------
    if tokenizer_type is not None:
        resolved_type = tokenizer_type
        print(f"Using explicitly provided tokenizer type: '{resolved_type}'")
    elif include_tokenizer:
        # Raises ValueError if type cannot be determined — no silent fallback.
        resolved_type = detect_tokenizer_type(full_yaml.get("tokenizer", {}))
        print(f"Auto-detected tokenizer type: '{resolved_type}'")
    else:
        raise ValueError(
            "Cannot determine tokenizer type when --no_tokenizer is not set. "
            "Pass --tokenizer_type explicitly (choices: myte, bpe)."
        )

    if resolved_type not in TOKENIZER_DEFAULTS:
        raise ValueError(
            f"Unsupported tokenizer type '{resolved_type}'. "
            f"Choose from: {', '.join(SUPPORTED_TOKENIZER_TYPES)}."
        )

    tok_defaults = TOKENIZER_DEFAULTS[resolved_type]

    # ------------------------------------------------------------------
    # Architecture parameters
    # ------------------------------------------------------------------
    n_layers                = olmo2_config["n_layers"]
    n_heads                 = olmo2_config["n_heads"]
    dim                     = olmo2_config["d_model"]
    dims_per_head           = dim // n_heads
    base                    = olmo2_config["rope_theta"]
    inv_freq                = 1.0 / (base ** (torch.arange(0, dims_per_head, 2).float() / dims_per_head))
    max_position_embeddings = olmo2_config["max_sequence_length"]

    vocab_size = olmo2_config.get("embedding_size", olmo2_config["vocab_size"])

    # Override vocab_size with the tokenizer-specific default when available.
    default_vocab = tok_defaults.get("vocab_size")
    if default_vocab is not None and vocab_size != default_vocab:
        print(
            f"Warning: config vocab_size={vocab_size} differs from '{resolved_type}' "
            f"default ({default_vocab}). Using tokenizer default to avoid embedding mismatch."
        )
        vocab_size = default_vocab

    if olmo2_config.get("n_kv_heads") is not None:
        num_key_value_heads = olmo2_config["n_kv_heads"]
    elif olmo2_config.get("multi_query_attention"):
        num_key_value_heads = 1
    else:
        num_key_value_heads = n_heads

    # ------------------------------------------------------------------
    # Load weights
    # ------------------------------------------------------------------
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
            f"model.layers.{layer_i}.self_attn.q_proj.weight":          q_proj_weight,
            f"model.layers.{layer_i}.self_attn.k_proj.weight":          k_proj_weight,
            f"model.layers.{layer_i}.self_attn.v_proj.weight":          v_proj_weight,
            f"model.layers.{layer_i}.self_attn.o_proj.weight":          loaded[f"transformer.blocks.{layer_i}.attn_out.weight"],
            f"model.layers.{layer_i}.self_attn.q_norm.weight":          loaded[f"transformer.blocks.{layer_i}.q_norm.weight"],
            f"model.layers.{layer_i}.self_attn.k_norm.weight":          loaded[f"transformer.blocks.{layer_i}.k_norm.weight"],
            f"model.layers.{layer_i}.mlp.gate_proj.weight":             gate_proj_weight,
            f"model.layers.{layer_i}.mlp.down_proj.weight":             loaded[f"transformer.blocks.{layer_i}.ff_out.weight"],
            f"model.layers.{layer_i}.mlp.up_proj.weight":               up_proj_weight,
            f"model.layers.{layer_i}.post_attention_layernorm.weight":   loaded[f"transformer.blocks.{layer_i}.attn_norm.weight"],
            f"model.layers.{layer_i}.post_feedforward_layernorm.weight": loaded[f"transformer.blocks.{layer_i}.ff_norm.weight"],
            f"model.layers.{layer_i}.self_attn.rotary_emb.inv_freq":    inv_freq,
        }
        for k, v in state_dict.items():
            index_dict["weight_map"][k] = filename
            param_count += v.numel()
        torch.save(state_dict, os.path.join(tmp_model_path, filename))

    filename = f"pytorch_model-{n_layers + 1}-of-{n_layers + 1}.bin"
    state_dict = {
        "model.embed_tokens.weight": loaded["transformer.wte.weight"],
        "model.norm.weight":         loaded["transformer.ln_f.weight"],
        "lm_head.weight":            loaded["transformer.ff_out.weight"]
                                     if "transformer.ff_out.weight" in loaded
                                     else loaded["transformer.wte.weight"],
    }
    for k, v in state_dict.items():
        index_dict["weight_map"][k] = filename
        param_count += v.numel()
    torch.save(state_dict, os.path.join(tmp_model_path, filename))

    index_dict["metadata"] = {"total_size": param_count * 2}
    write_json(index_dict, os.path.join(tmp_model_path, "pytorch_model.bin.index.json"))

    # ------------------------------------------------------------------
    # Build HF config
    # ------------------------------------------------------------------
    if olmo2_config.get("mlp_hidden_size") is not None:
        intermediate_size = olmo2_config["mlp_hidden_size"] // 2
    else:
        intermediate_size = (dim * olmo2_config["mlp_ratio"]) // 2

    # Fix the OLMo bug where eos_token_id was recorded as 0.
    eos_token_id = olmo2_config["eos_token_id"]
    if fix_eos_token_id and eos_token_id == 0:
        correct_eos = tok_defaults["eos_token_id"]
        print(f"Fixing eos_token_id: 0 -> {correct_eos} ('{resolved_type}' default).")
        eos_token_id = correct_eos

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
        eos_token_id=eos_token_id,
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
            output_path=model_path,
            config=config,
            checkpoint_dir=input_base_path,
            tokenizer_path=tokenizer_path,
            tokenizer_type=resolved_type,
        )

    print("Loading the checkpoint in an OLMo2 model.")
    model = Olmo2ForCausalLM.from_pretrained(tmp_model_path, torch_dtype=torch.float32, low_cpu_mem_usage=True)
    del model.config._name_or_path
    print("Saving in the Transformers format.")
    model.save_pretrained(model_path, safe_serialization=safe_serialization)
    if tmp_cleanup:
        shutil.rmtree(tmp_model_path)


# ---------------------------------------------------------------------------
# Tokenizer writer
# ---------------------------------------------------------------------------


def _write_tokenizer(
    output_path: Path,
    config: Olmo2Config,
    checkpoint_dir: str,
    tokenizer_path: Optional[Path] = None,
    tokenizer_type: str = "myte",
) -> None:
    """
    Write a custom tokenizer (myte or bpe) that has already been
    converted to AutoTokenizer / HF format via convert_myte_tokenizer.py or equivalent.
    """
    if tokenizer_type not in TOKENIZER_DEFAULTS:
        raise ValueError(
            f"Unknown tokenizer_type '{tokenizer_type}'. "
            f"Choose from: {', '.join(SUPPORTED_TOKENIZER_TYPES)}."
        )

    config_path = Path(checkpoint_dir) / "config.yaml"
    olmo_tokenizer_cfg = yaml.safe_load(config_path.read_text()).get("tokenizer", {})
    identifier = olmo_tokenizer_cfg.get("identifier", "")

    if tokenizer_path is not None:
        identifier = str(tokenizer_path)

    if not identifier:
        raise ValueError(
            f"Cannot locate tokenizer for type '{tokenizer_type}'. "
            "Pass --tokenizer_json_path pointing to the pre-converted HF tokenizer directory."
        )

    print(f"Loading '{tokenizer_type}' tokenizer from {identifier}.")
    identifier_path = Path(identifier).resolve()
    is_local = identifier_path.exists()

    if is_local:
        # Load tokenizer files directly to avoid huggingface_hub repo-id validation
        # on absolute paths (broken in newer huggingface_hub versions).
        tokenizer_config_path = identifier_path / "tokenizer_config.json"
        with open(tokenizer_config_path) as f:
            tok_cfg = json.load(f)

        auto_map = tok_cfg.get("auto_map", {})

        if auto_map.get("AutoTokenizer"):
            # Custom tokenizer class — add the directory to sys.path so the
            # module referenced in auto_map can be imported.
            import sys
            if str(identifier_path) not in sys.path:
                sys.path.insert(0, str(identifier_path))
            tokenizer = AutoTokenizer.from_pretrained(
                identifier_path,          # Pass Path object, not str
                trust_remote_code=True,
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                identifier_path,
                trust_remote_code=True,
            )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            identifier,
            trust_remote_code=True,
        )

    def _id_to_str(tok_id: int) -> Optional[str]:
        s = tokenizer.convert_ids_to_tokens(tok_id)
        if s is None:
            s = tokenizer.decode([tok_id], skip_special_tokens=False) or None
        return s

    eos_token_id = config.eos_token_id
    if eos_token_id is not None and tokenizer.eos_token_id != eos_token_id:
        eos_str = _id_to_str(eos_token_id)
        if eos_str:
            print(f"  Overriding eos_token: '{tokenizer.eos_token}' (id={tokenizer.eos_token_id}) -> '{eos_str}' (id={eos_token_id})")
            tokenizer.eos_token = eos_str
        else:
            print(f"  Warning: eos_token_id {eos_token_id} not found in vocab; skipping EOS override.")

    pad_token_id = config.pad_token_id
    if pad_token_id is not None and tokenizer.pad_token_id != pad_token_id:
        pad_str = _id_to_str(pad_token_id)
        if pad_str:
            print(f"  Overriding pad_token: '{tokenizer.pad_token}' (id={tokenizer.pad_token_id}) -> '{pad_str}' (id={pad_token_id})")
            tokenizer.pad_token = pad_str
        else:
            print(f"  Warning: pad_token_id {pad_token_id} not found in vocab; skipping PAD override.")

    tokenizer.save_pretrained(output_path)

    if tokenizer_type == "myte":
        for asset_name in ("decompose.json", "morf_map_mc4_8192.json"):
            src_asset = Path(identifier) / asset_name
            dst_asset = Path(output_path) / asset_name
            if src_asset.is_file() and not dst_asset.is_file():
                print(f"  Copying tokenizer asset: {src_asset} -> {dst_asset}")
                shutil.copy(src_asset, dst_asset)

    # Copy any local Python file referenced via auto_map so the saved directory
    # is fully self-contained (important for myte's custom tokenizer class).
    src_cfg_path = Path(identifier) / "tokenizer_config.json"
    if src_cfg_path.is_file():
        with open(src_cfg_path) as f:
            src_cfg = json.load(f)
        for _key, class_ref in src_cfg.get("auto_map", {}).items():
            if isinstance(class_ref, list):
                class_ref = class_ref[0]
            if not class_ref:
                continue
            module_name = class_ref.split(".")[0]
            src_py = Path(identifier) / f"{module_name}.py"
            dst_py = Path(output_path) / f"{module_name}.py"
            if src_py.is_file() and not dst_py.is_file():
                print(f"  Copying custom tokenizer file: {src_py} -> {dst_py}")
                shutil.copy(src_py, dst_py)

    print(f"'{tokenizer_type}' tokenizer saved to {output_path}.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


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
        help="If set, do not convert the OLMo tokenizer to HF format.",
    )
    parser.add_argument(
        "--tokenizer_json_path",
        type=Path,
        default=None,
        help="Path to the pre-converted HF tokenizer directory (myte or bpe).",
    )
    parser.add_argument(
        "--tokenizer_type",
        type=str,
        default=None,
        choices=SUPPORTED_TOKENIZER_TYPES,
        help=(
            "Tokenizer type. If omitted, auto-detected from the checkpoint config. "
            "'myte' uses eos_token_id=1, vocab_size=384. "
            "'bpe' uses eos_token_id=90369, vocab_size=90372."
        ),
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Location to write HF model and tokenizer.",
    )
    parser.add_argument(
        "--no_fix_eos_token_id",
        action="store_false",
        dest="fix_eos_token_id",
        help=(
            "If set, do not fix eos_token_id when it is 0. "
            "The fix replaces 0 with the per-tokenizer default "
            "(1 for myte, 90369 for bpe)."
        ),
    )
    parser.add_argument(
        "--no_tmp_cleanup",
        action="store_false",
        dest="tmp_cleanup",
        help="If passed, don't remove the temp dir at end of conversion.",
    )
    parser.add_argument(
        "--no_safe_serialization",
        action="store_false",
        dest="safe_serialization",
        help="Save using .bin instead of safetensors.",
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