#!/bin/bash

mkdir -p /localhome/kieron/consolidated

export PYTHONPATH="/localhome/kieron/8192_myte_SEA_1m/OLMo:${PYTHONPATH:-}"

echo "Starting at $(date)"

python /localhome/kieron/8192_myte_SEA_1m/OLMo/scripts/convert_olmo2_to_hf_myte_bpe.py \
    --input_dir /localhome/kieron/unshard \
    --output_dir /localhome/kieron/consolidated \
    --tokenizer_type myte \
    --tokenizer_json_path /localhome/kieron/8192_myte_SEA_1m
