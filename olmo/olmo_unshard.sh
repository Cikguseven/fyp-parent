#!/bin/bash

mkdir -p /localhome/kieron/unshard

export PYTHONPATH="/localhome/kieron/8192_myte_SEA_1m/OLMo:${PYTHONPATH:-}"

python /localhome/kieron/8192_myte_SEA_1m/OLMo/scripts/unshard.py \
    /localhome/kieron/step60000 \
    /localhome/kieron/unshard
