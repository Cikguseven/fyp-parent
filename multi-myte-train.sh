#!/bin/bash

RANK=$1
MASTER_NODE=$2
MASTER_PORT=$3
MASTER_PORT2=$4
nnode=${5:-1}

# Activate the target virtual environment
source /hpctmp/e0968891/virtualenvs/fyp/bin/activate

# Clear CUDA cache
python -c "import gc, torch; torch.cuda.empty_cache(); gc.collect()"

echo "Starting training at $(date)"
echo "Running on host: $(hostname)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits || echo "No GPU detected"
echo "Node Rank: [${RANK}] Master: ${MASTER_NODE}:${MASTER_PORT}, ${MASTER_NODE}:${MASTER_PORT2}"

# Launch Torchrun
# Since we are on 2 nodes, --nnodes=2 and --node-rank uses the OMPI rank (0 or 1)
torchrun \
    --nnodes ${nnode} \
    --nproc_per_node=2 \
    --master-addr ${MASTER_NODE} \
    --master-port ${MASTER_PORT} \
    --node-rank ${RANK} \
    /scratch/Projects/CFP-01/CFP01-CF-060/kieron/8192_myte_SEA_1m/OLMo/scripts/train.py \
    /scratch/Projects/CFP-01/CFP01-CF-060/kieron/olmo/myte_stage1_dual.yaml \
    2>&1 | tee -a /scratch/Projects/CFP-01/CFP01-CF-060/kieron/olmo/myte_stage1_dual_rank${RANK}.log

sleep 15

torchrun \
    --nnodes ${nnode} \
    --nproc_per_node=2 \
    --master-addr ${MASTER_NODE} \
    --master-port ${MASTER_PORT2} \
    --node-rank ${RANK} \
    /scratch/Projects/CFP-01/CFP01-CF-060/kieron/8192_myte_SEA_1m/OLMo/scripts/train.py \
    /scratch/Projects/CFP-01/CFP01-CF-060/kieron/olmo/myte_stage2_dual.yaml \
    2>&1 | tee -a /scratch/Projects/CFP-01/CFP01-CF-060/kieron/olmo/myte_stage2_dual_rank${RANK}.log

echo "Torchrun execution complete on rank ${RANK}."