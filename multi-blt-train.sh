#!/bin/bash

RANK=$1
MASTER_NODE=$2
MASTER_PORT=$3
MASTER_PORT2=$4
nnode=${5:-1}

# Activate the target virtual environment
source /hpctmp/e0968891/virtualenvs/cuda/bin/activate

# Clear CUDA cache
python -c "import gc, torch; torch.cuda.empty_cache(); gc.collect()"

echo "Starting training at $(date)"
echo "Running on host: $(hostname)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits || echo "No GPU detected"
echo "Node Rank: [${RANK}] Master: ${MASTER_NODE}:${MASTER_PORT}, ${MASTER_NODE}:${MASTER_PORT2}"

# Navigate to the BLT code directory
cd /scratch/Projects/CFP-01/CFP01-CF-060/kieron/blt

# Launch Torchrun
# Since we are on 2 nodes, --nnodes=2 and --node-rank uses the OMPI rank (0 or 1)
torchrun \
    --nnodes ${nnode} \
    --nproc_per_node=8 \
    --master-addr ${MASTER_NODE} \
    --master-port ${MASTER_PORT} \
    --node-rank ${RANK} \
    -m bytelatent.train \
    config=/scratch/Projects/CFP-01/CFP01-CF-060/kieron/blt/bytelatent/configs/blt_1b_olmo_stage1.yaml \
    2>&1 | tee -a /scratch/Projects/CFP-01/CFP01-CF-060/kieron/logs/blt_train_rank${RANK}.log

sleep 15

torchrun \
    --nnodes ${nnode} \
    --nproc_per_node=8 \
    --master-addr ${MASTER_NODE} \
    --master-port ${MASTER_PORT2} \
    --node-rank ${RANK} \
    -m bytelatent.train \
    config=/scratch/Projects/CFP-01/CFP01-CF-060/kieron/blt/bytelatent/configs/blt_1b_olmo_stage2.yaml \
    2>&1 | tee -a /scratch/Projects/CFP-01/CFP01-CF-060/kieron/logs/blt_train_rank${RANK}.log

echo "Torchrun execution complete on rank ${RANK}."