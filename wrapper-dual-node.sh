#!/bin/bash

MASTER_NODE=$1
MASTER_PORT=$2
MASTER_PORT2=$3
nnode=${4:-1}

module load singularity

# The required image for your BLT run
IMAGE="/app1/common/singularity-img/hopper/cuda/cuda_12.1.0-cudnn8-devel-u20.04.sif"

# Pass Weights & Biases configs into the Singularity container via APPTAINERENV
export APPTAINERENV_WANDB_API_KEY=$WANDB_API_KEY
export APPTAINERENV_WANDB_ENTITY="cikguseven-national-university-of-singapore"
export APPTAINERENV_WANDB_PROJECT="Training"
export APPTAINERENV_WANDB_DIR="/scratch/Projects/CFP-01/CFP01-CF-060/kieron/logs"
export APPTAINERENV_TORCHMETRICS_SKIP_IMAGE_DIST_TESTS=1

# Pass the new highly-optimized NCCL variables into the container
export APPTAINERENV_NCCL_IB_DISABLE=$NCCL_IB_DISABLE
export APPTAINERENV_NCCL_IB_HCA=$NCCL_IB_HCA
export APPTAINERENV_NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME
export APPTAINERENV_NCCL_P2P_DISABLE=$NCCL_P2P_DISABLE
export APPTAINERENV_NCCL_DEBUG=$NCCL_DEBUG
export APPTAINERENV_NCCL_NET_GDR_LEVEL=$NCCL_NET_GDR_LEVEL
export APPTAINERENV_CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES
export APPTAINERENV_NCCL_NVLS_ENABLE=$NCCL_NVLS_ENABLE

# OMPI_COMM_WORLD_RANK is automatically injected by mpirun (0 for master, 1 for worker)
singularity exec \
    --bind $HOME:$HOME \
    --bind /scratch:/scratch \
    --bind /hpctmp:/hpctmp \
    -e $IMAGE \
    bash /scratch/Projects/CFP-01/CFP01-CF-060/kieron/multi-blt-train.sh ${OMPI_COMM_WORLD_RANK} ${MASTER_NODE} ${MASTER_PORT} ${MASTER_PORT2} ${nnode}