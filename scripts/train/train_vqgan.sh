#!/usr/bin/env bash
# Stage 1: pretrain the VQGAN codebook on high-quality images (datasets/train/HQ_sub).
# Default config expects 4 GPUs; edit CUDA_VISIBLE_DEVICES / --nproc_per_node and num_gpu in the yml.
cd "$(dirname "$0")/../.." || exit 1
PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run \
    --nproc_per_node=4 --master_port=1234 \
    basicsr/train.py -opt options/01train_FeMaSR_HQ_pretrain_stage.yml --launcher pytorch
