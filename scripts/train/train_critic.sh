#!/usr/bin/env bash
# Stage 3: train the Critic (needs the Stage 2 predictor checkpoint referenced in the yml).
# Default config expects 4 GPUs; edit CUDA_VISIBLE_DEVICES / --nproc_per_node and num_gpu in the yml.
cd "$(dirname "$0")/../.." || exit 1
PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run \
    --nproc_per_node=4 --master_port=1234 \
    basicsr/train.py -opt options/03train_critic.yml --launcher pytorch
