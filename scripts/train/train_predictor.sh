#!/usr/bin/env bash
# Stage 2: train the Predictor (needs the Stage 1 VQGAN in pretrained_models/vqgan.pth).
# Default config expects 4 GPUs; edit CUDA_VISIBLE_DEVICES / --nproc_per_node and num_gpu in the yml.
cd "$(dirname "$0")/../.." || exit 1
PYTHONPATH="./:${PYTHONPATH}" CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run \
    --nproc_per_node=4 --master_port=1234 \
    basicsr/train.py -opt options/02train_predictor.yml --launcher pytorch
