#!/bin/bash

source cuda-switch 12.8
source .venv/bin/activate

python examples/gradio_demo.py \
    --config configs/shape_denoiser_multimesh.yaml \
    --checkpoint weights/multi_part_dit.safetensors \
    --vae-checkpoint weights/vae.safetensors \
    --low-vram
