#!/bin/bash

# Grab the latest AXLearn from upstream
git init /root && cd /root && git remote add origin https://github.com/apple/axlearn
git -c protocol.version=2 fetch --no-tags --prune --no-recurse-submodules --depth=1 origin && git checkout main
uv pip install .[core,gpu]

# Start the training loop
python3 -m axlearn.common.launch_trainer_main --module=text.gpt.c4_trainer --config=fuji-70B-v2-flash \
    --trainer_dir=${GCS_PREFIX} --data_dir=gs://axlearn-public/tensorflow_datasets \
    --jax_backend=gpu --mesh_selector=gpu-a4-highgpu-8g-256 --trace_at_steps=5