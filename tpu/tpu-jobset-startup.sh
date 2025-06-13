#!/bin/bash

# Grab the latest AXLearn from upstream
git init /root && cd /root && git remote add origin https://github.com/apple/axlearn
git -c protocol.version=2 fetch --no-tags --prune --no-recurse-submodules --depth=1 origin && git checkout main
uv pip install .[core,tpu]

# Modify the batch size to account for TPU v6e 4x4
sed -i \"s/train_batch_size=train_batch_size/train_batch_size=32/g\" /root/axlearn/experiments/text/gpt/fuji.py

# Start the training loop
python3 -m axlearn.common.launch_trainer_main \
    --module=text.gpt.c4_trainer --config=fuji-7B-v2-flash \
    --trainer_dir=gs://axlearn-arc-testing/a4-sa-test \
    --data_dir=gs://axlearn-public/tensorflow_datasets \
    --jax_backend=tpu \
    --mesh_selector=tpu-v6e-16 \
    --trace_at_steps=5