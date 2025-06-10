#!/bin/bash

# Fetch the latest version of AXLearn directly from upstream
cd /root
git clone --depth 1 https://github.com/apple/axlearn.git
cd /root/axlearn

# Install the requirements for AXLearn
uv pip install .[core,gpu,gcp] 

# Patch Fuji to limit the max_steps = 10
sed -i 's/max_step=max_step/max_step=10/g' axlearn/experiments/text/gpt/fuji.py

# Execute the training test job
python3 -m axlearn.common.launch_trainer_main \
    --module=text.gpt.c4_trainer --config=fuji-7B-v2-flash-single-host \
    --trainer_dir=gs://axlearn-arc-testing/a4-sa-test \
    --data_dir=gs://axlearn-public/tensorflow_datasets \
    --jax_backend=gpu \
    --mesh_selector=gpu-a4-highgpu-8g-256 \
    --trace_at_steps=5

# Hang to do some further analysis
while true; do sleep 1; done