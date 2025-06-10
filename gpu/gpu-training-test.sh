#!/bin/bash

# Fetch the latest version of AXLearn directly from upstream
cd /root
git clone --depth 1 https://github.com/apple/axlearn.git
cd /root/axlearn

# Install the requirements for AXLearn
uv pip install .[core,gpu,gcp] 

# Ensure we have no checkpoints stored
gsutil -m rm -rf gs://axlearn-arc-testing/a4-sa-test/launch_trainer_flags gs://axlearn-arc-testing/a4-sa-test/model_analysis.txt gs://axlearn-arc-testing/a4-sa-test/trainer_config gs://axlearn-arc-testing/a4-sa-test/trainer_state_tree.txt gs://axlearn-arc-testing/a4-sa-test/checkpoints

# Execute the training test job and timeout after 5 minutes
timeout 300s python3 -m axlearn.common.launch_trainer_main \
    --module=text.gpt.c4_trainer --config=fuji-7B-v2-flash-single-host \
    --trainer_dir=gs://axlearn-arc-testing/a4-sa-test \
    --data_dir=gs://axlearn-public/tensorflow_datasets \
    --jax_backend=gpu \
    --mesh_selector=gpu-a4-highgpu-8g-256 \
    --trace_at_steps=5
