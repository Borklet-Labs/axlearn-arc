#!/bin/bash

cd /root

# Execute the training test job and timeout after 5 minutes
timeout 300s python3 -m axlearn.common.launch_trainer_main \
    --module=text.gpt.c4_trainer --config=fuji-7B-v2-flash-single-host \
    --trainer_dir=${GCS_PREFIX} \
    --data_dir=gs://axlearn-public/tensorflow_datasets \
    --jax_backend=gpu \
    --mesh_selector=gpu-a4-highgpu-8g-256 \
    --trace_at_steps=5

# Check to see that the training timed out, versus erroring out
test "$?" -eq 124