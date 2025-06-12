#!/bin/bash

cd /root/axlearn

# Execute the training test job and timeout after 5 minutes
timeout 300s python3 -m axlearn.common.launch_trainer_main \
    --module=text.gpt.c4_trainer --config=fuji-7B-v2-flash-single-host \
    --trainer_dir=${GCS_PREFIX} \
    --data_dir=gs://axlearn-public/tensorflow_datasets \
    --jax_backend=gpu \
    --mesh_selector=gpu-a4-highgpu-8g-256 \
    --trace_at_steps=5 2>&1 | tee -a /home/runner/_work/axlearn.log && \
    gsutil -m cp /home/runner/_work/axlearn.log ${GCS_PREFIX}/testing/gpu-fuji-7b-single-host-$(date +"%Y-%m-%d-%T")-${HOSTNAME}.log

# Check to see that the training timed out, versus erroring out
test "$?" -eq 124