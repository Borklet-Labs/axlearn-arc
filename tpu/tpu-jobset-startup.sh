#!/bin/bash

# Check for custom origin
if [ -z "$CUSTOM_GIT_ORIGIN" ]; then
    GIT_ORIGIN="https://github.com/Borklet-Labs/axlearn"
else
    GIT_ORIGIN="$CUSTOM_GIT_ORIGIN"
fi

# Check for branch name
if [ -z "$CUSTOM_GIT_BRANCH" ]; then
    GIT_BRANCH="main"
else
    GIT_BRANCH="$CUSTOM_GIT_BRANCH"
fi

echo "About to pull branch $GIT_BRANCH from origin $GIT_ORIGIN"

# Grab the latest AXLearn from upstream
echo "1"
git init /root && cd /root
echo "2"
git remote add origin $GIT_ORIGIN
echo "3"
git -c protocol.version=2 fetch --no-tags --prune --no-recurse-submodules --depth=1 origin
echo "4"
git checkout $GIT_BRANCH
echo "5"
# Show the commit information
git log -1 --stat --pretty=format:"%H" --no-patch
echo "6"
export UV_FIND_LINKS="https://storage.googleapis.com/axlearn-wheels/wheels.html"
uv pip install -e -v .[core,tpu]
echo "Done installing AxLearn"

# Modify the batch size to account for TPU v6e 4x4
sed -i 's/train_batch_size=train_batch_size/train_batch_size=32/g' /root/axlearn/experiments/text/gpt/fuji.py

# Start the training loop
python3 -m axlearn.common.launch_trainer_main \
    --module=text.gpt.c4_trainer --config=fuji-7B-v2-flash \
    --trainer_dir=${GCS_PREFIX} \
    --data_dir=gs://axlearn-public/tensorflow_datasets \
    --jax_backend=tpu \
    --mesh_selector=tpu-v6e-16 \
    --trace_at_steps=5
