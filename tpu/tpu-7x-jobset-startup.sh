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
git init /root && cd /root 
git remote add origin $GIT_ORIGIN
git -c protocol.version=2 fetch --no-tags --prune --no-recurse-submodules --depth=1 origin
git checkout $GIT_BRANCH

# Show the commit information
git log -1 --stat --pretty=format:"%H" --no-patch

uv pip install --prerelease=allow .[core,tpu]

# Run any post-setup command if defined and not set to INSERT_POST_SETUP_CMD
if [ "$POST_SETUP_CMD" != "INSERT_POST_SETUP_CMD" ]; then
    eval "$POST_SETUP_CMD"
fi

# Patch fuji.py to add new mesh selectors
if [ "$FUJI_PATCH_FILE" != "INSERT_FUJI_PATCH_FILE" ]; then
    echo "Applying patch to axlearn/experiments/text/gpt/fuji.py"
    git apply $FUJI_PATCH_FILE || exit 1
else
    echo "Not applying any mesh selector patches to fuji.py"
fi

# Modify the batch size to account for TPU 7x 2x2x1
echo "Updating global batch size to 32"
sed -i 's/train_batch_size=train_batch_size/train_batch_size=32/g' /root/axlearn/experiments/text/gpt/fuji.py

# Start the training loop
python3 -m axlearn.common.launch_trainer_main \
    --module=text.gpt.c4_trainer --config=fuji-7B-v2-flash \
    --trainer_dir=${GCS_PREFIX} \
    --data_dir=gs://axlearn-public/tensorflow_datasets \
    --jax_backend=tpu \
    --mesh_selector=arc-tpu-7x-8 \
    --trace_at_steps=5 \
    --trainer_log_every_n_steps=1