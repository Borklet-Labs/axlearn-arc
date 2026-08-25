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

# Check for custom commit or fallback to branch
if [ -n "$CUSTOM_GIT_COMMIT" ]; then
    TARGET_REF="$CUSTOM_GIT_COMMIT"
    echo "Using custom commit: $TARGET_REF"
else
    TARGET_REF="$GIT_BRANCH"
    echo "Using branch: $TARGET_REF"
fi

echo "About to pull ref $TARGET_REF from origin $GIT_ORIGIN"

# Check if Checkpoint Steps were passed. If not 100 default.
if [ -z "$STEPS_CHECKPOINT" ]; then
    STEPS_CHECKPOINT=30
else
    STEPS_CHECKPOINT="$STEPS_CHECKPOINT"
fi

# Check Max Steps. Default 500.
if [ -z "$MAX_STEPS" ]; then
    MAX_STEPS=100
fi

# Grab AXLearn from upstream at target ref
git init /root && cd /root
git remote add origin $GIT_ORIGIN
git -c protocol.version=2 fetch --no-tags --prune --no-recurse-submodules --depth=1 origin $TARGET_REF
git checkout $TARGET_REF

# Show the commit information
git log -1 --stat --pretty=format:"%H" --no-patch

if [ "$ENABLE_JAX_DEV" == "true" ]; then
    echo "Enabling prerelease Jax and specifying extra index"
    export UV_PRELEASE=allow
    export UV_INDEX=https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/
fi

uv pip install .[core,tpu]

# Run any post-setup command if defined and not set to INSERT_POST_SETUP_CMD
if [ "$POST_SETUP_CMD" != "INSERT_POST_SETUP_CMD" ] && [ -n "$POST_SETUP_CMD" ]; then
    eval "$POST_SETUP_CMD"
fi

# Patch tpu_splash_attention.py for JAX 0.9.2 get_kernel_name compatibility if needed
python3 -c "
import re
path = '/root/axlearn/common/flash_attention/tpu_splash_attention.py'
try:
    with open(path) as f:
        c = f.read()
    c_fixed = re.sub(r'kernel_name = get_kernel_name\(\s*(?:dataclasses\.asdict\(block_sizes\)|dict\([^)]+\)),\s*', 'kernel_name = get_kernel_name(\n        ', c)
    with open(path, 'w') as f:
        f.write(c_fixed)
except Exception:
    pass
"

# Patch fuji.py to add new mesh selectors if provided
if [ "$FUJI_PATCH_FILE" != "INSERT_FUJI_PATCH_FILE" ] && [ -n "$FUJI_PATCH_FILE" ] && [ -f "$FUJI_PATCH_FILE" ]; then
    echo "Applying patch to axlearn/experiments/text/gpt/fuji.py"
    git apply $FUJI_PATCH_FILE || exit 1
else
    echo "Not applying any mesh selector patches to fuji.py"
fi

# Modify the batch size to account for TPU v5p-8
sed -i 's/train_batch_size=train_batch_size/train_batch_size=64/g' /root/axlearn/experiments/text/gpt/fuji.py
sed -i 's/fsdp=8/fsdp=4/g' /root/axlearn/experiments/text/gpt/fuji.py

LOG_FILE="training_log_dump.log"

cleanup_logs() {
    if [ -f "$LOG_FILE" ]; then
        echo "Uploading captured logs to GCS before exiting..."
        gsutil cp "$LOG_FILE" ${GCS_PREFIX}/runs/${GIT_BRANCH}/${GH_RUN_ID}/training_log_dump.log
    fi
}
trap cleanup_logs EXIT


# Modify checkpointing steps
sed -i 's/lr_warmup_steps: int = 2000/lr_warmup_steps: int = 15/g' /root/axlearn/experiments/text/gpt/common.py
sed -i "/trn2_config = _generate_trn2_custom_configs/a \    max_step=$MAX_STEPS" /root/axlearn/experiments/text/gpt/fuji.py
sed -i "/max_step=max_step,/a \            save_every_n_steps=$STEPS_CHECKPOINT," /root/axlearn/experiments/text/gpt/fuji.py

# Start the training loop
python3 -m axlearn.common.launch_trainer_main \
    --module=text.gpt.c4_trainer --config=fuji-7B-v2-flash \
    --trainer_dir=${GCS_PREFIX}/runs/${GIT_BRANCH}/${GH_RUN_ID} \
    --data_dir=gs://axlearn-public/tensorflow_datasets \
    --jax_backend=proxy \
    --mesh_selector=arc-tpu-v5p-8 \
    --trainer_log_every_n_steps=1 2>&1 | tee ${LOG_FILE}
