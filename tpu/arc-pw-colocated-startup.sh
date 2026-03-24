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

if [ "$ENABLE_JAX_DEV" == "true" ]; then
    echo "Enabling prerelease Jax and specifying extra idnex"
    export UV_PRELEASE=allow
    export UV_INDEX=https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/
fi

uv pip install .[core,tpu]

# Run any post-setup command if defined and not set to INSERT_POST_SETUP_CMD
if [ "$POST_SETUP_CMD" != "INSERT_POST_SETUP_CMD" ]; then
    eval "$POST_SETUP_CMD"
fi

# Run the colocated python benchmark script to calculate Data loading time.
python3 axlearn/cloud/gcp/examples/colocated_python_benchmark.py \
    --ckpt_path gs://axlearn-arc-testing/testing/runs/main/22747426067/checkpoints/step_00000100 --method colocated
