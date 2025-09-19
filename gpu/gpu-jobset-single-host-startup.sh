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

export UV_FIND_LINKS="https://storage.googleapis.com/jax-releases/jax_cuda_releases.html,https://storage.googleapis.com/axlearn-wheels/wheels.html"
echo "UV links: ${UV_FIND_LINKS}"

uv pip install --find-links https://storage.googleapis.com/axlearn-wheels/wheels.html .[core,gpu,gcp]
pip uninstall -y jax jaxlib jax-cuda12-plugin
pip install -U --pre jax jaxlib jax-cuda12-plugin -i https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/ -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

echo "Installing JAX nightly"

JAX_VER=$(python3 -c 'import jax; print(jax.version.__version__)')
echo "JAX_VERSION_OUTPUT:${JAX_VER}"

gsutil -h "x-goog-meta-jax-version:${JAX_VER}" -m cp /dev/null ${GCS_PREFIX}/metadata/jax_version_tag_${GH_RUN_ID}

# Modify the batch size to account for B200
sed -i 's/train_batch_size=train_batch_size/train_batch_size=64/g' /root/axlearn/experiments/text/gpt/fuji.py

# Start the training loop
# python3 -m axlearn.common.launch_trainer_main --module=text.gpt.c4_trainer \
#     --config=fuji-7B-v2-flash-single-host \
#     --trainer_dir=${GCS_PREFIX}/gpu_single_host_test \
#     --data_dir=gs://axlearn-public/tensorflow_datasets \
#     --jax_backend=gpu \
#     --mesh_selector=gpu-a4-highgpu-8g-256 \
#     --trace_at_steps=5

echo "Starting 8 parallel training processes on this single host."

# 1. Define Cluster and Communication Variables (Single Host)
# The framework should be able to deduce WORLD_SIZE from this.
export WORLD_SIZE=8
# This is a fixed, high-numbered port for internal process communication
export MASTER_PORT=29500 
# Set the master address to the host's own internal IP
export MASTER_ADDR=$(hostname -i)
# Set the base training command
TRAINING_COMMAND="python3 -m axlearn.common.launch_trainer_main \
    --module=text.gpt.c4_trainer \
    --config=fuji-7B-v2-flash-single-host \
    --trainer_dir=${GCS_PREFIX}/gpu_single_host_test \
    --data_dir=gs://axlearn-public/tensorflow_datasets \
    --jax_backend=gpu \
    --mesh_selector=gpu-a4-highgpu-8g-256 \
    --trace_at_steps=5"

# Array to store Process IDs (PIDs)
PIDS=()

# 2. Loop and Fork 8 Processes (0 to 7)
for i in $(seq 0 7); do
    echo "Launching process $i assigned to GPU $i"
    
    # CRITICAL FIX: Define RANK and LOCAL_RANK only for the child process.
    # The variables are prefixed to the command, ensuring they are local environment variables.
    
    CUDA_VISIBLE_DEVICES=$i RANK=$i LOCAL_RANK=$i ${TRAINING_COMMAND} &
    
    # Store the process ID to wait for it later
    PIDS+=($!) 
done

# 3. Wait for all 8 background processes to complete
echo "Waiting for all 8 worker processes to complete..."
wait ${PIDS[*]}

# Check the exit status of the wait command (returns 0 if all processes were successful)
if [ $? -eq 0 ]; then
    echo "All training processes completed successfully."
    exit 0
else
    echo "One or more training processes failed."
    exit 1
fi