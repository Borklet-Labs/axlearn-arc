# ... (All setup remains the same) ...

# 1. Define Cluster and Communication Variables (Single Host)
# The framework should be able to deduce WORLD_SIZE from this.
export WORLD_SIZE=8
# This is a fixed, high-numbered port for internal process communication
export MASTER_PORT=29500 
# Set the master address to the host's own internal IP
export MASTER_ADDR=$(hostname -i)
# Set the coordinator for the framework
export DISTRIBUTED_COORDINATOR="${MASTER_ADDR}:${MASTER_PORT}"
export JAX_PROCESS_COUNT=8 # Total number of processes in the collective (8)

# Set the base training command
TRAINING_COMMAND="python3 -m axlearn.common.launch_trainer_main \
    --module=text.gpt.c4_trainer \
    --config=fuji-7B-v2-flash-single-host \
    --trainer_dir=${GCS_PREFIX}/gpu_single_host_test \
    --data_dir=gs://axlearn-public/tensorflow_datasets \
    --jax_backend=gpu \
    --mesh_selector=gpu-a4-highgpu-8g-256 \
    --trace_at_steps=5"

# Array to store Process IDs (PIDS)
PIDS=()

echo "Starting 8 parallel training processes on this single host."

# 2. Loop and Fork 8 Processes (0 to 7) - Use only this loop
for i in $(seq 0 7); do
    echo "Launching process $i assigned to GPU $i"
    
    # Pass all necessary variables to the child process environment on a single line
    CUDA_VISIBLE_DEVICES=$i RANK=$i LOCAL_RANK=$i JAX_PROCESS_INDEX=$i ${TRAINING_COMMAND} &
    
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