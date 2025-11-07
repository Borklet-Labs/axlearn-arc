#!/bin/bash

# Setup output directories
mkdir -p /home/runner/_work/csv_results

# Run the git configuration
bash /var/arc/git-setup.sh

# Install dependencies
cd /root && uv pip install .[core,dev,gcp,open_api,audio,orbax] pytest pytest-instafail pytest-xdist pytest-csv pytest-timeout

# Remove CUDA-enabled TensorFlow and install CPU-only variant
TF_VER=$(pip freeze | grep -w tensorflow= | awk -F '==' {'print $2'}) && \
    pip uninstall -y -qq tensorflow && \
    uv pip install --no-deps tensorflow-cpu==$TF_VER && \
    uv cache clean

# Run any post-setup command if defined and not set to INSERT_POST_SETUP_CMD
if [ "$POST_SETUP_CMD" != "INSERT_POST_SETUP_CMD" ]; then
    eval "$POST_SETUP_CMD"
fi

# Run the unit tests
if [ $? -eq 0 ]; then
    bash /var/arc/cpu-unit-tests.sh 2>&1 | tee -a /home/runner/_work/csv_results/full_output.log
else
    exit 1
fi

# See if we got a pass or fail
[[ ! -f /home/runner/_work/test_failed ]] && echo "All tests passed successfully" && exit 0 || exit 1