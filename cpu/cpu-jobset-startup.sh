#!/bin/bash

# Setup output directories
mkdir -p /home/runner/_work/csv_results

# Run the git configuration
bash /var/arc/git-setup.sh

# Install dependencies
export UV_FIND_LINKS="https://storage.googleapis.com/jax-releases/libtpu_releases.html,https://storage.googleapis.com/axlearn-wheels/wheels.html"
echo "UV links: ${UV_FIND_LINKS}"
cd /root && uv pip install .[core,dev,gcp,open_api,audio] pytest pytest-instafail pytest-xdist pytest-csv pytest-timeout

# Remove CUDA-enabled TensorFlow and install CPU-only variant
# TF_VER=$(pip freeze | grep -w tensorflow= | awk -F '==' {'print $2'})
# # Capture TF-Text version if it exists
# TF_TEXT_VER=$(pip freeze | grep -w tensorflow-text= | awk -F '==' {'print $2'})

# echo "Checking for tensorflow-cpu binary for version: $TF_VER"

# if uv pip install --no-deps --index-strategy unsafe-best-match "tensorflow-cpu==$TF_VER"; then
#     echo "Successfully swapped to exact CPU match: $TF_VER"
#     # Re-sync TF-Text to the exact version
#     uv pip install --no-deps --prerelease=allow --index-strategy unsafe-best-match "tensorflow-text==$TF_TEXT_VER"
# else
#     echo "Exact match $TF_VER not found. Attempting fallback to compatible patch..."
#     if uv pip install --no-deps --index-strategy unsafe-best-match "tensorflow-cpu~=${TF_VER%.*}"; then
#          echo "Fallback successful. Installed latest compatible 2.19.1.x variant."
#          # Re-sync TF-Text using the same fuzzy logic to ensure ABI compatibility
#          uv pip install --no-deps --prerelease=allow --index-strategy unsafe-best-match "tensorflow-text~=${TF_VER%.*}"
#     else
#          echo "CRITICAL: Could not find any compatible tensorflow-cpu. Keeping original install."
#     fi
# fi

# uv cache clean
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
