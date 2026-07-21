#!/bin/bash

# Setup output directories
mkdir -p /home/runner/_work/csv_results && mkdir -p /var/arc/bazel_cache

# Run the git configuration
bash /var/arc/git-setup.sh

if [ "$ENABLE_JAX_DEV" == "true" ]; then
    echo "Enabling prerelease Jax and specifying extra idnex"
    export UV_PRELEASE=allow
    export UV_INDEX=https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/
fi

# Validate that we are able to calculate dependencies
cd /root/axlearn && uv pip install --dry-run .[core,dev,gcp,audio,orbax] 
if [ $? -ne 0 ]; then
    echo "Unable to successfully resolve dependencies. Exiting before testing fails." && exit 1
fi

# Run any post-setup command if defined and not set to INSERT_POST_SETUP_CMD
if [ "$POST_SETUP_CMD" != "INSERT_POST_SETUP_CMD" ]; then
    eval "$POST_SETUP_CMD"
fi

# Generate the requirements lock file
bazel run //:requirements.update
if [ $? -ne 0 ]; then
    echo "Unable to update requirements via bazel run //:requirements.update" && exit 1
fi

# Copy the requirements lock to the right destination
cp -L -v bazel-bin/requirements.out /root/axlearn/requirements_lock.txt

# Run the unit tests
if [ $? -eq 0 ]; then
    bash /var/arc/bazel-unit-tests.sh 2>&1 | tee -a /home/runner/_work/csv_results/full_output.log
else
    exit 1
fi

# See if we got a pass or fail
[[ ! -f /home/runner/_work/test_failed ]] && echo "All tests passed successfully" && exit 0 || exit 1