#!/bin/bash

# Setup output directories
mkdir -p /home/runner/_work/csv_results

# Run the git configuration
bash /var/arc/git-setup.sh

if [ "$ENABLE_JAX_DEV" == "true" ]; then
    echo "Enabling prerelease Jax and specifying extra idnex"
    export UV_PRELEASE=allow
    export UV_INDEX=https://us-python.pkg.dev/ml-oss-artifacts-published/jax/simple/
fi

# Install dependencies
cd /root && uv pip install .[core,dev,gcp,open_api,audio,orbax] pytest pytest-instafail pytest-xdist pytest-csv pytest-timeout

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