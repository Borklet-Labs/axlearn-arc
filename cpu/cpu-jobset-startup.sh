#!/bin/bash

# Setup output directories
mkdir -p /home/runner/_work/csv_results

# Run the git configuration
bash /var/arc/git-setup.sh

# Install dependencies
export UV_FIND_LINKS="https://storage.googleapis.com/jax-releases/libtpu_releases.html,https://storage.googleapis.com/axlearn-wheels/wheels.html"
echo "UV links: ${UV_FIND_LINKS}"
cd /root && uv pip install --prerelease=allow .[core,dev,gcp,open_api,audio,orbax] pytest pytest-instafail pytest-xdist pytest-csv pytest-timeout

# Run the unit tests
if [ $? -eq 0 ]; then
    bash /var/arc/cpu-unit-tests.sh 2>&1 | tee -a /home/runner/_work/csv_results/full_output.log
else
    exit 1
fi

# See if we got a pass or fail
[[ ! -f /home/runner/_work/test_failed ]] && echo "All tests passed successfully" && exit 0 || exit 1