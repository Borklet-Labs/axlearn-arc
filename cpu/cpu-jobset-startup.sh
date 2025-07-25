#!/bin/bash

# Setup output directories
mkdir -p /home/runner/_work/csv_results

# Run the git configuration
bash /var/arc/git-setup.sh

# Install dependencies
cd /root && uv pip install .[core,dev,gcp,open_api,audio] pytest pytest-instafail pytest-xdist pytest-csv pytest-timeout

# Run the unit tests
bash /var/arc/cpu-unit-tests.sh

# See if we got a pass or fail
[[ ! -f /home/runner/_work/test_failed ]] && echo "All tests passed successfully" && exit 0 || exit 1