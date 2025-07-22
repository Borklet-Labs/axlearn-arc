#!/bin/bash

cd /root

# Set ulimit to avoid crashes with newer versions of containerd
echo "Setting ulimit to 1,000,000 before tests"
ulimit -n 1000000

# Get CSV results for easier reading
AXLEARN_CI_GPU_TESTS=1 pytest -v  \
    --csv /home/runner/_work/csv_results/gpu_tests.csv \
    -n 8 $(find axlearn/common -type f -name "*gpu*test*.py" -printf '%p ') \
    --dist worksteal --timeout=120

# Compress the results
cd /home/runner/_work
tar -czvf results.tar.gz csv_results

# Upload to GCS, including the date and hostname inside the pod
timestamp=$(date +"%Y-%m-%d-%T")
gsutil -m cp results.tar.gz ${GCS_PREFIX}/results/gpu-unit-tests-${timestamp}-${HOSTNAME}.tar.gz
gsutil -m cp /home/runner/_work/csv_results/gpu_tests.csv ${GCS_PREFIX}/results/gpu-unit-tests-${timestamp}-${HOSTNAME}.csv

# Check to see if there were any real test failures
if grep -q ",failed," /home/runner/_work/csv_results/gpu_tests.csv; then
    echo "Test failures detected"
    touch /home/runner/_work/test_failed
else
    echo "All tests passed / skipped"
fi

[[ ! -f /home/runner/_work/test_failed ]] && echo "All tests passed successfully"