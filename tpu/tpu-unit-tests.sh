#!/bin/bash

cd /root

# Get the timestamp of when the tests started
TIMESTAMP=$(date +"%Y-%m-%d-%T")
GITHUB_HASH=$(git log -1 --stat --pretty=format:"%h" --no-patch)

# Set ulimit to avoid crashes with newer versions of containerd
echo "Setting ulimit to 1,000,000 before tests"
ulimit -n 1000000

# Get the CSV results for easier reading, limit to 1 process for TPU
pytest -v --csv /home/runner/_work/csv_results/tpu_tests.csv \
    -n 1 $(find axlearn/common -type f -name "*tpu*test*.py" -printf '%p ') \
    --dist worksteal --timeout=30

# Compress the results
cd /home/runner/_work
tar -czvf results.tar.gz csv_results

# Upload to GCS, including the date and hash of the commit
gsutil -m cp results.tar.gz ${GCS_PREFIX}/results/archive/tpu-unit-tests-${GITHUB_HASH}-${TIMESTAMP}.tar.gz
gsutil -m cp /home/runner/_work/csv_results/tpu_tests.csv ${GCS_PREFIX}/results/unit-tests-tpu-${GITHUB_HASH}-${TIMESTAMP}.csv

# Check to see if there were any real test failures
if grep -q ",failed," /home/runner/_work/csv_results/tpu_tests.csv; then
    echo "Test failures detected"
    touch /home/runner/_work/test_failed
else
    echo "All tests passed / skipped"
fi

[[ ! -f /home/runner/_work/test_failed ]] && echo "All tests passed successfully"