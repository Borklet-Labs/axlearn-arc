#!/bin/bash

cd /root

# Get the timestamp of when the tests started
TIMESTAMP=$(date +"%Y-%m-%d-%T")
GITHUB_HASH=$(git log -1 --stat --pretty=format:"%h" --no-patch)
JAX_VER=$(python3 -c 'import jax; print(jax.version.__version__)')
GH_RUN_ID=$(cat /var/arc/run_id)

# Get CSV results for easier reading 
AXLEARN_CI_GPU_TESTS=1 pytest -v  \
    --csv /home/runner/_work/csv_results/gpu_tests.csv \
    -n 8 $(find axlearn/common -type f -name "*gpu*test*.py" ! -name '*gpu_client_test*' -printf '%p ') \
    --dist worksteal --timeout=30

# Compress the results
cd /home/runner/_work
tar -czvf results.tar.gz csv_results

# Upload to GCS, including the date and commit hash
gsutil -m cp results.tar.gz ${GCS_PREFIX}/results/archive/gpu-unit-tests-${GITHUB_HASH}-${JAX_VER}-${GH_RUN_ID}-${TIMESTAMP}.tar.gz
gsutil -h "x-goog-meta-test-type:unit-tests" -h "x-goog-meta-processor:gpu" \
   -h "x-goog-meta-commit-hash:${GITHUB_HASH}" -h "x-goog-meta-jax-version:${JAX_VER}" \
   -h "x-goog-meta-github-run-id:${GH_RUN_ID}" -h "x-goog-meta-run-timestamp:${TIMESTAMP}" \
   -m cp /home/runner/_work/csv_results/gpu_tests.csv ${GCS_PREFIX}/results/unit-tests-gpu-${GITHUB_HASH}-${JAX_VER}-${GH_RUN_ID}-${TIMESTAMP}.csv

# Check to see if there were any real test failures
if grep -q ",failed," /home/runner/_work/csv_results/gpu_tests.csv; then
    echo "Test failures detected"
    touch /home/runner/_work/test_failed
else
    echo "All tests passed / skipped"
fi

[[ ! -f /home/runner/_work/test_failed ]] && echo "All tests passed successfully"