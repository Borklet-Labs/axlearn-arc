#!/bin/bash

cd /root

# Get the timestamp of when the tests started
TIMESTAMP=$(date +"%Y-%m-%d-%T")
GITHUB_HASH=$(git log -1 --stat --pretty=format:"%h" --no-patch)
JAX_VER=$(python3 -c 'import jax; print(jax.version.__version__)')

pytest -v  \
    --csv /home/runner/_work/csv_results/golden_config_test.csv \
    -n auto --durations=100 --dist worksteal --timeout=60 \
    axlearn/experiments/golden_config_test.py

if [ -f /home/runner/_work/csv_results/golden_config_test.csv ]; then
    echo Checking for test failures
    if grep -q ",failed," /home/runner/_work/csv_results/golden_config_test.csv; then
        echo "Test failures detected"
        touch /home/runner/_work/test_failed
    else
        echo "All tests passed / skipped"
    fi
fi

# Compress the results
cd /home/runner/_work
tar -czvf results.tar.gz csv_results

# Upload to GCS, including the date and commit hash
gsutil -m cp results.tar.gz ${GCS_PREFIX}/results/archive/golden-config-test-${GITHUB_HASH}-${JAX_VER}-${GH_RUN_ID}-${TIMESTAMP}.tar.gz
gsutil -m cp /home/runner/_work/csv_results/golden_config_test.csv ${GCS_PREFIX}/results/golden-config-test-${GITHUB_HASH}-${JAX_VER}-${GH_RUN_ID}-${TIMESTAMP}.csv

exit