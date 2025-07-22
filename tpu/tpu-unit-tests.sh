#!/bin/bash

cd /root

# Get the XML output and the CSV results for easier reading
pytest -v --csv /home/runner/_work/csv_results/tpu_tests.csv \
    -n auto $(find axlearn/common -type f -name "*tpu*test*.py" -printf '%p ') \
    --dist worksteal --timeout=120

# Compress the results
cd /home/runner/_work
tar -czvf results.tar.gz csv_results

# Upload to GCS, including the date and hostname inside the pod
timestamp=$(date +"%Y-%m-%d-%T")
gsutil -m cp results.tar.gz ${GCS_PREFIX}/results/tpu-unit-tests-${timestamp}-${HOSTNAME}.tar.gz
gsutil -m cp /home/runner/_work/csv_results/tpu_tests.csv ${GCS_PREFIX}/results/tpu-unit-tests-${timestamp}-${HOSTNAME}.csv

# Check to see if there were any real test failures
if grep -q ",failed," /home/runner/_work/csv_results/tpu_tests.csv; then
    echo "Test failures detected"
    touch /home/runner/_work/test_failed
else
    echo "All tests passed / skipped"
fi

[[ ! -f /home/runner/_work/test_failed ]] && echo "All tests passed successfully"