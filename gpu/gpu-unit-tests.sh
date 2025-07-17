#!/bin/bash

cd /root

# Get the XML output and the CSV results for easier reading
AXLEARN_CI_GPU_TESTS=1 pytest -v --junit-xml=/home/runner/_work/xml_results/gpu_results.xml \
    --csv /home/runner/_work/csv_results/gpu_tests.csv \
    -n 8 $(find axlearn/common -type f -name "*gpu*test*.py" -printf '%p ') \
    || touch /home/runner/_work/test_failed

# Compress the results
cd /home/runner/_work
tar -czvf results.tar.gz xml_results csv_results

# Upload to GCS, including the date and hostname inside the pod
gsutil -m cp results.tar.gz ${GCS_PREFIX}/results/gpu-unit-tests-$(date +"%Y-%m-%d-%T")-${HOSTNAME}.tar.gz

[[ ! -f /home/runner/_work/test_failed ]] && echo "All tests passed successfully"