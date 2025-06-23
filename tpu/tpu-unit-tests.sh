#!/bin/bash

# Set ulimit before running tests
echo "Setting ulimit before tests"
ulimit -n 1000000

cd /root

#pytest -v --junit-xml=/home/runner/_work/xml_results/tpu_tests.xml --alluredir /home/runner/_work/allure_results axlearn/common/flash_attention/tpu_attention_test.py || touch /home/runner/_work/test_failed
pytest -v --junit-xml=/home/runner/_work/xml_results/orbax_tests.xml --alluredir /home/runner/_work/allure_results axlearn/common/checkpointer_orbax_emergency_test.py || touch /home/runner/_work/test_failed
pytest -v --junit-xml=/home/runner/_work/xml_results/tpu_tests.xml --alluredir /home/runner/_work/allure_results $(find axlearn/common -type f -name "*tpu*test*.py" -printf '%p ') || touch /home/runner/_work/test_failed

# Compress the results
cd /home/runner/_work
tar -czvf results.tar.gz xml_results allure_results

# Upload to GCS, including the date and hostname inside the pod
gsutil -m cp results.tar.gz ${GCS_PREFIX}/testing/tpu-unit-tests-$(date +"%Y-%m-%d-%T")-${HOSTNAME}.tar.gz

[[ ! -f /home/runner/_work/test_failed ]] && echo "All tests passed successfully"