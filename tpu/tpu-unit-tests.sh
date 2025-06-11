#!/bin/bash

cd /root/axlearn

pytest -v --junit-xml=/home/runner/_work/xml_results/tpu_tests.xml --alluredir /home/runner/_work/allure_results axlearn/common/flash_attention/tpu_attention_test.py || touch /home/runner/_work/test_failed

# Compress the results
tar -czvf pytest_xml.tar.gz /home/runner/_work/xml_results
tar -czvf allure_results.tar.gz /home/runner/_work/allure_results

# Upload to GCS, including the date and hostname inside the pod
gsutil -m cp pytest_xml.tar.gz ${GCS_PREFIX}/testing/tpu-unit-tests-$(date +"%Y-%m-%d-%T")-${HOSTNAME}_pytest_xml.tar.gz
gsutil -m cp allure_results.tar.gz ${GCS_PREFIX}/testing/tpu-unit-test-$(date +"%Y-%m-%d-%T")-${HOSTNAME}_allure.tar.gz

[[ ! -f /home/runner/_work/test_failed ]] && echo "All tests passed successfully"