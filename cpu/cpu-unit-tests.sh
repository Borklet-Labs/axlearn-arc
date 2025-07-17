#!/bin/bash

pytest -v --junit-xml=/home/runner/_work/xml_results/cpu_tests.xml \
    --alluredir /home/runner/_work/allure_results --csv /home/runner_work/csv_results/ \
    -n 8 $(find axlearn -type f -name "*test*.py" ! -name "*gpu*" ! -name "*vertex*" ! -name "*tpu*" -printf '%p ') \
    || touch /home/runner/_work/test_failed

# Compress the results
cd /home/runner/_work
tar -czvf results.tar.gz xml_results allure_results csv_results

# Upload to GCS, including the date and hostname inside the pod
gsutil -m cp results.tar.gz ${GCS_PREFIX}/testing/cpu-unit-tests-$(date +"%Y-%m-%d-%T")-${HOSTNAME}.tar.gz

[[ ! -f /home/runner/_work/test_failed ]] && echo "All tests passed successfully"