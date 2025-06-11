#!/bin/bash

# Fetch the latest version of AXLearn directly from upstream
cd /root
git clone --depth 1 https://github.com/apple/axlearn.git
cd /root/axlearn

# Install the GCP utils and PyTest suite
uv pip install .[core,gpu,gcp,tpu] pytest pytest-instafail allure-pytest torch pytest-xdist

# Clean any previous test results
rm -rf /home/runner/_work/xml_results
rm -rf /home/runner/_work/allure_results
rm pytest_xml.tar.gz allure_results.tar.gz

# Create a test output directory
mkdir -p /home/runner/_work/xml_results
mkdir -p /home/runner/_work/allure_results

# Create an array of tests, excluding anything for GPU or Vertex AI
readarray -d '' gcp_array < <(find axlearn -type f -name "*test*.py" ! -name "*gpu*" ! -name "*vertex*" ! -name "*tpu*" -print0)

pytest -v --junit-xml=/home/runner/_work/xml_results/cpu_tests.xml --alluredir /home/runner/_work/allure_results -n 8 $gcp_array

# Compress the results
tar -czvf pytest_xml.tar.gz /home/runner/_work/xml_results
tar -czvf allure_results.tar.gz /home/runner/_work/allure_results

# Upload to GCS, including the date and hostname inside the pod
gsutil -m cp pytest_xml.tar.gz ${GCS_PREFIX}/testing/$(date +"%Y-%m-%d-%T")-${HOSTNAME}_pytest_xml.tar.gz
gsutil -m cp allure_results.tar.gz ${GCS_PREFIX}/testing/$(date +"%Y-%m-%d-%T")-${HOSTNAME}_allure.tar.gz