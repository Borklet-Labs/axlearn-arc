#!/bin/bash

# Fetch the latest version of AXLearn directly from upstream
cd /root
git clone --depth 1 https://github.com/apple/axlearn.git
cd /root/axlearn

# Install the GCP utils and PyTest suite
uv pip install .[core,gpu,gcp] pytest pytest-instafail allure-pytest torch

# Clean any previous test results
rm -rf /home/runner/_work/xml_results
rm -rf /home/runner/_work/allure_results
rm pytest_xml.tar.gz allure_results.tar.gz

# Create a test output directory
mkdir -p /home/runner/_work/xml_results
mkdir -p /home/runner/_work/allure_results

# Create an array of GPU tests
readarray -d '' gpu_array < <(find axlearn/common -type f -name "*gpu*test*.py" -print0)

for test in "${gpu_array[@]}"; do
    # Change the / in the filename to - for readability
    export output_name=$(sed 's/\//\-/g' <<< "$test")
    echo "[GPU] Testing $test..."
    # Get the XML output and the Allure results for easier reading
    pytest -v --junit-xml=/home/runner/_work/xml_results/$output_name.xml --alluredir /home/runner/_work/allure_results $test
done

# Compress the results
tar -czvf pytest_xml.tar.gz /home/runner/_work/xml_results
tar -czvf allure_results.tar.gz /home/runner/_work/allure_results

# Upload to GCS, including the date and hostname inside the pod
gsutil -m cp pytest_xml.tar.gz gs://supercomputer-testing-axlearn/a4-sa-test/testing/$(date +"%Y-%m-%d-%T")-${HOSTNAME}_pytest_xml.tar.gz
gsutil -m cp allure_results.tar.gz gs://supercomputer-testing-axlearn/a4-sa-test/testing/$(date +"%Y-%m-%d-%T")-${HOSTNAME}_allure.tar.gz