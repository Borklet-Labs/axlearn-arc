#!/bin/bash

# Fetch the latest version of AXLearn directly from upstream
cd /root
git clone --depth 1 https://github.com/apple/axlearn.git
cd /root/axlearn

# Install the GCP utils and PyTest suite
uv pip install .[core,gpu,gcp,dev] pytest pytest-instafail allure-pytest torch

# Create an array of tests, excluding anything for TPU or Vertex AI
readarray -d '' gcp_array < <(find axlearn/cloud/gcp -type f -name "*test*.py" ! -name "*tpu*" ! -name "*vertex*" -print0)

# Clean any previous test results
rm -rf /root/axlearn/xml_results
rm -rf /root/axlearn/allure_results
rm pytest_xml.tar.gz allure_results.tar.gz

# Create a test output directory
mkdir -p /root/axlearn/xml_results
mkdir -p /root/axlearn/allure_results

# Iterate over the tests in axlearn/cloud/gcp
for test in "${gcp_array[@]}"; do
    # Change the / in the filename to - for readability
    export output_name=$(sed 's/\//\-/g' <<< "$test")
    echo "[GCP] Testing $test..."
    # Get the XML output and the Allure results for easier reading
    CUDA_VISIBLE_DEVICES=0 pytest -v --junit-xml=/root/axlearn/xml_results/$output_name.xml --alluredir /root/axlearn/allure_results $test
done

# Create an array of tests, excluding anything for TPU, GPU, Vertex AI, or Neuron (AWS) -- defer layer_test.py until later since it takes a while
readarray -d '' common_array < <(find axlearn/common -type f -name "*test*.py" ! -name "*tpu*" ! -name "*vertex*" ! -name "*neuron*" ! -name "*layer_test*.py" ! -name "*gpu*" -print0)

for test in "${common_array[@]}"; do
    # Change the / in the filename to - for readability
    export output_name=$(sed 's/\//\-/g' <<< "$test")
    echo "[COMMON] Testing $test..."
    # Get the XML output and the Allure results for easier reading
    CUDA_VISIBLE_DEVICES=0 pytest -v --junit-xml=/root/axlearn/xml_results/$output_name.xml --alluredir /root/axlearn/allure_results $test
done

# Create an array of GPU tests
readarray -d '' gpu_array < <(find axlearn/common -type f -name "*gpu*test*.py" -print0)

for test in "${gpu_array[@]}"; do
    # Change the / in the filename to - for readability
    export output_name=$(sed 's/\//\-/g' <<< "$test")
    echo "[GPU] Testing $test..."
    # Get the XML output and the Allure results for easier reading
    pytest -v --junit-xml=/root/axlearn/xml_results/$output_name.xml --alluredir /root/axlearn/allure_results $test
done

# Finish with axlearn/common/flash_attention/layer_test.py since this one takes a long time
echo "[COMMON] Testing axlearn/common/flash_attention/layer_test.py..."
pytest -v --junit-xml=/root/axlearn/xml_results/axlearn-common-flash_attention-layer_test.py.xml --alluredir /root/axlearn/allure_results axlearn/common/flash_attention/layer_test.py

# Compress the results
tar -czvf pytest_xml.tar.gz axlearn/xml_results
tar -czvf allure_results.tar.gz axlearn/allure_results

# Upload to GCS, including the date and hostname inside the pod
gsutil -m cp pytest_xml.tar.gz gs://supercomputer-testing-axlearn/a4-sa-test/testing/$(date +"%Y-%m-%d-%T")-${HOSTNAME}_pytest_xml.tar.gz
gsutil -m cp allure_results.tar.gz gs://supercomputer-testing-axlearn/a4-sa-test/testing/$(date +"%Y-%m-%d-%T")-${HOSTNAME}_allure.tar.gz