#!/bin/bash

# Fetch the latest version of AXLearn directly from upstream
cd /root
git clone --depth 1 https://github.com/apple/axlearn.git
cd /root/axlearn

# Modify the verbosity with TPU since the output can be overwhelming
export TF_CPP_MIN_LOG_LEVEL=2
export TPU_STDERR_LOG_LEVEL=2
export TPU_MIN_LOG_LEVEL=2

# Install the GCP utils and PyTest suite
uv pip install .[core,tpu,gcp,dev] pytest pytest-instafail allure-pytest torch

# Clean any previous test results
rm -rf /root/axlearn/xml_results
rm -rf /root/axlearn/allure_results
rm pytest_xml.tar.gz allure_results.tar.gz

# Create a test output directory
mkdir -p /root/axlearn/xml_results
mkdir -p /root/axlearn/allure_results

# Create an array of tests, excluding anything for GPU or Vertex AI
readarray -d '' gcp_array < <(find axlearn/cloud/gcp -type f -name "*test*.py" ! -name "*gpu*" ! -name "*vertex*" -print0)

# Iterate over the tests in axlearn/cloud/gcp
for test in "${gcp_array[@]}"; do
    # Change the / in the filename to - for readability
    export output_name=$(sed 's/\//\-/g' <<< "$test")
    echo "[GCP] Testing $test..."
    # Get the XML output and the Allure results for easier reading
    pytest -v --junit-xml=/root/axlearn/xml_results/$output_name.xml --alluredir /root/axlearn/allure_results $test
done

# Create an array of tests, excluding anything for TPU, GPU, Vertex AI, or Neuron (AWS) -- defer layer_test.py until later since it takes a while. Ignore FP8 since it's not supported
readarray -d '' common_array < <(find axlearn/common -type f -name "*test*.py" ! -name "*tpu*" ! -name "*vertex*" ! -name "*neuron*" ! -name "*layer_test*.py" ! -name "*gpu*" ! -name "*quantized*" -print0)

for test in "${common_array[@]}"; do
    # Change the / in the filename to - for readability
    export output_name=$(sed 's/\//\-/g' <<< "$test")
    echo "[COMMON] Testing $test..."
    # Get the XML output and the Allure results for easier reading
    pytest -v --junit-xml=/root/axlearn/xml_results/$output_name.xml --alluredir /root/axlearn/allure_results $test
done

# Create an array of TPU tests
readarray -d '' tpu_array < <(find axlearn/common -type f -name "*tpu*test*.py" -print0)

for test in "${tpu_array[@]}"; do
    # Change the / in the filename to - for readability
    export output_name=$(sed 's/\//\-/g' <<< "$test")
    echo "[TPU] Testing $test..."
    # Get the XML output and the Allure results for easier reading
    pytest -v --junit-xml=/root/axlearn/xml_results/$output_name.xml --alluredir /root/axlearn/allure_results $test
done

# Finish with axlearn/common/flash_attention/layer_test.py since this one takes a long time
#echo "[COMMON] Testing axlearn/common/flash_attention/layer_test.py..."
#pytest -v --junit-xml=/root/axlearn/xml_results/axlearn-common-flash_attention-layer_test.py.xml --alluredir /root/axlearn/allure_results axlearn/common/flash_attention/layer_test.py

# Create an array of experimental tests
readarray -d '' experimental_array < <(find axlearn/experiments -type f -name "*test*.py" -print0)

for test in "${experimental_array[@]}"; do
    # Change the / in the filename to - for readability
    export output_name=$(sed 's/\//\-/g' <<< "$test")
    echo "[Experiments] Testing $test..."
    # Get the XML output and the Allure results for easier reading
    pytest -v --junit-xml=/root/axlearn/xml_results/$output_name.xml --alluredir /root/axlearn/allure_results $test
done

# Compress the results
tar -czvf pytest_xml.tar.gz axlearn/xml_results
tar -czvf allure_results.tar.gz axlearn/allure_results

# Upload to GCS, including the date and hostname inside the pod
gsutil -m cp pytest_xml.tar.gz gs://supercomputer-testing-axlearn/a4-sa-test/testing/$(date +"%Y-%m-%d-%T")-${HOSTNAME}_pytest_xml.tar.gz
gsutil -m cp allure_results.tar.gz gs://supercomputer-testing-axlearn/a4-sa-test/testing/$(date +"%Y-%m-%d-%T")-${HOSTNAME}_allure.tar.gz