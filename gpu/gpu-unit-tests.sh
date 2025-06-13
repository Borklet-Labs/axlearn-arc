#!/bin/bash

cd /root

# Create an array of GPU tests
readarray -d '' gpu_array < <(find axlearn/common -type f -name "*gpu*test*.py" -print0)

for test in "${gpu_array[@]}"; do
    # Change the / in the filename to - for readability
    export output_name=$(sed 's/\//\-/g' <<< "$test")
    echo "[GPU] Testing $test..."
    # Get the XML output and the Allure results for easier reading
    AXLEARN_CI_GPU_TESTS=1 pytest -v --junit-xml=/home/runner/_work/xml_results/$output_name.xml --alluredir /home/runner/_work/allure_results -n 8 $test || touch /home/runner/_work/test_failed
done

# Compress the results
cd /home/runner/_work
tar -czvf results.tar.gz xml_results allure_results

# Upload to GCS, including the date and hostname inside the pod
gsutil -m cp results.tar.gz ${GCS_PREFIX}/testing/gpu-unit-tests-$(date +"%Y-%m-%d-%T")-${HOSTNAME}.tar.gz

[[ ! -f /home/runner/_work/test_failed ]] && echo "All tests passed successfully"