#!/bin/bash

cd /root

# Fetch resources needed
mkdir -p axlearn/data/tokenizers/sentencepiece
mkdir -p axlearn/data/tokenizers/bpe
curl https://huggingface.co/t5-base/resolve/main/spiece.model -o axlearn/data/tokenizers/sentencepiece/t5-base
curl https://huggingface.co/FacebookAI/roberta-base/raw/main/merges.txt -o axlearn/data/tokenizers/bpe/roberta-base-merges.txt
curl https://huggingface.co/FacebookAI/roberta-base/raw/main/vocab.json -o axlearn/data/tokenizers/bpe/roberta-base-vocab.json

# Create eight groups of pytest files, needed to avoid OOM with stack trace failures
groups=(00 01 02 03 04)
find axlearn -type f -name "*_test*.py" ! -name "*gpu*" ! -name "*vertex*" ! -name "*tpu*" > pytest_files.txt
split -n r/5 -a 2 -d pytest_files.txt split_pytest_files_
touch /home/runner/_work/csv_results/cpu_tests_all_results.csv

for i in ${groups[@]}; do
    echo Starting standard tests for group ${i}
    pytest -v --junit-xml=/home/runner/_work/xml_results/cpu_tests_${i}.xml \
        --csv /home/runner/_work/csv_results/cpu_tests_${i}.csv \
        -n auto -m "not (gs_login or tpu or high_cpu or fp64 or for_8_devices)" \
        $(tr '\n' ' ' < split_pytest_files_${i}) --dist worksteal \
        || touch /home/runner/_work/test_failed
    echo Adding results from group ${i} to master CSV
    if [[ "$i" == 00 ]]; then
        cat /home/runner/_work/csv_results/cpu_tests_${i}.csv >> /home/runner/_work/csv_results/cpu_tests_all_results.csv
    else
        sed -i '1d' /home/runner/_work/csv_results/cpu_tests_${i}.csv && \
        cat /home/runner/_work/csv_results/cpu_tests_${i}.csv >> /home/runner/_work/csv_results/cpu_tests_all_results.csv
    fi
done

for i in ${groups[@]}; do
    echo Starting FP64 for group ${i}
    JAX_ENABLE_X64=1 pytest -v --junit-xml=/home/runner/_work/xml_results/cpu_tests_fp64_${i}.xml \
        --csv /home/runner/_work/csv_results/cpu_tests_fp64_${i}.csv \
        -n auto -m "fp64" \
        $(tr '\n' ' ' < split_pytest_files_${i}) --dist worksteal \
        || touch /home/runner/_work/test_failed
    echo Adding results from group ${i} to master CSV
    sed -i '1d' /home/runner/_work/csv_results/cpu_tests_fp64_${i}.csv && \
    cat /home/runner/_work/csv_results/cpu_tests_fp64_${i}.csv >> /home/runner/_work/csv_results/cpu_tests_all_results.csv
done

for i in ${groups[@]}; do
    echo Starting for_8_devices tests for group ${i}
    XLA_FLAGS="--xla_force_host_platform_device_count=8" \
    pytest -v --junit-xml=/home/runner/_work/xml_results/cpu_tests_for_8_devices_${i}.xml \
        --csv /home/runner/_work/csv_results/cpu_tests_for_8_devices_${i}.csv \
        -n auto -m "for_8_devices" \
        $(tr '\n' ' ' < split_pytest_files_${i}) --dist worksteal \
        || touch /home/runner/_work/test_failed
    echo Adding results from group ${i} to master CSV
    sed -i '1d' /home/runner/_work/csv_results/cpu_tests_for_8_devices_${i}.csv && \
    cat /home/runner/_work/csv_results/cpu_tests_for_8_devices_${i}.csv >> /home/runner/_work/csv_results/cpu_tests_all_results.csv
done

# Compress the results
cd /home/runner/_work
tar -czvf results.tar.gz xml_results csv_results
timestamp=$(date +"%Y-%m-%d-%T")

# Upload to GCS, including the date and hostname inside the pod
gsutil -m cp results.tar.gz ${GCS_PREFIX}/testing/cpu-unit-tests-${timestamp}-${HOSTNAME}.tar.gz
gsutil -m cp /home/runner/_work/csv_results/cpu_tests_all_results.csv ${GCS_PREFIX}/testing/cpu-unit-tests-${timestamp}-${HOSTNAME}-all.csv

[[ ! -f /home/runner/_work/test_failed ]] && echo "All tests passed successfully"