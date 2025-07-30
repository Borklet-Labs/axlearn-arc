#!/bin/bash

cd /root

# Get the timestamp of when the tests started
TIMESTAMP=$(date +"%Y-%m-%d-%T")
GITHUB_HASH=$(git log -1 --stat --pretty=format:"%h" --no-patch)
JAX_VER=$(python3 -c 'import jax; print(jax.version.__version__)')

# Set ulimit to avoid crashes with newer versions of containerd
echo "Setting ulimit to 1,000,000 before tests"
ulimit -n 1000000

# Fetch resources needed
mkdir -p axlearn/data/tokenizers/sentencepiece
mkdir -p axlearn/data/tokenizers/bpe
curl https://huggingface.co/t5-base/resolve/main/spiece.model -o axlearn/data/tokenizers/sentencepiece/t5-base
curl https://huggingface.co/FacebookAI/roberta-base/raw/main/merges.txt -o axlearn/data/tokenizers/bpe/roberta-base-merges.txt
curl https://huggingface.co/FacebookAI/roberta-base/raw/main/vocab.json -o axlearn/data/tokenizers/bpe/roberta-base-vocab.json

# Create five groups of pytest files, needed to avoid OOM with stack trace failures
groups=(00 01 02 03 04)
find axlearn -type f -name "*_test*.py" ! -name "*gpu*" ! -name "*vertex*" ! -name "*tpu*" > pytest_files.txt
split -n r/5 -a 2 -d pytest_files.txt split_pytest_files_
touch /home/runner/_work/csv_results/cpu_tests_all_results.csv

for i in ${groups[@]}; do
    echo Starting standard tests for group ${i}
    pytest -v  \
        --csv /home/runner/_work/csv_results/cpu_tests_${i}.csv \
        -n auto -m "not (gs_login or tpu or high_cpu or fp64 or for_8_devices)" --durations=100 --dist worksteal --timeout=60 \
        $(tr '\n' ' ' < split_pytest_files_${i}) 

    echo Adding results from group ${i} to master CSV
    if [[ "$i" == 00 ]]; then
        cat /home/runner/_work/csv_results/cpu_tests_${i}.csv >> /home/runner/_work/csv_results/cpu_tests_all_results.csv
    else
        sed -i '1d' /home/runner/_work/csv_results/cpu_tests_${i}.csv && \
        cat /home/runner/_work/csv_results/cpu_tests_${i}.csv >> /home/runner/_work/csv_results/cpu_tests_all_results.csv \
        || echo "No result CSV found for group ${i}. Skipping..."
    fi

    if [ -f /home/runner/_work/csv_results/cpu_tests_${i}.csv ]; then
        echo Checking for test failures
        if grep -q ",failed," /home/runner/_work/csv_results/cpu_tests_${i}.csv; then
            echo "Test failures detected in group ${i}"
            touch /home/runner/_work/test_failed
        else
            echo "All tests passed / skipped in group ${i}"
        fi
    fi

    echo Starting FP64 for group ${i}
    JAX_ENABLE_X64=1 pytest -v \
        --csv /home/runner/_work/csv_results/cpu_tests_fp64_${i}.csv \
        -n auto -m "fp64" --durations=100 --dist worksteal --timeout=60 \
        $(tr '\n' ' ' < split_pytest_files_${i})

    echo Adding results from group ${i} to master CSV
    sed -i '1d' /home/runner/_work/csv_results/cpu_tests_fp64_${i}.csv && \
    cat /home/runner/_work/csv_results/cpu_tests_fp64_${i}.csv >> /home/runner/_work/csv_results/cpu_tests_all_results.csv \
        || echo "No result CSV found for group ${i}. Skipping..."

    if [ -f /home/runner/_work/csv_results/cpu_tests_fp64_${i}.csv ]; then
        echo Checking for test failures
        if grep -q ",failed," /home/runner/_work/csv_results/cpu_tests_fp64_${i}.csv; then
            echo "Test failures detected in group FP64 ${i}"
            touch /home/runner/_work/test_failed
        else
            echo "All tests passed / skipped in group FP64 ${i}"
        fi
    fi

    echo Starting for_8_devices tests for group ${i}
    XLA_FLAGS="--xla_force_host_platform_device_count=8" \
    pytest -v \
        --csv /home/runner/_work/csv_results/cpu_tests_for_8_devices_${i}.csv \
        -n auto -m "for_8_devices" --durations=100 --dist worksteal --timeout=60 \
        $(tr '\n' ' ' < split_pytest_files_${i}) 

    echo Adding results from group ${i} to master CSV
    sed -i '1d' /home/runner/_work/csv_results/cpu_tests_for_8_devices_${i}.csv && \
    cat /home/runner/_work/csv_results/cpu_tests_for_8_devices_${i}.csv >> /home/runner/_work/csv_results/cpu_tests_all_results.csv \
        || echo "No result CSV found for group ${i}. Skipping..."

    if [ -f /home/runner/_work/csv_results/cpu_tests_for_8_devices_${i}.csv ]; then
        echo Checking for test failures
        if grep -q ",failed," /home/runner/_work/csv_results/cpu_tests_for_8_devices_${i}.csv; then
            echo "Test failures detected in group for_8_devices ${i}"
            touch /home/runner/_work/test_failed
        else
            echo "All tests passed / skipped in group for_8_devices ${i}"
        fi
    fi
done

# Compress the results
cd /home/runner/_work
tar -czvf results.tar.gz csv_results

# Upload to GCS, including the date and commit hash
gsutil -m cp results.tar.gz ${GCS_PREFIX}/results/archive/cpu-unit-tests-${GITHUB_HASH}-${TIMESTAMP}.tar.gz
gsutil -m cp /home/runner/_work/csv_results/cpu_tests_all_results.csv ${GCS_PREFIX}/results/unit-tests-cpu-${GITHUB_HASH}-${TIMESTAMP}.csv

exit