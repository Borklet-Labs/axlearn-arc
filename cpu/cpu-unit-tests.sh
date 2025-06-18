#!/bin/bash

# Check for custom origin
if [ -z "$CUSTOM_GIT_ORIGIN" ]; then
    GIT_ORIGIN="https://gitub.com/andersensam/axlearn"
else
    GIT_ORIGIN="$CUSTOM_GIT_ORIGIN"
fi

# Check for branch name
if [ -z "$CUSTOM_GIT_BRANCH" ]; then
    GIT_BRANCH="main"
else
    GIT_BRANCH="$CUSTOM_GIT_BRANCH"
fi

echo "About to pull branch $GIT_BRANCH from origin $GIT_ORIGIN"

# Grab the latest AXLearn from upstream
git init /root && cd /root 
git remote add origin $GIT_ORIGIN
git -c protocol.version=2 fetch --no-tags --prune --no-recurse-submodules --depth=1 origin
git checkout $GIT_BRANCH

# Show the commit information
git log -1 --stat

# Install the GCP utils and PyTest suite
uv pip install .[core,gpu,gcp,tpu] pytest pytest-instafail allure-pytest torch pytest-xdist

# Clean any previous test results
rm -rf /home/runner/_work/xml_results
rm -rf /home/runner/_work/allure_results
rm pytest_xml.tar.gz allure_results.tar.gz

# Create a test output directory
mkdir -p /home/runner/_work/xml_results
mkdir -p /home/runner/_work/allure_results

pytest -v --junit-xml=/home/runner/_work/xml_results/cpu_tests.xml --alluredir /home/runner/_work/allure_results -n 8 $(find axlearn -type f -name "*test*.py" ! -name "*gpu*" ! -name "*vertex*" ! -name "*tpu*" -printf '%p ') || touch /home/runner/_work/test_failed

# Compress the results
cd /home/runner/_work
tar -czvf results.tar.gz xml_results allure_results

# Upload to GCS, including the date and hostname inside the pod
gsutil -m cp results.tar.gz ${GCS_PREFIX}/testing/cpu-unit-tests-$(date +"%Y-%m-%d-%T")-${HOSTNAME}.tar.gz

[[ ! -f /home/runner/_work/test_failed ]] && echo "All tests passed successfully"