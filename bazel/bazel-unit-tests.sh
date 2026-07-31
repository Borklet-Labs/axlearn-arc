#!/bin/bash

cd /root/axlearn

# Get the timestamp of when the tests started
TIMESTAMP=$(date +"%Y-%m-%d-%T")
GITHUB_HASH=$(git log -1 --stat --pretty=format:"%h" --no-patch)
JAX_VER=$(python3 -c 'import jax; print(jax.__version__)' 2>/dev/null || grep "jax==" requirements.in | cut -d'=' -f3)

# Extract the GCS bucket name / path, without gs://
GCS_BUCKET_PATH=$(echo $GCS_PREFIX | cut -c 6-)

# Check for branch name
if [ -z "$CUSTOM_GIT_BRANCH" ]; then
    GIT_BRANCH="main"
else
    GIT_BRANCH="$CUSTOM_GIT_BRANCH"
fi

# Process using the bazel cache
if [ "$USE_BAZEL_CACHE" == "true" ]; then
    echo "Enabling usage of the bazel cache @ GCS_PREFIX"
    BAZEL_CACHE_CMD="--remote_cache=https://storage.googleapis.com/$GCS_BUCKET_PATH/bazel/$GIT_BRANCH --google_default_credentials=true"
else
    echo "Not using any bazel cache"
    BAZEL_CACHE_CMD=""
fi

# Check if we really want to update the bazel cache
if [ "$UPDATE_BAZEL_CACHE" != "true" ]; then
    echo "Using readonly bazel cache (not updating)"
    READONLY_CACHE="--remote_upload_local_results=false"
else
    echo "Updating the bazel cache with new results!"
    READONLY_CACHE=""
fi

bazel test $BAZEL_CACHE_CMD $READONLY_CACHE --jobs=50 --test_timeout=1800 --test_output=errors -- //...

if [ $? -ne 0 ]; then
    echo "Detected test failures / timeouts"
    touch /home/runner/_work/test_failed
fi

# Generate the master CSV report from Bazel testlogs
python3 /var/arc/parse_bazel_xml_results.py /home/runner/_work/csv_results/bazel_tests_all_results.csv

# Compress the results
cd /home/runner/_work
tar -czvf results.tar.gz csv_results

# Upload to GCS, including the date and commit hash
gsutil -m cp results.tar.gz ${GCS_PREFIX}/results/archive/bazel-unit-tests-${GITHUB_HASH}-${JAX_VER}-${GH_RUN_ID}-${TIMESTAMP}.tar.gz
gsutil -h "x-goog-meta-test-type:unit-tests" -h "x-goog-meta-processor:bazel" \
   -h "x-goog-meta-commit-hash:${GITHUB_HASH}" -h "x-goog-meta-jax-version:${JAX_VER}" \
   -h "x-goog-meta-github-run-id:${GH_RUN_ID}" -h "x-goog-meta-run-timestamp:${TIMESTAMP}" \
   -m cp /home/runner/_work/csv_results/bazel_tests_all_results.csv ${GCS_PREFIX}/results/unit-tests-bazel-${GITHUB_HASH}-${JAX_VER}-${GH_RUN_ID}-${TIMESTAMP}.csv

exit
