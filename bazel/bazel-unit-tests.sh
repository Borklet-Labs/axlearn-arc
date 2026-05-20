#!/bin/bash

cd /root

# Get the timestamp of when the tests started
TIMESTAMP=$(date +"%Y-%m-%d-%T")
GITHUB_HASH=$(git log -1 --stat --pretty=format:"%h" --no-patch)
JAX_VER=$(python3 -c 'import jax; print(jax.version.__version__)')

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

bazel test $BAZEL_CACHE_CMD $READONLY_CACHE --jobs=50 --test_timeout=1800 -- //...

if [ $? -ne 0 ]; then
    echo "Detected test failures / timeouts"
    touch /home/runner/_work/test_failed
fi

# Compress the results
cd /home/runner/_work
tar -czvf results.tar.gz csv_results

# Upload to GCS, including the date and commit hash
gsutil -m cp results.tar.gz ${GCS_PREFIX}/results/archive/bazel-unit-tests-${GITHUB_HASH}-${JAX_VER}-${GH_RUN_ID}-${TIMESTAMP}.tar.gz
