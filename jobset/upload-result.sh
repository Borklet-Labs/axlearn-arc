#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- All your existing git commands remain the same ---
# Check for custom origin
if [ -z "$CUSTOM_GIT_ORIGIN" ]; then
    GIT_ORIGIN="https://github.com/Borklet-Labs/axlearn"
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
git init /root && cd /root 
git remote add origin $GIT_ORIGIN
git -c protocol.version=2 fetch --no-tags --prune --no-recurse-submodules --depth=1 origin
git checkout $GIT_BRANCH
git log -1 --stat --pretty=format:"%H" --no-patch
# --- End of git commands ---


# Get other metadata
TIMESTAMP=$(date +"%Y-%m-%d-%T")
GITHUB_HASH=$(git log -1 --stat --pretty=format:"%h" --no-patch)

JAX_VER=$(python3 -c 'import jax; print(jax.__version__)' 2>/dev/null || grep "jax==" requirements.in | cut -d'=' -f3)


# Check for test type
if [ -z "$TEST_TYPE" ]; then
    TEST_TYPE="training-test"
fi

# Add optional metadata headers if they are defined
EXTRA_HEADERS=()
if [ -n "${PW_PROXY_IMAGE}" ]; then
    EXTRA_HEADERS+=("-h" "x-goog-meta-pw-proxy-image:${PW_PROXY_IMAGE}")
fi
if [ -n "${PW_SERVER_IMAGE}" ]; then
    EXTRA_HEADERS+=("-h" "x-goog-meta-pw-server-image:${PW_SERVER_IMAGE}")
fi

# Upload the result CSV to GCS with the correct metadata
gsutil "${EXTRA_HEADERS[@]}" -h "x-goog-meta-test-type:${TEST_TYPE}" -h "x-goog-meta-commit-hash:${GITHUB_HASH}" \
   -h "x-goog-meta-jax-version:${JAX_VER}" -h "x-goog-meta-github-run-id:${GH_RUN_ID}" \
   -h "x-goog-meta-run-timestamp:${TIMESTAMP}" -h "x-goog-meta-accelerator:${ACCELERATOR}" \
   -m cp /var/arc/result.csv ${GCS_PREFIX}/results/${TEST_TYPE}-${ACCELERATOR}-${GITHUB_HASH}-${JAX_VER}-${GH_RUN_ID}-${TIMESTAMP}.csv
