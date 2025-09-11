#!/bin/bash

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

# Grab the latest AXLearn from upstream
git init /root && cd /root 
git remote add origin $GIT_ORIGIN
git -c protocol.version=2 fetch --no-tags --prune --no-recurse-submodules --depth=1 origin
git checkout $GIT_BRANCH

# Show the commit information
git log -1 --stat --pretty=format:"%H" --no-patch

# Get the timestamp of when the tests started
TIMESTAMP=$(date +"%Y-%m-%d-%T")
GITHUB_HASH=$(git log -1 --stat --pretty=format:"%h" --no-patch)
JAX_VER=$(cat pyproject.toml | grep jax== | sed 's/\s*"jax==//g' | sed 's/",\s*//g')

# Upload the result CSV to GCS
gsutil -h "x-goog-meta-github-hash:${GITHUB_HASH}" -h "x-goog-meta-jax-version:${JAX_VER}" \
   -h "x-goog-meta-github-run-id:${GH_RUN_ID}" -h "x-goog-meta-timestamp:${TIMESTAMP}" \
   -h "x-goog-meta-accelerator: ${ACCELERATOR}" \
   -m cp /var/arc/result.csv ${GCS_PREFIX}/results/training-test-${ACCELERATOR}-${GITHUB_HASH}-${JAX_VER}-${GH_RUN_ID}-${TIMESTAMP}.csv