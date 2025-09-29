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

echo "Checking for JAX version..."
GCS_VERSION_TAG_FILE="${GCS_PREFIX}/metadata/jax_version_tag_${GH_RUN_ID}"

# 1. Primary Method: Check if JAX_VER was passed in from the previous step.
if [ -n "${JAX_VER}" ]; then
  echo "Using JAX version passed from previous step: ${JAX_VER}"
else
  # 2. Fallback Method: If not, attempt to fetch it directly from GCS.
  echo "JAX_VER was not provided. Attempting to fetch from GCS as a fallback..."
  
  # Corrected the grep command to look for "jax-version:"
  JAX_VER_FALLBACK=$(gsutil stat "${GCS_VERSION_TAG_FILE}" 2>/dev/null | grep 'jax-version' | sed 's/.*: *//')

  if [ -n "${JAX_VER_FALLBACK}" ]; then
    echo "Successfully fetched JAX version from GCS: ${JAX_VER_FALLBACK}"
    JAX_VER="${JAX_VER_FALLBACK}"
  fi
fi

# 3. Final Check: If JAX_VER is still empty after both attempts, exit.
if [ -z "${JAX_VER}" ]; then
  echo "ERROR: Failed to obtain JAX version from both the previous step and GCS fallback."
  exit 1
fi

echo "JAX version: ${JAX_VER}"

# Upload the result CSV to GCS with the correct metadata
gsutil -h "x-goog-meta-test-type:training-test" -h "x-goog-meta-commit-hash:${GITHUB_HASH}" \
   -h "x-goog-meta-jax-version:${JAX_VER}" -h "x-goog-meta-github-run-id:${GH_RUN_ID}" \
   -h "x-goog-meta-run-timestamp:${TIMESTAMP}" -h "x-goog-meta-accelerator:${ACCELERATOR}" \
   -m cp /var/arc/result.csv ${GCS_PREFIX}/results/training-test-${ACCELERATOR}-${GITHUB_HASH}-${JAX_VER}-${GH_RUN_ID}-${TIMESTAMP}.csv

if [[ -z "$GCS_PREFIX" || "$GCS_PREFIX" != gs://* ]]; then
  echo "FATAL ERROR: The GCS_PREFIX environment variable is not set or is invalid."
  echo "It must start with 'gs://'. Current value: '$GCS_PREFIX'"
  exit 1
fi

# Clean up the temporary metadata file
gsutil rm "${GCS_VERSION_TAG_FILE}"