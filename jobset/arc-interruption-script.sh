#!/bin/bash

# --- Settings ---
WORKLOAD_ID=${ARC_JOBSET_NAME_TO_CANCEL:-$1}
PROJECT="tpu-prod-env-one-vm"
ZONE="us-central2-b"
CLUSTER="axlearn-arc-cluster"
NAMESPACE="axlearn-arc"
DRY_RUN=true
MAX_RETRIES=30

if [ -z "$WORKLOAD_ID" ]; then
    echo "ERROR: ARC_JOBSET_NAME_TO_CANCEL is not set."
    exit 1
fi

# 0. Connect to the cluster
echo "Initializing cluster connection..."
gcloud container clusters get-credentials "$CLUSTER" --zone "$ZONE" --project "$PROJECT"

echo "Searching for pods associated with workload: $WORKLOAD_ID..."

# 1. Loop until at least one pod name is found
RETRY_COUNT=0
POD_NAMES=""

while [ -z "$POD_NAMES" ]; do
    # Get all pod names associated with the jobset
    POD_NAMES=$(kubectl get pods -n "$NAMESPACE" -l "jobset.sigs.k8s.io/jobset-name=$WORKLOAD_ID" -o jsonpath='{.items[*].metadata.name}' 2>/dev/null)

    if [ -z "$POD_NAMES" ]; then
        if [ "$RETRY_COUNT" -ge "$MAX_RETRIES" ]; then
            echo "TIMEOUT: No pods found for $WORKLOAD_ID after $MAX_RETRIES attempts."
            exit 1
        fi
        echo "Waiting for pods to be created for $WORKLOAD_ID... (Attempt $RETRY_COUNT/$MAX_RETRIES)"
        sleep 10
        ((RETRY_COUNT++))
    fi
done

echo "Found pods: $POD_NAMES"

# 2. Action: Force Delete the Pods
for POD in $POD_NAMES; do
    if [ "$DRY_RUN" = true ]; then
        echo "DRY RUN: Would force delete pod: $POD in namespace: $NAMESPACE"
    else
        echo "FORCE DELETING pod: $POD..."
        # Force delete handles pods stuck in "Terminating"
        # kubectl delete pod "$POD" -n "$NAMESPACE" --force --grace-period=0
    fi
done

echo "Cleanup complete for $WORKLOAD_ID."
