 #  ________   ___   __    ______   ______   ______    ______   ______   ___   __    ______   ________   ___ __ __     
 # /_______/\ /__/\ /__/\ /_____/\ /_____/\ /_____/\  /_____/\ /_____/\ /__/\ /__/\ /_____/\ /_______/\ /__//_//_/\    
 # \::: _  \ \\::\_\\  \ \\:::_ \ \\::::_\/_\:::_ \ \ \::::_\/_\::::_\/_\::\_\\  \ \\::::_\/_\::: _  \ \\::\| \| \ \   
 #  \::(_)  \ \\:. `-\  \ \\:\ \ \ \\:\/___/\\:(_) ) )_\:\/___/\\:\/___/\\:. `-\  \ \\:\/___/\\::(_)  \ \\:.      \ \  
 #   \:: __  \ \\:. _    \ \\:\ \ \ \\::___\/_\: __ `\ \\_::._\:\\::___\/_\:. _    \ \\_::._\:\\:: __  \ \\:.\-/\  \ \ 
 #    \:.\ \  \ \\. \`-\  \ \\:\/.:| |\:\____/\\ \ `\ \ \ /____\:\\:\____/\\. \`-\  \ \ /____\:\\:.\ \  \ \\. \  \  \ \
 #     \__\/\__\/ \__\/ \__\/ \____/_/ \_____\/ \_\/ \_\/ \_____\/ \_____\/ \__\/ \__\/ \_____\/ \__\/\__\/ \__\/ \__\/    
 #                                                                                                               
 # Project: AXLearn ARC Testing: Launch a GPU or TPU training job
 # @author : Samuel Andersen
 # @version: 2025-06-23
 #

import json
import sys
import time
import os
import signal
import kubernetes

# Get the JobSet info from the environment
JOBSET_NAME = os.environ['ARC_JOBSET_NAME']
JOBSET_JSON = os.environ['ARC_JOBSET_JSON']
DOCKER_IMAGE = os.environ['TRAINING_DOCKER_IMAGE']
GCS_PREFIX = os.environ['GCS_PREFIX']

# Use the dynamic client to leverage the JobSet API
CLIENT = kubernetes.dynamic.DynamicClient(
    kubernetes.client.ApiClient(
        configuration=kubernetes.config.load_incluster_config()))
# Custom objects API
CUSTOM_OBJECT_API = kubernetes.client.CustomObjectsApi()
# Import the JobSet API into our k8s client
JOBSET_API = CLIENT.resources.get(api_version = "jobset.x-k8s.io/v1alpha2", kind = "JobSet")
# Prepare the standard API for use
kubernetes.config.load_incluster_config()
KUBE_API = kubernetes.client.CoreV1Api()

def receive_signal(signum, signal):
    """Receive a signal from Github and exit
    
    Args:
        signum: Number of signal
        signal: Signal"""

    print(f"Received signal {signum}. Exiting...", file=sys.stderr)
    cleanup_jobset_and_exit(JOBSET_NAME, -1)

signal.signal(signal.SIGTERM, receive_signal)
signal.signal(signal.SIGINT, receive_signal)

def get_current_jobset(jobset_name: str):
    """Fetch a list of active JobSets in a predefined namespace, returning the JobSet
    that matches the name provided
    
    Args:
        jobnet_name: String containing the JobSet name
        
    Returns:
        Returns the JobSet object matching the name, or None"""

    current_jobsets = CUSTOM_OBJECT_API.list_namespaced_custom_object(
        "jobset.x-k8s.io", "v1alpha2", "axlearn-arc", "jobsets")

    for jobset in current_jobsets["items"]:
        if jobset_name in jobset["metadata"]["name"]:
            return jobset

def get_jobset_status(jobset_name: str):
    """Get the status of a JobSet
    
    Args:
        jobnet_name: String containing the JobSet name
        
    Returns:
        Returns the JobSet status object matching the name, or None"""

    return get_current_jobset(jobset_name)["status"]["replicatedJobsStatus"][0]

def cleanup_jobset(jobset_name: str):
    """Delete a JobSet when finishing execution
    
    Args:
        jobset_name: String containing the JobSet to delete"""

    print(f"Deleting JobSet {jobset_name}...", file=sys.stderr)
    JOBSET_API.delete(name=jobset_name, namespace="axlearn-arc")

def cleanup_jobset_and_exit(jobset_name: str, exit_code: int):
    """Delete a JobSet and exit
    
    Args:
        jobset_name: String containing the JobSet to delete
        exit_code: Integer code to return"""
    
    cleanup_jobset(jobset_name)
    sys.exit(exit_code)

def get_pod_status(pod_name: str):
    """Get the status of a pod
    
    Args:
        pod_name: String with the name of the pod
        
    Returns:
        Returns the current status of the pod"""

    pods = KUBE_API.list_namespaced_pod(namespace="axlearn-arc", watch=False)

    for pod in pods.items:
        if pod_name in pod.metadata.name:
            return pod.status

def check_jobset_healthy(jobset_name: str, before_schedule = False) -> bool:
    """Check if a JobSet was accepted and if the pods are in a healthy state
    
    Args:
        jobset_name: String containing the JobSet name
        
    Returns:
        True if the JobSet is active and pods are running
        False if the JobSet is active but pods aren't scheduled"""

    jobset_status = get_jobset_status(jobset_name)

    if jobset_status["failed"] != 0 or jobset_status["suspended"] != 0:
        return False
    elif jobset_status["active"] != 0:
        if before_schedule:
            return True
        
        pod_status = get_pod_status(jobset_name)
        if "Running" in pod_status.phase:
            return True
    elif jobset_status["succeeded"] != 0:
        return True

    return False

def check_jobset_completed(jobset_name: str) -> bool:
    """Check to see if a JobSet completed
    
    Args:
        jobset_name: String with the JobSet name
        
    Returns:
        True if completed
        False if not completed"""

    if not check_jobset_healthy(jobset_name):
        return False

    jobset_status = get_jobset_status(jobset_name)
    if jobset_status["succeeded"] != 0:
        return True

    return False

def update_jobset(jobset_base_config: dict) -> dict:
    """Take in a JobSet config dict and update with new Git info
    and information about the owner to handle proper termination
    
    Args:
        jobset_base_config: Dict containing the JobSet config
        
    Returns:
        Returns a new dict for the JobSet"""
    
    updated_jobset = json.dumps(jobset_base_config)

    if "CUSTOM_GIT_ORIGIN" in os.environ:
        if os.environ["CUSTOM_GIT_ORIGIN"] != "INSERT_GIT_ORIGIN":
            print(f"Found custom git origin {os.environ['CUSTOM_GIT_ORIGIN']}", file=sys.stderr)
            updated_jobset = updated_jobset.replace("INSERT_GIT_ORIGIN", os.environ["CUSTOM_GIT_ORIGIN"])
        else:
            updated_jobset = updated_jobset.replace("INSERT_GIT_ORIGIN", "")
    else:
        updated_jobset = updated_jobset.replace("INSERT_GIT_ORIGIN", "")

    if "CUSTOM_GIT_BRANCH" in os.environ:
        if os.environ["CUSTOM_GIT_BRANCH"] != "INSERT_GIT_BRANCH":
            print(f"Found custom git branch {os.environ['CUSTOM_GIT_BRANCH']}", file=sys.stderr)
            updated_jobset = updated_jobset.replace("INSERT_GIT_BRANCH", os.environ["CUSTOM_GIT_BRANCH"])
        else:
            updated_jobset = updated_jobset.replace("INSERT_GIT_BRANCH", "")
    else:
        updated_jobset = updated_jobset.replace("INSERT_GIT_BRANCH", "")

    # Get the pod UID to ensure clean deletion later
    pods = KUBE_API.list_namespaced_pod(namespace="axlearn-arc", watch=False)
    pod_metadata = None

    for pod in pods.items:
        if pod.metadata.name == os.environ["HOSTNAME"]:
            pod_metadata = pod.metadata
            break

    if pod_metadata:
        updated_jobset = updated_jobset.replace("POD_NAME_HERE", os.environ["HOSTNAME"])
        updated_jobset = updated_jobset.replace("UID_HERE", pod_metadata.uid)
    else:
        print("Unable to determine pod info and cannot create JobSet safely", file=sys.stderr)
        sys.exit(-1)

    # Insert the correct Docker image into the JobSet
    updated_jobset = updated_jobset.replace("INSERT_DOCKER_IMAGE", DOCKER_IMAGE)
    # Insert the GCS prefix
    updated_jobset = updated_jobset.replace("INSERT_GCS_PREFIX", GCS_PREFIX)

    return json.loads(updated_jobset)

if __name__ == '__main__':

    # Read in the JobSet JSON
    with open(JOBSET_JSON, "r", encoding="utf-8") as js_file:
        jobset_config = json.load(js_file)

    jobset_config = update_jobset(jobset_config)
    # Create the JobSet
    print(f"Creating new JobSet {JOBSET_NAME} from template {JOBSET_JSON}", file=sys.stderr)
    JOBSET_API.create(body=jobset_config, namespace="axlearn-arc")

    # Sleep to ensure that the JobSet is created and is not suspended, usually from
    # nodepool scale up events
    time.sleep(15)

    # Create a context for pulling custom objects
    jobset_status = get_jobset_status(JOBSET_NAME)

    # Ensure the JobSet is not marked as failed
    if jobset_status["failed"] != 0 or jobset_status["suspended"] != 0:
        print(f"JobSet {JOBSET_NAME} failed.", file=sys.stderr)
        cleanup_jobset_and_exit(JOBSET_NAME, -1)

    time_elapsed = 0
    while time_elapsed < 600:
        pod_status = get_pod_status(JOBSET_NAME)
        print(f"[{time_elapsed}/600]: Waiting for pod for JobSet {JOBSET_NAME} to be scheduled: {pod_status.phase}",
              file=sys.stderr)
        # Check to see if the pod has been scheduled
        if "Running" in pod_status.phase:
            print(f"Pod scheduled successfully for {JOBSET_NAME}", file=sys.stderr)
            break
        elif "Error" in pod_status.phase or "Terminating" in pod_status.phase:
            print(f"Error detected in pod for JobSet {JOBSET_NAME}. Cleaning up.", file=sys.stderr)
            cleanup_jobset_and_exit(JOBSET_NAME, -1)
        
        if not check_jobset_healthy(JOBSET_NAME, before_schedule=True):
            print(f"Error detected in pod for JobSet {JOBSET_NAME}. Cleaning up.", file=sys.stderr)
            cleanup_jobset_and_exit(JOBSET_NAME, -1)
            
        time.sleep(15)
        time_elapsed += 15

    if not check_jobset_healthy(JOBSET_NAME):
        print(f"Error: Pod for JobSet {JOBSET_NAME} was unable to be scheduled or crashed", file=sys.stderr)
        cleanup_jobset_and_exit(JOBSET_NAME, -1)

    # Wait up to 10 minutes for a JobSet error to be reported, otherwise assume
    # execution was successful
    time_elapsed = 0
    while time_elapsed < 600:
        jobset_healthy = check_jobset_healthy(JOBSET_NAME)
        print(f"[{time_elapsed}/600]: Current JobSet status for {JOBSET_NAME}: Healthy: {jobset_healthy}",
              file=sys.stderr)
        # Check for failures
        if not jobset_healthy:
            print(f"Error detected in pod for JobSet {JOBSET_NAME}. Cleaning up.", file=sys.stderr)
            cleanup_jobset_and_exit(JOBSET_NAME, -1)
        else:
            if check_jobset_completed(JOBSET_NAME):
                print(f"JobSet {JOBSET_NAME} completed successfully.", file=sys.stderr)
                cleanup_jobset_and_exit(JOBSET_NAME, 0)
        # Sleep 30 seconds before polling the status of the JobSet again
        time.sleep(30)
        time_elapsed += 30

    if check_jobset_healthy(JOBSET_NAME):
        print(f"JobSet {JOBSET_NAME} running as expected. Reporting success.", file=sys.stderr)
        cleanup_jobset_and_exit(JOBSET_NAME, 0)

    print(f"Failure detected in JobSet {JOBSET_NAME}", file=sys.stderr)
    cleanup_jobset_and_exit(JOBSET_NAME, -1)
