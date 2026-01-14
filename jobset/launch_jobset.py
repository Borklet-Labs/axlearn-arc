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
 # @version: 2026-01-14
 #

import json
import sys
import time
import os
import signal
import threading
import kubernetes

# Get the JobSet info from the environment
JOBSET_NAME = os.environ['ARC_JOBSET_NAME']
JOBSET_JSON = os.environ['ARC_JOBSET_JSON']
DOCKER_IMAGE = os.environ['JOBSET_DOCKER_IMAGE']
GCS_PREFIX = os.environ['GCS_PREFIX']
JOBSET_HEALTHY_TIMEOUT = int(os.environ['JOBSET_HEALTHY_TIMEOUT'])
GH_RUN_ID = os.environ['GH_RUN_ID']
SCHEDULE_TIMEOUT = int(os.environ['SCHEDULE_TIMEOUT']) if "SCHEDULE_TIMEOUT" in os.environ else 15 * 60
POST_SETUP_CMD = os.environ['POST_SETUP_CMD'] if "POST_SETUP_CMD" in os.environ else None

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

def cleanup_jobset_and_exit(jobset_name: str,
                            exit_code: int,
                            log_worker: threading.Thread = None,
                            stop_log: threading.Event = None):
    """Delete a JobSet and exit
    
    Args:
        jobset_name: String containing the JobSet to delete
        exit_code: Integer code to return
        log_worker: Thread where the logger is running
        stop_log: Event to stop the log_worker before join"""
    if log_worker and stop_log:
        attempts = 1
        stop_log.set()
        thread_running = log_worker.is_alive()
        while thread_running:
            if attempts == 5:
                print("WARN: Unable to stop log thread. Proceeding with killing execution...",
                      file=sys.stderr)
                break
            print(f"Attempting to stop log thread with a 10 second timeout.. Attempt {attempts}/5",
                  file=sys.stderr)
            log_worker.join(timeout=10.0)
            thread_running = log_worker.is_alive()
            attempts += 1
        print("Log thread stopped successfully. Finishing cleanup...", file=sys.stderr)
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

def get_pod_logs(pod_name: str, stop: threading.Event):
    """Spawn a stream to print out pod logs during execution
    
    Args:
        pod_name: String with the name of the pod
        stop: Threading event to stop execution"""

    pods = KUBE_API.list_namespaced_pod(namespace="axlearn-arc", watch=False)
    target_pod = None

    for pod in pods.items:
        if pod_name in pod.metadata.name:
            target_pod = pod.metadata.name
            break
    if not target_pod:
        return "Unable to infer pod name. Returning empty result"

    try:
        stream = kubernetes.watch.Watch().stream(KUBE_API.read_namespaced_pod_log,
            namespace="axlearn-arc", name=target_pod)
        for line in stream:
            if stop.is_set():
                break
            print(line, file=sys.stderr)
    except Exception as e:
        print(f"Error in streaming logs, Exception: {e}", file=sys.stderr)
        print("Turning off log streaming... full log will be available in GCS after the run",
                file=sys.stderr)

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

    # Ensure we check for success before seeing if it is still active
    elif jobset_status["succeeded"] != 0:
        return True

    elif jobset_status["active"] != 0:
        if before_schedule:
            return True

        pod_status = get_pod_status(jobset_name)
        if "Running" in pod_status.phase:
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
    # Insert the Github ARC run id
    updated_jobset = updated_jobset.replace("INSERT_GH_RUN_ID", GH_RUN_ID)

    # Check to see if we override the maxRestarts in the JobSet
    if "JOBSET_MAX_RESTARTS" in os.environ:
        print(f'Detected maxRestarts override: {os.environ["JOBSET_MAX_RESTARTS"]}', file=sys.stderr)
        updated_jobset = updated_jobset.replace('"INSERT_MAX_RESTARTS"', str(os.environ["JOBSET_MAX_RESTARTS"]))
    else:
        updated_jobset = updated_jobset.replace('"INSERT_MAX_RESTARTS"', str(0))

    # Add any additional setup commands if defined
    if POST_SETUP_CMD:
        print(f'Detected post-setup command: {POST_SETUP_CMD}', file=sys.stderr)
        updated_jobset = updated_jobset.replace("INSERT_POST_SETUP_CMD", POST_SETUP_CMD)

    return json.loads(updated_jobset)

def write_result(success: bool):
    """Write the result of a test to a CSV
    
    Args:
        success: True if the loop ended successful"""

    result_text = "failed"
    if success:
        result_text = "passed"

    # Open the result CSV file in write mode
    with open("/var/arc/result.csv", "w", encoding="utf-8") as csv_file:
        csv_file.write("id,module,name,file,doc,markers,status,message,duration\n")
        csv_file.write(f"axlearn/experiments/text/gpt/fuji.py::RunTrainingLoop::fuji_training,axlearn.experiments.text.gpt.fuji,fuji_training,axlearn/experiments/text/gpt/fuji.py,,,{result_text},See full logs in GCS or Github,1.0\n")

def create_jobset_and_wait(jobset_config, skip_creation: bool = False):
    """Create a JobSet using the CRD API and ensure its completion happens
    
    Args:
        jobset_config: A config to submit to the kube API
        skip_creation: Don't create the JobSet, just wait for creation"""

    if not skip_creation:
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
    while time_elapsed < (SCHEDULE_TIMEOUT):
        pod_status = get_pod_status(JOBSET_NAME)
        print(f"[{time_elapsed}/{SCHEDULE_TIMEOUT}]: Waiting for pod for JobSet {JOBSET_NAME} to be scheduled: {pod_status.phase}",
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

def monitor_jobset_status():
    """Monitor the progression of a JobSet, looking at the health of the pod and counting down
    to JOBSET_HEALTHY_TIMEOUT
    
    Returns: Returns references to stop_log and log_worker"""

    # Spawn a thread to print pod logs to stderr
    stop_log = threading.Event()
    log_worker = threading.Thread(target=get_pod_logs, args=(JOBSET_NAME,stop_log))
    log_worker.start()
    # Wait up to 10 minutes for a JobSet error to be reported, otherwise assume
    # execution was successful
    time_elapsed = 0
    while time_elapsed < JOBSET_HEALTHY_TIMEOUT:
        jobset_healthy = check_jobset_healthy(JOBSET_NAME)
        print(f"[{time_elapsed}/{JOBSET_HEALTHY_TIMEOUT}]: Current JobSet status for {JOBSET_NAME}: Healthy: {jobset_healthy}",
              file=sys.stderr)
        # Check for failures
        if not jobset_healthy:
            print(f"Error detected in pod for JobSet {JOBSET_NAME}. Cleaning up.", file=sys.stderr)
            write_result(False)
            cleanup_jobset_and_exit(JOBSET_NAME, -1, log_worker, stop_log)
        else:
            if check_jobset_completed(JOBSET_NAME):
                print(f"JobSet {JOBSET_NAME} completed successfully.", file=sys.stderr)
                write_result(True)
                cleanup_jobset_and_exit(JOBSET_NAME, 0, log_worker, stop_log)
        # Sleep 30 seconds before polling the status of the JobSet again
        time.sleep(30)
        time_elapsed += 30

    return log_worker, stop_log

if __name__ == '__main__':

    # Read in the JobSet JSON
    with open(JOBSET_JSON, "r", encoding="utf-8") as js_file:
        jobset_config = json.load(js_file)
    # Update the JobSet config with any variables that need to be injected
    jobset_config = update_jobset(jobset_config)

    # Configure the log_worker and stop_log references
    log_worker: threading.Thread = None
    stop_log: threading.Event = None
    jobset_resumed = False
    # Check to see if the JobSet already exists
    print(f"Checking to see if JobSet {JOBSET_NAME} already exists", file=sys.stderr)
    if get_current_jobset(JOBSET_NAME):
        jobset_status = get_jobset_status(JOBSET_NAME)
        if jobset_status["active"] != 0:
            if "RESUME_JOBSET" in os.environ:
                print(f"JobSet {JOBSET_NAME} is still active and RESUME_JOBSET is configured... Reattaching",
                      file=sys.stderr)
                jobset_resumed = True
                create_jobset_and_wait(jobset_config, skip_creation=True)
                log_worker, stop_log = monitor_jobset_status()
            else:
                print(f"JobSet {JOBSET_NAME} still active, but RESUME_JOBSET is not configured. Killing and exiting",
                      file=sys.stderr)
                cleanup_jobset_and_exit(JOBSET_NAME, 1)
        elif jobset_status["failed"] != 0 or jobset_status["succeeded"] != 0 or jobset_status["suspended"] != 0:
            print(f"JobSet {JOBSET_NAME} is stale. Deleting and exiting", file=sys.stderr)
            cleanup_jobset_and_exit(JOBSET_NAME, 1)

    # Create the JobSet and wait for creation to complete
    if not jobset_resumed:
        create_jobset_and_wait(jobset_config)
        if not check_jobset_healthy(JOBSET_NAME):
            print(f"Error: Pod for JobSet {JOBSET_NAME} was unable to be scheduled or crashed",
                  file=sys.stderr)
            cleanup_jobset_and_exit(JOBSET_NAME, -1)

        # Monitor the JobSet status for the duration of the execution
        log_worker, stop_log = monitor_jobset_status()

    if check_jobset_healthy(JOBSET_NAME):
        print(f"JobSet {JOBSET_NAME} running as expected. Reporting success.", file=sys.stderr)
        write_result(True)
        cleanup_jobset_and_exit(JOBSET_NAME, 0, log_worker, stop_log)

    print(f"Failure detected in JobSet {JOBSET_NAME}", file=sys.stderr)
    write_result(False)
    cleanup_jobset_and_exit(JOBSET_NAME, -1, log_worker, stop_log)
