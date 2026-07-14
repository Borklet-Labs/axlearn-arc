
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
 # @version: 2026-01-30
 #

import os
import logging
import re
from datetime import datetime, timedelta, timezone
import kubernetes
import sys
import time

from typing import Optional
from typing import List
from google.cloud import storage
from google.cloud import logging as logging_api


# Constants  Axlearn-arc project
PROJECT_ID = "tpu-prod-env-one-vm"
LOCATION = "us-central1"
CLUSTER_NAME = "axlearn-arc-cluster"
NAMESPACE = "axlearn-arc"
LOG_CONFIG = {
    "project_id": PROJECT_ID,
    "location": LOCATION,
    "cluster_name": CLUSTER_NAME,
    "namespace": NAMESPACE,
}

# Common Environment variables
JOBSET_NAME = os.environ['ARC_JOBSET_NAME'] if "ARC_JOBSET_NAME" in os.environ else None
JOBSET_NAME_TARGET = os.environ['ARC_JOBSET_NAME_TARGET'] if "ARC_JOBSET_NAME_TARGET" in os.environ else JOBSET_NAME
JOBSET_HEALTHY_TIMEOUT = int(os.environ['JOBSET_HEALTHY_TIMEOUT']) if "JOBSET_HEALTHY_TIMEOUT" in os.environ else 15 * 60
GH_RUN_ID = os.environ['GH_RUN_ID'] if "GH_RUN_ID" in os.environ else None
START_TIME = os.environ['START_TIME'] if "START_TIME" in os.environ else None
END_TIME = os.environ['END_TIME'] if "END_TIME" in os.environ else None
GCS_PREFIX = os.environ['GCS_PREFIX'] if "GCS_PREFIX" in os.environ else None
GIT_BRANCH = os.environ['CUSTOM_GIT_BRANCH'] if "CUSTOM_GIT_BRANCH" in os.environ else "main"
VALIDATION_METHOD = os.environ['VALIDATION_METHOD'] if "VALIDATION_METHOD" in os.environ else None
DELETE_MODE= os.environ['DELETE_MODE'] if "DELETE_MODE" in os.environ else None
MIN_SLICES = os.environ['MIN_SLICES'] if "MIN_SLICES" in os.environ else None



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


def delete_node_pw(dry_run: bool = False):
    """ Delete Pathways worker node
    Args:
        dry_run: If True, only print the node that would be deleted.
    Returns:
        True if completed
        False if not completed"""

    # This way we only query the pods related with <JOBSET_NAME>
    label_selector = f"jobset.sigs.k8s.io/jobset-name={JOBSET_NAME_TARGET}"
    try:
      pods_pw = KUBE_API.list_namespaced_pod(
          namespace="axlearn-arc",
          label_selector=label_selector
      )
      for pod in pods_pw.items:
          if "-pwwk-0" in pod.metadata.name and pod.spec.node_name is not None:
            node_name = pod.spec.node_name
            prefix = "[DRY RUN] Would delete node" if dry_run else "Found and deleting"
            print(f"{prefix}: {node_name}", file=sys.stderr)
            if not dry_run:
                KUBE_API.delete_node(
                    name=node_name,
                    grace_period_seconds=0
                )
                print(f"Deleting succeeded for node : {node_name}", file=sys.stderr)
            sys.exit(0)
    except ValueError as e:
        print(f"Exception when calling CoreV1Api->delete_node_name: {e}", file=sys.stderr)
    print(f"No worker found for JobSet: {JOBSET_NAME_TARGET}", file=sys.stderr)
    sys.exit(1)


def drain_nodes_pw(dry_run: bool = False):
    """ Drain Pathways worker nodes by cordoning and evicting pods.
    Args:
        dry_run: If True, only print the operations that would be performed.
    This function terminates the execution by calling sys.exit upon completion."""

    # This way we only query the pods related with <JOBSET_NAME>
    label_selector = f"jobset.sigs.k8s.io/jobset-name={JOBSET_NAME_TARGET}"
    try:
      # Step 1: Identify the nodes for slice -0
      pods_pw = KUBE_API.list_namespaced_pod(
          namespace=NAMESPACE,
          label_selector=label_selector
      )
      target_nodes = set()
      for pod in pods_pw.items:
          if "-pwwk-0" in pod.metadata.name and pod.spec.node_name is not None:
            target_nodes.add(pod.spec.node_name)

      if not target_nodes:
          print(f"No workers found for JobSet: {JOBSET_NAME_TARGET} (slice -0)", file=sys.stderr)
          sys.exit(1)

      print(f"Found target nodes to drain: {target_nodes}", file=sys.stderr)

      # Step 2: Process each node
      for node_name in target_nodes:
          prefix = "[DRY RUN] Would cordon node" if dry_run else "Cordoning node"
          print(f"{prefix}: {node_name}", file=sys.stderr)
          if not dry_run:
              try:
                  # Patching spec.unschedulable to True (Cordon)
                  KUBE_API.patch_node(
                      name=node_name,
                      body={"spec": {"unschedulable": True}}
                  )
                  print(f"Cordon succeeded for node: {node_name}", file=sys.stderr)
              except Exception as e:
                  print(f"Failed to cordon {node_name}: {e}", file=sys.stderr)
                  continue

          # Step 3: Evict pods
          try:
              # List all pods on the node regardless of dry-run status for verbose confirmation
              node_pods = KUBE_API.list_pod_for_all_namespaces(field_selector=f"spec.nodeName={node_name}")
              for pod in node_pods.items:
                  # Skip terminal pods
                  if pod.status.phase in ["Succeeded", "Failed"]:
                      continue

                  # Check for DaemonSet owners
                  is_daemonset = False
                  if pod.metadata.owner_references:
                      for owner in pod.metadata.owner_references:
                          if owner.kind == "DaemonSet":
                              is_daemonset = True
                              break

                  is_mirror = "kubernetes.io/config.mirror" in (pod.metadata.annotations or {})

                  if is_daemonset or is_mirror:
                      continue

                  action_prefix = "[DRY RUN] Would evict" if dry_run else "Evicting"
                  print(f"{action_prefix} pod: {pod.metadata.namespace}/{pod.metadata.name}", file=sys.stderr)

                  if not dry_run:
                      eviction = kubernetes.client.V1Eviction(
                          api_version="policy/v1",
                          kind="Eviction",
                          metadata=kubernetes.client.V1ObjectMeta(
                              name=pod.metadata.name,
                              namespace=pod.metadata.namespace
                          )
                      )
                      try:
                          KUBE_API.create_namespaced_pod_eviction(
                              name=pod.metadata.name,
                              namespace=pod.metadata.namespace,
                              body=eviction
                          )
                      except Exception as ev_e:
                          print(f"Eviction failed for {pod.metadata.name} due to {ev_e}. Falling back to forceful delete.", file=sys.stderr)
                          try:
                              # Matches author intent established in existing delete_node_pw/delete_pod_pw functions
                              KUBE_API.delete_namespaced_pod(
                                  name=pod.metadata.name,
                                  namespace=pod.metadata.namespace,
                                  grace_period_seconds=0
                              )
                          except Exception as del_e:
                              print(f"Delete failed for {pod.metadata.name}: {del_e}", file=sys.stderr)
          except Exception as e:
              print(f"Failed to list or process pods for {node_name}: {e}", file=sys.stderr)

      print(f"Drain action completed for nodes: {target_nodes}", file=sys.stderr)
      sys.exit(0)

    except Exception as e:
        print(f"Exception encountered during drain operation: {e}", file=sys.stderr)
        sys.exit(1)




def delete_pod_pw(dry_run: bool = False):
    """ Delete Pathways head pod
    Args:
        dry_run: If True, only print the pod that would be deleted.
    Returns:
        True if completed
        False if not completed"""

    # This way we only query the pods related with <JOBSET_NAME>
    label_selector = f"jobset.sigs.k8s.io/jobset-name={JOBSET_NAME_TARGET}"
    try:
      pods_pw = KUBE_API.list_namespaced_pod(
          namespace="axlearn-arc",
          label_selector=label_selector
      )
      for pod in pods_pw.items:
          if "-pwhd-0" in pod.metadata.name:
            pod_name = pod.metadata.name
            prefix = "[DRY RUN] Would delete pod" if dry_run else "Found and deleting"
            print(f"{prefix}: {pod_name}", file=sys.stderr)
            if not dry_run:
                KUBE_API.delete_namespaced_pod(
                    name=pod_name,
                    namespace="axlearn-arc",
                    grace_period_seconds=0
                )
                print(f"Deleting succeeded for: {pod_name}", file=sys.stderr)
            sys.exit(0)
    except ValueError as e:
        print(f"Exception when calling CoreV1Api->delete_namespaced_pod: {e}", file=sys.stderr)
    print(f"No head pod found for JobSet: {JOBSET_NAME_TARGET}", file=sys.stderr)
    sys.exit(1)

def list_log_entries(
    project_id: str,
    location: str,
    cluster_name: str,
    namespace: str = "default",
    pod_pattern: str = ".*",
    container_name: Optional[str] = None,
    text_filter: Optional[str] = None,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
) -> list[logging_api.LogEntry]:
  """
  List log entries for the specified Google Cloud project.
  This function connects to Google Cloud Logging,
  constructs a filter for Kubernetes container logs
  within a specific project, location, cluster, namespace,
  and pod name pattern, and retrieves log
  entries from the specified time range.
  It prints the timestamp, severity, resource information,
  and payload for each log entry found.

  Args:
    project_id: The Google Cloud project ID
    location: GKE cluster location
    cluster_name: GKE cluster name
    namespace: Kubernetes namespace (defaults to "default")
    pod_pattern: Pattern to match pod names (defaults to "*")
    container_name: Optional container name to filter logs
    text_filter: Optional comma-separated string to
      filter log entries by textPayload content
    start_time: Optional start time for log retrieval
      (defaults to 12 hours ago)
    end_time: Optional end time for log retrieval (defaults to now)
  Returns:
    bool: Number of log entries found
  """

  logging_client = logging_api.Client(project=project_id)

  # Set the time window for log retrieval:
  # default to last 12 hours if not provided
  if end_time is None:
    end_time = datetime.now(timezone.utc)
  if start_time is None:
    start_time = end_time - timedelta(hours=12)

  # Format times as RFC3339 UTC "Zulu" format required by the Logging API
  start_time_str = start_time.strftime("%Y-%m-%dT%H:%M:%SZ")
  end_time_str = end_time.strftime("%Y-%m-%dT%H:%M:%SZ")

  conditions = [
      f'resource.labels.project_id="{project_id}"',
      f'resource.labels.location="{location}"',
      f'resource.labels.cluster_name="{cluster_name}"',
      f'resource.labels.namespace_name="{namespace}"',
      f'resource.labels.pod_name=~"{pod_pattern}"',
      "severity>=DEFAULT",
      f'timestamp>="{start_time_str}"',
      f'timestamp<="{end_time_str}"',

  ]
  if container_name:
    conditions.append(f'resource.labels.container_name="{container_name}"')
  if text_filter:
    conditions.append(f"{text_filter}")

  log_filter = " AND ".join(conditions)

  print(f"Log filter constructed: {log_filter}",file=sys.stderr)
  return list(logging_client.list_entries(filter_=log_filter))


def convert_unix_timestamps(start_time:str = None, end_time:str = None)->tuple[datetime, datetime]:
  """
  Converts start_time and end_time environment variables from unix timestamps
  to UTC datetime objects.

  Returns:
      tuple: (start_dt, end_dt) as timezone-aware datetime objects.
  """
  if start_time is None or end_time is None:
    raise EnvironmentError(
        f"Missing required environment variables. "
        f"start_time: {start_time}, end_time: {end_time}. "
        "Ensure 'id: launch_step' is set in your YAML."
    )
  try:
    start_dt = datetime.fromtimestamp(int(start_time), tz=timezone.utc)
    end_dt = datetime.fromtimestamp(int(end_time), tz=timezone.utc)
  except (TypeError, ValueError) as e:
    raise ValueError(
            f"Could not parse timestamps (START='{start_time}', END='{end_time}'). "
            f"Error details: {e}"
    ) from None
  return start_dt, end_dt

def get_chkp_gcs(gcs_path: str) -> List[str]:
    """
    Lists files in a GCS bucket at a specified path using the standard GCS Client.
    """
    client = storage.Client()

    pattern = re.compile(r"^gs://(?P<bucket>[^/]+)/(?P<prefix>.+)$")
    m = pattern.match(gcs_path)

    if not m:
        logging.error(f"Invalid GCS path format: {gcs_path}")
        return []

    bucket_name = m.group("bucket")
    prefix = m.group("prefix")

    try:
        bucket = client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=prefix, delimiter='/')

        for _ in blobs:
          pass

        valid_checkpoints = []
        for folder_path in blobs.prefixes:
          index_check = bucket.list_blobs(
              prefix=f"{folder_path}index",
              max_results=1
          )

          # If the iterator has at least one item, the index exists
          # Checkpoint was successfully commit.
          if any(index_check):
              # Extract the folder name (e.g., 'step_00000100')
              folder_name = folder_path.rstrip('/').split('/')[-1]
              valid_checkpoints.append(folder_name)

        steps_chkp = []
        for file in valid_checkpoints:
            step = int(file.split("_")[-1])
            steps_chkp.append(step)
        print(f"Querying GCS path: {gcs_path}")
        print(f"Found {len(steps_chkp)} checkpoints.")
        return steps_chkp

    except Exception as e:
        print(f"Failed to list GCS files: {e}")
        return []

def extract_log_metrics(log_pattern:str, pod_pattern:str="arc-pw-training-pwhd-0-.*",
                        start_time:datetime = None, end_time:datetime = None,
                        is_benchmark:bool = False)-> set[int]:
  """
  Retrieves checkpoint step numbers from Cloud Logging based on a specific pattern.
  Args:
      log_pattern: Regex pattern to search for in logs.
      start_time: Start of the time window.
      end_time: End of the time window.

  Returns:
      set[int]: A set of step numbers found in the logs.
  """

  complied_pattern = re.compile(log_pattern)
  entries = list_log_entries(
        **LOG_CONFIG,
        text_filter=f'jsonPayload.message=~"{log_pattern}"',
        pod_pattern=pod_pattern,
        start_time=start_time,
        end_time=end_time,
    )

  # Get checkpoint steps given the log pattern
  chkp_in_lgs: set[int] = set()

  # Get times. (For benchmarking colocated python )
  chkp_in_lgs: set[float] = set()

  # Get times or steps depending of flag benchmark
  for entry in entries:
    if not isinstance(entry, logging_api.StructEntry):
      raise ValueError(
          "Log entry must be contain a jsonPayload attribute."
      )
    message = entry.payload.get("message")
    if not message:
      raise ValueError(f"Failed to parse entry {entry}")

    m = complied_pattern.search(message)
    if m:
      if is_benchmark:
        chkp_in_lgs.add(float(m.group(1)))
      else:
        chkp_in_lgs.add(int(m.group(1)))

  return chkp_in_lgs

def get_logs_benchmark(start_time:str = None, end_time:str = None):
  time_pattern = r"Deserialize took (?P<duration>\d+\.?\d*) seconds"
  pod_pttr_colocated = "arc-pw-colocated-pwhd-0-0-.*"
  duartion = extract_log_metrics(log_pattern=time_pattern,pod_pattern=pod_pttr_colocated,
                                start_time=start_time, end_time=end_time, is_benchmark=True)
  print(f"Times from logs --> {duartion}", file=sys.stderr)
  if len(duartion) > 0:
    return True
  raise ValueError(
        f"Benchmark validation failed: No 'Deserialize took' entries found in logs "
        f"between {start_time} and {end_time}."
    )

def validate_restore(target_step: int, start_time: Optional[datetime] = None, end_time: Optional[datetime] = None) -> bool:
    """
    Validates that the training has restored from a specific target step.
    Args:
        target_step: The step number to verify restoration from.
        start_time: Start of the time window.
        end_time: End of the time window.

    Returns:
        bool: True if validation succeeds.
    """

    # We need to pass a <step> group so it can return the chckp step found.
    pod_name_pattern =  ".*-elastic-training-pwhd-0-.*" if "elastic" in JOBSET_NAME_TARGET else "arc-pw-training-pwhd-0-.*"
    restore_pattern = r"Restoring.*step_(?P<step>\d{8})"
    chkp_in_logs = extract_log_metrics(log_pattern=restore_pattern, pod_pattern=pod_name_pattern, start_time=start_time, end_time=end_time)
    print(f"Checkpoints found on Restore: {list(chkp_in_logs)}")
    if target_step in chkp_in_logs:
      print(f"Successfully validated restoration from steps: {list(chkp_in_logs)}")
      return True
    return False


if __name__ == '__main__':

  # TODO: Need to pass this variable from the yaml file.
  # This is the target step we are aiming for deleting the pod and
  # validating the restored step checkpoint.

  print(f"DELETE_MODE: {DELETE_MODE}", file=sys.stderr)
  print(f"VALIDATION METHOD: {VALIDATION_METHOD}", file=sys.stderr)
  print(f"MIN_SLICES: {MIN_SLICES}", file=sys.stderr)

  deletion_target_step = 25
  restore_step_target= 20
  if JOBSET_NAME_TARGET == "arc-pw-ss-elastic-training":
    deletion_target_step = 105
    restore_step_target= 100

  # If DELETE_MODE: pod
  # - Trigger the pod deletion of the pathways head pod. This will make the jobset
  # restarts again.
  # If DELETE_MODE: node
  # - Trigger the node deletion of the pathways worker node. (For Pathways)
  # If DELETE_MODE: drain
  # - Trigger the node drain of the pathways worker node. (For Pathways)

  if DELETE_MODE:
    time_elapsed = 0
    while time_elapsed < JOBSET_HEALTHY_TIMEOUT:
      now = str(int(datetime.now(timezone.utc).timestamp()))
      start_dt, end_dt = convert_unix_timestamps(start_time=START_TIME, end_time=now)
      log_pattern = rf"gpt_trainer\s+process\s+\d+\s+step\s+{deletion_target_step}\s*\]"
      complied_pattern = re.compile(log_pattern)
      # Fetch logs from Cloud Logging API for the specified time window.
      entries = list_log_entries(
            **LOG_CONFIG,
            text_filter=f'jsonPayload.message=~"{log_pattern}"',
            start_time=start_dt,
            end_time=end_dt,
        )
      print(f"Entries ==>  {entries}",file=sys.stderr)
      if entries:
        if DELETE_MODE == "pod":
          print(f"Target step {deletion_target_step} reached. Triggering pod deletion...", file=sys.stderr)
          delete_pod_pw(dry_run=False)
          sys.exit(0)
        elif DELETE_MODE == "node":
          print(f"Target step {deletion_target_step} reached. Triggering node deletion...", file=sys.stderr)
          delete_node_pw(dry_run=False)
          sys.exit(0)
        elif DELETE_MODE == "drain":
          print(f"Target step {deletion_target_step} reached. Triggering node drain...", file=sys.stderr)
          drain_nodes_pw(dry_run=False)
          sys.exit(0)
      print(f"[{time_elapsed}/{JOBSET_HEALTHY_TIMEOUT}]: Waiting for target step {deletion_target_step} in logs...", file=sys.stderr)
      time.sleep(30)
      time_elapsed += 30
    raise ValueError(f"Timeout reached: Target step {deletion_target_step} not found in logs "
              f"within {JOBSET_HEALTHY_TIMEOUT} seconds.")

  # Validation method for "default" and "colocated python" benchmarking.
  # Using axlearn/cloud/gcp/examples/colocated_python_benchmark.py script.
  if VALIDATION_METHOD == "benchmark":
    print("START_TIME: {START_TIME}, END_TIME: {END_TIME}", file=sys.stderr)
    start_dt, end_dt = convert_unix_timestamps(start_time=START_TIME, end_time=END_TIME)
    get_logs_benchmark(start_dt, end_dt)

  # Validation for save cycle from checkpoint. Validated <step> is save in gcs bucket.
  if VALIDATION_METHOD == "save":
    start_dt, end_dt = convert_unix_timestamps(start_time=START_TIME, end_time=END_TIME)

    # We need to pass a <step> group so it can return the chckp step found.
    log_pattern = r"^Serialization.*?step_(?P<step>\d+).*"
    chkp_in_logs = extract_log_metrics(log_pattern=log_pattern, start_time=start_dt, end_time=end_dt)

    # Get saved checkpoints in GCS Bucket.
    trainer_dir = f"{GCS_PREFIX}/runs/{GIT_BRANCH}/{GH_RUN_ID}/checkpoints/"
    chkp_in_gcs = get_chkp_gcs(gcs_path=trainer_dir)

    # Checkpoints must match in logging and in gcs bucket.
    if len(chkp_in_gcs) != len(chkp_in_logs):
      raise ValueError(
          f"Failed to validate checkpoints of the run. Checkp in Logs: {chkp_in_logs} "
          f"Checkp in GCS Bucket: {chkp_in_gcs}"
      )
    print(f"Successfully validated {len(chkp_in_gcs)} checkpoints.")
    print(f"Checkpoints found: {sorted(list(chkp_in_logs))}")

  # Validation for restory from checkpoint cycle after pod interruption
  if VALIDATION_METHOD == "restore":
    time.sleep(90) # ~2 minutes of buffer time so the jobset can restore.
    time_elapsed = 0
    while time_elapsed < JOBSET_HEALTHY_TIMEOUT:

      # Get current time every 60 seconds.
      now = str(int(datetime.now(timezone.utc).timestamp()))
      start_dt, end_dt = convert_unix_timestamps(start_time=START_TIME, end_time=now)

      # Query 'ONLY' axlearn-arc namespace pods.
      label_selector = f"jobset.sigs.k8s.io/jobset-name={JOBSET_NAME_TARGET}"
      pods_pw = KUBE_API.list_namespaced_pod(
          namespace="axlearn-arc",
          label_selector=label_selector
      )

      # We need to check that all pods 'axlearn-arc' namespace  are in RUNNING state.
      # For Elastic Training Replica Resize we only need to wait for MIN_SLICES to be up.
      pods_targets = [pod for pod in pods_pw.items if "pwwk-1" in pod.metadata.name] if MIN_SLICES else pods_pw.items
      all_running = bool(pods_targets) and all(pod.status.phase == "Running" for pod in pods_targets)

      if all_running and len(pods_targets) > 0:
          print(f"All pods for {JOBSET_NAME_TARGET} are Running. Proceeding with restore validation.", file=sys.stderr)

          # We need to give time so the restoring logs appear in the pw-head
          # and recalculate NOW. (Extra ~2 min)
          print(f"Waiting 100 seconds to give time to restore appear logs", file=sys.stderr)
          time.sleep(120)
          now = str(int(datetime.now(timezone.utc).timestamp()))
          start_dt, end_dt = convert_unix_timestamps(start_time=START_TIME, end_time=now)

          # Validate that the training has restored from the target step.
          if validate_restore(target_step=restore_step_target, start_time=start_dt, end_time=end_dt):
             sys.exit(0)
          raise ValueError(
              f"Restore validation failed: Could not find log entry indicating "
              f"restoration from any checkpoint in the time window"
              f"for the restore step {restore_step_target}."
          )
      print(f"[{time_elapsed}/{JOBSET_HEALTHY_TIMEOUT}]: Waiting for all pods to be Running...", file=sys.stderr)
      time.sleep(60)
      time_elapsed += 60
