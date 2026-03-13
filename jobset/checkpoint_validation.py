
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
from google.cloud import logging as logging_api
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional
from typing import List
from google.cloud import storage
import re


# 1. Constants  Axlearn-arc project
PROJECT_ID = "tpu-prod-env-one-vm"
LOCATION = "us-central1"
CLUSTER_NAME = "axlearn-arc-cluster"
NAMESPACE = "axlearn-arc"
POD_PATTERN = "arc-pw-training-pwhd-0-.*"
CONTAINER_NAME = "arc-pw-training-hd"

LOG_CONFIG = {
    "project_id": PROJECT_ID,
    "location": LOCATION,
    "cluster_name": CLUSTER_NAME,
    "namespace": NAMESPACE,
    "pod_pattern": POD_PATTERN,
    "container_name": CONTAINER_NAME,
}

# Environment variables
GH_RUN_ID = os.environ['GH_RUN_ID']
START_TIME = os.environ['START_TIME'] # Start time of the launch job task
END_TIME = os.environ['END_TIME'] # End time of the launch job task
GCS_PREFIX = os.environ['GCS_PREFIX']
GIT_BRANCH = os.environ['CUSTOM_GIT_BRANCH'] if "CUSTOM_GIT_BRANCH" in os.environ else "main"
GH_RUN_ID = os.environ['GH_RUN_ID']

def get_files_gcs(gcs_path: str) -> List[str]:
    """
    Lists files in a GCS bucket at a specified path using the standard GCS Client.
    """
    # 1. Initialize the GCS Client
    # It will automatically use your environment credentials
    client = storage.Client()

    # 2. Parse the GCS path
    pattern = re.compile(r"^gs://(?P<bucket>[^/]+)/(?P<prefix>.+)$")
    m = pattern.match(gcs_path)

    if not m:
        logging.error(f"Invalid GCS path format: {gcs_path}")
        return []

    bucket_name = m.group("bucket")
    prefix = m.group("prefix")

    # 3. Query the bucket
    try:
        bucket = client.bucket(bucket_name)
        # list_blobs returns an iterator of blob objects
        blobs = bucket.list_blobs(prefix=prefix, delimiter='/')

        for _ in blobs:
          pass

        valid_checkpoints = []
        for folder_path in blobs.prefixes:
          # 2. Check for the existence of the 'index' file inside this folder
          # We limit max_results=1 because we only need to know if it exists
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

  print(f"Log filter constructed: {log_filter}")
  return list(logging_client.list_entries(filter_=log_filter))


if __name__ == '__main__':

  # Ensure START_TIME and END_TIME are available
  if START_TIME is None or END_TIME is None:
    raise EnvironmentError(
        f"Missing required environment variables. "
        f"START_TIME: {START_TIME}, END_TIME: {END_TIME}. "
        "Ensure 'id: launch_step' is set in your YAML."
    )
  try:
    start_dt = datetime.fromtimestamp(int(START_TIME), tz=timezone.utc)
    end_dt = datetime.fromtimestamp(int(END_TIME), tz=timezone.utc)
  except (TypeError, ValueError) as e:
    raise ValueError(
            f"Could not parse timestamps (START='{START_TIME}', END='{END_TIME}'). "
            f"Error details: {e}"
    ) from None

  # Gcs bucket to later compare if indeed is being store.
  trainer_dir = f"{GCS_PREFIX}/runs/{GIT_BRANCH}/{GH_RUN_ID}/checkpoints/"

  # This log pattern will be looged in a random pod of the first slice.
  log_pattern = r"^Serialization.*?step_(?P<step>\d+).*"
  complied_pattern = re.compile(log_pattern)

  # Fetch logs from Cloud Logging API for the specified time window.
  entries = list_log_entries(
        **LOG_CONFIG,
        text_filter=f'jsonPayload.message=~"{log_pattern}"',
        start_time=start_dt,
        end_time=end_dt,
    )

  # Get saved steps in the logs.
  chkp_in_lgs: set[int] = set()
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
      chkp_in_lgs.add(int(m.group(1)))

  # Get saved steps in GCS Bucket.
  chkp_in_gcs = get_files_gcs(trainer_dir)

  # Failed if does not match
  if len(chkp_in_gcs) != len(chkp_in_lgs):
    raise ValueError(
        f"Failed to validate checkpoints of the run. Checkp in Logs: {chkp_in_lgs} "
        f"Checkp in GCS Bucket: {chkp_in_gcs}"
    )
  print(f"Successfully validated {len(chkp_in_gcs)} checkpoints.")
  print(f"Checkpoints found: {sorted(list(chkp_in_lgs))}")

