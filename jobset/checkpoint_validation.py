
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
import re

# 1. Constants  Axlearn-arc project
PROJECT_ID = "tpu-prod-env-one-vm"
LOCATION = "us-central1"
CLUSTER_NAME = "axlearn-arc-cluster"
NAMESPACE = "axlearn-arc"
POD_PATTERN = "arc-pw-training-pwhd-0-.*"
CONTAINER_NAME = "arc-pw-training-hd"
LOG_PATTERN = r"^Serialization.*?step_(?P<step>\d+).*"

LOG_CONFIG = {
    "project_id": PROJECT_ID,
    "location": LOCATION,
    "cluster_name": CLUSTER_NAME,
    "namespace": NAMESPACE,
    "pod_pattern": POD_PATTERN,
    "container_name": CONTAINER_NAME,
    "log_pattern": LOG_PATTERN,
}

# Environment variables
GH_RUN_ID = os.environ['GH_RUN_ID']
START_TIME = os.environ['START_TIME'] # Start time of the launch job task
END_TIME = os.environ['END_TIME'] # End time of the launch job task
GCS_PREFIX = os.environ['GCS_PREFIX']
GIT_BRANCH = os.environ['CUSTOM_GIT_BRANCH'] if "CUSTOM_GIT_BRANCH" in os.environ else "main"
GH_RUN_ID = os.environ['GH_RUN_ID']

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

def validate_checkpoint_at_steps_are_saved(
    project_id: str,
    location: str,
    cluster_name: str,
    steps_to_validate: list[int],
    ram_disk: str = "/local",
    pod_pattern: Optional[str] = ".*",
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
) -> None:
  """
  Validates that a workload is training correctly by checking for specific log
  steps.

  This function queries logs from a specified GKE cluster and namespace.
  It searches for a log entry containing the string '(blocking + background)'
  and then compares the number of steps found against an expected list of
  steps.

  A mismatch in the number of steps will cause the validation to fail. This can
  happen if, for example, a restore operation causes the step count to restart
  from zero, leading to `len(steps_to_validate) != len(found_steps)`.

  Args:
    project_id: The Google Cloud project ID
    location: GKE cluster location
    cluster_name: GKE cluster name
    start_time: Optional start time for log retrieval
      (defaults to 12 hours ago)
    end_time: Optional end time for log retrieval (defaults to now)
    steps_to_validate: Optional to validate list of steps
  Returns:
    None: This function does not return a value.
  """

  directory_pattern = (
      rf"{re.escape(ram_disk)}/(\d+)"
      if ram_disk != "gcs"
      else r"gs://[^/]+/[^/]+/[^/]+/checkpoints/(\d+)"
  )
  log_pattern = (
      rf"Finished async_save \(blocking \+ background\)\. "
      rf"Time taken: \d+\.\d+s\. directory={directory_pattern}"
  )

  complied_pattern = re.compile(log_pattern)
  entries = list_log_entries(
      project_id=project_id,
      location=location,
      cluster_name=cluster_name,
      pod_pattern=pod_pattern,
      text_filter=f'textPayload=~"{log_pattern}"',
      start_time=start_time,
      end_time=end_time,
  )

  steps_are_saved: set[int] = set()  # Use a set for faster lookup.
  for entry in entries:
    if not isinstance(entry, logging_api.TextEntry):
      raise Exception(
          "Log entry must be contain a textPayload attribute."
      )

    message = entry.payload
    m = complied_pattern.search(message)
    if m:
      steps_are_saved.add(int(m.group(1)))

  for step in steps_to_validate:
    if step not in steps_are_saved:
      logging.info(f"Found entries: {entries}")
      raise Exception(
          f"Failed to validate. Expect steps are saved: {steps_to_validate}; "
          f"got: {steps_are_saved}"
      )

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

  # We validate first 100 steps were stored
  steps_to_validate = [100]

  # Gcs bucket to later compare if indeed is being store.
  trainer_dir = f"{GCS_PREFIX}/runs/{GIT_BRANCH}/{GH_RUN_ID}"

  # This log pattern will be looged in a random pod of the first slice.
  log_pattern = LOG_CONFIG["log_pattern"]
  complied_pattern = re.compile(log_pattern)

  # Fetch logs from Cloud Logging API for the specified time window.
  entries = list_log_entries(
        **LOG_CONFIG,
        namespace="axlearn-arc",
        text_filter=f'jsonPayload.message=~"{log_pattern}"',
        start_time=start_dt,
        end_time=end_dt,
    )

  # Compare if expected steps with found steps in logs.
  steps_are_saved: set[int] = set()
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
      steps_are_saved.add(int(m.group(1)))

  for step in steps_to_validate:
    if step not in steps_are_saved:
      print(f"Found entries: {entries}")
      raise ValueError(
          f"Failed to validate. Expect steps are saved: {steps_to_validate}; "
          f"got: {steps_are_saved}"
      )
    print(
      f"Successful Validation.\nExpected  Steps:{steps_to_validate}"
      f"\tFound Steps:{steps_are_saved}"
    )

