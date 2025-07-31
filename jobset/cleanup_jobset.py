 #  ________   ___   __    ______   ______   ______    ______   ______   ___   __    ______   ________   ___ __ __     
 # /_______/\ /__/\ /__/\ /_____/\ /_____/\ /_____/\  /_____/\ /_____/\ /__/\ /__/\ /_____/\ /_______/\ /__//_//_/\    
 # \::: _  \ \\::\_\\  \ \\:::_ \ \\::::_\/_\:::_ \ \ \::::_\/_\::::_\/_\::\_\\  \ \\::::_\/_\::: _  \ \\::\| \| \ \   
 #  \::(_)  \ \\:. `-\  \ \\:\ \ \ \\:\/___/\\:(_) ) )_\:\/___/\\:\/___/\\:. `-\  \ \\:\/___/\\::(_)  \ \\:.      \ \  
 #   \:: __  \ \\:. _    \ \\:\ \ \ \\::___\/_\: __ `\ \\_::._\:\\::___\/_\:. _    \ \\_::._\:\\:: __  \ \\:.\-/\  \ \ 
 #    \:.\ \  \ \\. \`-\  \ \\:\/.:| |\:\____/\\ \ `\ \ \ /____\:\\:\____/\\. \`-\  \ \ /____\:\\:.\ \  \ \\. \  \  \ \
 #     \__\/\__\/ \__\/ \__\/ \____/_/ \_____\/ \_\/ \_\/ \_____\/ \_____\/ \__\/ \__\/ \_____\/ \__\/\__\/ \__\/ \__\/    
 #                                                                                                               
 # Project: AXLearn ARC Testing: Cleanup a JobSet if ARC is cancelled
 # @author : Samuel Andersen
 # @version: 2025-07-30
 #

import json
import sys
import os
import kubernetes

# Get the JobSet info from the environment
JOBSET_NAME = os.environ['ARC_JOBSET_NAME']
JOBSET_JSON = os.environ['ARC_JOBSET_JSON']

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

def cleanup_jobset(jobset_name: str):
    """Delete a JobSet when finishing execution
    
    Args:
        jobset_name: String containing the JobSet to delete"""

    print(f"Deleting JobSet {jobset_name}...", file=sys.stderr)
    JOBSET_API.delete(name=jobset_name, namespace="axlearn-arc")

if __name__ == '__main__':

    # Read in the JobSet JSON
    with open(JOBSET_JSON, "r", encoding="utf-8") as js_file:
        jobset_config = json.load(js_file)
    # Clean the JobSet
    try:
        cleanup_jobset(JOBSET_NAME)
    except Exception as e:
        print(f"Exception from kube API: {e}", file=sys.stderr)
        sys.exit(1)

    sys.exit(0)
    