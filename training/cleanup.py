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
 # @version: 2025-06-18
 #

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

if __name__ == '__main__':

    cleanup_jobset_and_exit(JOBSET_NAME, 0)
