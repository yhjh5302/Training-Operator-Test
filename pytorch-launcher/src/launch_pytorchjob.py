import argparse
import datetime
from distutils.util import strtobool
import logging
import yaml
import threading

from kubernetes import client as k8s_client
from kubernetes import config

import launch_crd
from kubeflow.pytorchjob import V1PyTorchJob as V1PyTorchJob_original
from kubeflow.pytorchjob import V1PyTorchJobSpec as V1PyTorchJobSpec_original


logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

logger = logging.getLogger(__name__)


def yamlOrJsonStr(string):
    if string == "" or string is None:
        return None
    return yaml.safe_load(string)


def get_current_namespace():
    """Returns current namespace if available, else kubeflow"""
    try:
        namespace = "/var/run/secrets/kubernetes.io/serviceaccount/namespace"
        current_namespace = open(namespace).read()
    except FileNotFoundError:
        current_namespace = "kubeflow"
    return current_namespace


# Patch PyTorchJob APIs to align with k8s usage
class V1PyTorchJob(V1PyTorchJob_original):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.openapi_types = self.swagger_types


class V1PyTorchJobSpec(V1PyTorchJobSpec_original):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.openapi_types = self.swagger_types


def get_arg_parser():
    parser = argparse.ArgumentParser(description="Kubeflow Job launcher")
    parser.add_argument("--name", type=str,
                        default="pytorchjob",
                        help="Job name.")
    parser.add_argument("--namespace", type=str,
                        default=get_current_namespace(),
                        help="Job namespace.")
    parser.add_argument("--version", type=str,
                        default="v1",
                        help="Job version.")
    parser.add_argument("--activeDeadlineSeconds", type=int,
                        default=None,
                        help="Specifies the duration (in seconds) since startTime during which the job can remain active before it is terminated. Must be a positive integer. This setting applies only to pods where restartPolicy is OnFailure or Always.")
    parser.add_argument("--backoffLimit", type=int,
                        default=None,
                        help="Number of retries before marking this job as failed.")
    parser.add_argument("--cleanPodPolicy", type=str,
                        default="Running",
                        help="Defines the policy for cleaning up pods after the Job completes.")
    parser.add_argument("--ttlSecondsAfterFinished", type=int,
                        default=None,
                        help="Defines the TTL for cleaning up finished Jobs.")
    parser.add_argument("--masterSpec", type=yamlOrJsonStr,
                        default={},
                        help="Job master replicaSpecs.")
    parser.add_argument("--workerSpec", type=yamlOrJsonStr,
                        default={},
                        help="Job worker replicaSpecs.")
    parser.add_argument("--deleteAfterDone", type=strtobool,
                        default=True,
                        help="When Job done, delete the Job automatically if it is True.")
    parser.add_argument("--jobTimeoutMinutes", type=int,
                        default=60*24,
                        help="Time in minutes to wait for the Job to reach end")

    # Options that likely wont be used, but left here for future use
    parser.add_argument("--jobGroup", type=str,
                        default="kubeflow.org",
                        help="Group for the CRD, ex: kubeflow.org")
    parser.add_argument("--jobPlural", type=str,
                        default="pytorchjobs",  # We could select a launcher here and populate these automatically
                        help="Plural name for the CRD, ex: pytorchjobs")
    parser.add_argument("--kind", type=str,
                        default="PyTorchJob",
                        help="CRD kind.")
    return parser


def extract_replicas(worker_spec):
    replicas = worker_spec.get("replicas")
    if replicas is not None:
        logger.info(f"Replicas found: {replicas}")
        return int(replicas)
    else:
        logger.error("Replicas not found in worker spec.")
        return 0


COLOR_LIST = [
    "\033[36m", # Dark Cyan
    "\033[32m", # Dark Green
    "\033[33m", # Dark Yellow
    "\033[38;5;208m", # Dark Orange
    "\033[35m", # Dark Magenta
    "\033[34m", # Dark Blue
    "\033[38;5;154m", # Dark Lime Green
    "\033[31m", # Dark Red
]
COLOR_LIST = [color for _ in range(32) for color in COLOR_LIST]


def main(args):
    logger.setLevel(logging.INFO)
    logger.info("Generating job template.")

    jobSpec = V1PyTorchJobSpec(
        pytorch_replica_specs={
            "Master": args.masterSpec,
            "Worker": args.workerSpec,
        },
        active_deadline_seconds=args.activeDeadlineSeconds,
        backoff_limit=args.backoffLimit,
        clean_pod_policy=args.cleanPodPolicy,
        ttl_seconds_after_finished=args.ttlSecondsAfterFinished,
    )

    api_version = f"{args.jobGroup}/{args.version}"

    job = V1PyTorchJob(
        api_version=api_version,
        kind=args.kind,
        metadata=k8s_client.V1ObjectMeta(
            name=args.name,
            namespace=args.namespace,
        ),
        spec=jobSpec,
    )

    serialized_job = k8s_client.ApiClient().sanitize_for_serialization(job)

    logger.info("Creating launcher client.")

    config.load_incluster_config()
    api_client = k8s_client.ApiClient()
    launcher_client = launch_crd.K8sCR(
        group=args.jobGroup,
        plural=args.jobPlural,
        version=args.version,
        client=api_client
    )

    logger.info("Submitting CR.")
    create_response = launcher_client.create(serialized_job)

    expected_conditions = ["Succeeded", "Failed"]
    logger.info(
        f"Monitoring job until status is any of {expected_conditions}."
    )

    num_workers = extract_replicas(args.workerSpec)
    tracking_pod_list = [(f"{args.name}-master-0", "master", "\033[31m")] \
                      + [(f"{args.name}-worker-{index}", f"worker-{index}", COLOR_LIST[index]) for index in range(num_workers)]

    for (pod_name, prefix, color) in tracking_pod_list:
        thread = threading.Thread(target=launcher_client.print_pod_logs, args=(args.namespace, pod_name, prefix, color))
        thread.daemon = True
        thread.start()

    launcher_client.wait_for_condition(
        args.namespace, args.name, expected_conditions,
        timeout=datetime.timedelta(minutes=args.jobTimeoutMinutes),
        delete_after_done=args.deleteAfterDone)
    if args.deleteAfterDone:
        logger.info("Deleting job.")
        launcher_client.delete(args.name, args.namespace)


if __name__ == "__main__":
    parser = get_arg_parser()
    args = parser.parse_args()
    main(args)