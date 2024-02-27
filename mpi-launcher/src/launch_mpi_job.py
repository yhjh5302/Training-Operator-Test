import argparse
import datetime
from distutils.util import strtobool
import logging
import yaml
import threading

from kubernetes import client as k8s_client
from kubernetes import config

import launch_crd
from v1_mpi_job import V1MPIJob as V1MPIJob_original
from v1_mpi_job_spec import V1MPIJobSpec as V1MPIJobSpec_original


logging.basicConfig(level=logging.INFO, format="%(message)s")

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


# Patch MPIJob APIs to align with k8s usage
class V1MPIJob(V1MPIJob_original):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.openapi_types = self.swagger_types


class V1MPIJobSpec(V1MPIJobSpec_original):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.openapi_types = self.swagger_types


def get_arg_parser():
    parser = argparse.ArgumentParser(description="Kubeflow Job launcher")
    parser.add_argument("--name", type=str,
                        default="mpijob",
                        help="Job name.")
    parser.add_argument("--namespace", type=str,
                        default=get_current_namespace(),
                        help="Job namespace.")
    parser.add_argument("--version", type=str,
                        default="v1",
                        help="Job version.")
    parser.add_argument("--schedulingPolicy", type=yamlOrJsonStr,
                        default={},
                        help="MPIJob supports the gang-scheduling.")
    parser.add_argument("--scheduleTimeoutSeconds", type=int,
                        default=60*60*24,
                        help="Time in seconds to wait for the Job to reach end")
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
    parser.add_argument("--launcherSpec", type=yamlOrJsonStr,
                        default={},
                        help="Job launcher replicaSpecs.")
    parser.add_argument("--workerSpec", type=yamlOrJsonStr,
                        default={},
                        help="Job worker replicaSpecs.")
    parser.add_argument("--deleteAfterDone", type=strtobool,
                        default=True,
                        help="When Job done, delete the Job automatically if it is True.")
    parser.add_argument("--jobTimeoutMinutes", type=int,
                        default=60*24*3,
                        help="Time in minutes to wait for the Job to reach end")

    # Options that likely wont be used, but left here for future use
    parser.add_argument("--jobGroup", type=str,
                        default="kubeflow.org",
                        help="Group for the CRD, ex: kubeflow.org")
    parser.add_argument("--jobPlural", type=str,
                        default="mpijobs",  # We could select a launcher here and populate these automatically
                        help="Plural name for the CRD, ex: mpijobs")
    parser.add_argument("--kind", type=str,
                        default="MPIJob",
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


def main(args):
    logger.setLevel(logging.INFO)
    logger.info("Generating job template.")

    jobSpec = V1MPIJobSpec(
        mpi_replica_specs={
            "Launcher": args.launcherSpec,
            "Worker": args.workerSpec,
        },
        scheduling_policy=args.schedulingPolicy,
        active_deadline_seconds=args.activeDeadlineSeconds,
        backoff_limit=args.backoffLimit,
        clean_pod_policy=args.cleanPodPolicy,
        ttl_seconds_after_finished=args.ttlSecondsAfterFinished,
    )

    api_version = f"{args.jobGroup}/{args.version}"

    job = V1MPIJob(
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

    podgroup_client = launch_crd.K8sPodGroup(client=api_client)
    serivce_client = launch_crd.K8sService(client=api_client)
    service_account_client = launch_crd.K8sServiceAccount(client=api_client)

    logger.info("Submitting Launcher Service.")
    create_response = serivce_client.create(
        name=f"{args.name}-launcher",
        namespace=args.namespace,
        labels={
            "training.kubeflow.org/job-name": args.name,
            "training.kubeflow.org/job-role": "master",
            "training.kubeflow.org/operator-name": "mpijob-controller",
            "training.kubeflow.org/replica-type": "launcher"
        },
        ports=[k8s_client.V1ServicePort(name="mpijob-port", protocol="TCP", port=22, target_port=22)]
    )

    logger.info("Submitting Worker Services.")
    num_workers = extract_replicas(args.workerSpec)
    for index in range(num_workers):
        create_response = serivce_client.create(
            name=f"{args.name}-worker-{index}",
            namespace=args.namespace,
            labels={
                "training.kubeflow.org/job-name": args.name,
                "training.kubeflow.org/operator-name": "mpijob-controller",
                "training.kubeflow.org/replica-index": f"{index}",
                "training.kubeflow.org/replica-type": "worker"
            },
            ports=[k8s_client.V1ServicePort(name="mpijob-port", protocol="TCP", port=22, target_port=22)]
        )

    logger.info("Submitting PodGroups.")
    num_pod = num_workers + 1
    create_response = podgroup_client.create(
        name=f"{args.name}",
        namespace=args.namespace,
        num_pod=num_pod,
        schedule_timeout_seconds=args.scheduleTimeoutSeconds
    )

    expected_conditions = ["Succeeded", "Failed"]
    logger.info(
        f"Monitoring job until status is any of {expected_conditions}."
    )

    thread = threading.Thread(target=launcher_client.print_pod_logs, args=(args.namespace, f"{args.name}-launcher"))
    thread.daemon = True
    thread.start()

    launcher_client.wait_for_condition(
        args.namespace, args.name, expected_conditions,
        timeout=datetime.timedelta(minutes=args.jobTimeoutMinutes),
        delete_after_done=args.deleteAfterDone)
    if args.deleteAfterDone:
        logger.info("Deleting job.")
        launcher_client.delete(args.name, args.namespace)
        logger.info("Deleting podgroup.")
        podgroup_client.delete(args.name, args.namespace)
        logger.info("Deleting services.")
        serivce_client.delete(f"{args.name}-launcher", args.namespace)
        for index in range(num_workers):
            serivce_client.delete(f"{args.name}-worker-{index}", args.namespace)
        logger.info("Deleting service account.")
        service_account_client.delete(f"{args.name}-launcher", args.namespace)


if __name__ == "__main__":
    parser = get_arg_parser()
    args = parser.parse_args()
    main(args)
