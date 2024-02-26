import time

from kfp import Client as kfp_client
from kfp import auth
from kfp import dsl
from kfp import compiler
from kfp import components
from kfp_server_api import ApiException


def get_current_namespace():
    """Returns current namespace if available, else kubeflow"""
    try:
        current_namespace = open(
            "/var/run/secrets/kubernetes.io/serviceaccount/namespace"
        ).read()
    except:
        current_namespace = "kubeflow"
    return current_namespace


def Clear_MPI_Job(name, namespace, version):
    import kubernetes
    kubernetes.config.load_incluster_config()
    api_instance_custom = kubernetes.client.CustomObjectsApi()
    api_instance_core = kubernetes.client.CoreV1Api()
    group = "kubeflow.org"
    plural = "mpijobs"
    try:
        api_response = api_instance_custom.delete_namespaced_custom_object(
            name=name,
            namespace=namespace,
            group=group,
            version=version,
            plural=plural,
            body=kubernetes.client.models.V1DeleteOptions(
                propagation_policy='Foreground',
                grace_period_seconds=15
            )
        )
        print("MPI-Job %s/%s deleted. Status='%s'" % (name, namespace, str(api_response.get("status", None))))
    except kubernetes.client.rest.ApiException as e:
        print("Exception when calling CustomObjectsApi->delete_namespaced_custom_object: %s\n" % e)

    try:
        api_response = api_instance_core.delete_namespaced_service(name=f"{name}-launcher", namespace=namespace, body={"propagationPolicy": "Foreground"})
        print("Service %s/%s deleted. Status='%s'" % (f"{name}-launcher", namespace, str(api_response.status)))
        for index in range(num_workers):
            api_response = api_instance_core.delete_namespaced_service(name=f"{name}-worker-{index}", namespace=namespace, body={"propagationPolicy": "Foreground"})
            print("Service %s/%s deleted. Status='%s'" % (f"{name}-worker-{index}", namespace, str(api_response.status)))
    except kubernetes.client.rest.ApiException as e:
        print("Exception when calling CoreV1Api->delete_namespaced_service: %s\n" % e)

    try:
        api_response = api_instance_core.delete_namespaced_service_account(name=f"{name}-launcher", namespace=namespace, body={"propagationPolicy": "Foreground"})
        print("ServiceAccount %s/%s deleted. Status='%s'" % (f"{name}-launcher", namespace, str(api_response.status)))
    except kubernetes.client.rest.ApiException as e:
        print("Exception when calling CoreV1Api->delete_namespaced_service_account: %s\n" % e)


clear_mpijob_op = components.func_to_container_op(func=Clear_MPI_Job, packages_to_install=['kubernetes'])


@dsl.pipeline(
    name="launch-kubeflow-mpi-job",
    description="An example to launch deepspeed.",
)
def custom_pipeline(name: str, namespace: str, image: str, command: str, num_worker: int, cpu_per_worker: int, memory_per_worker: int, gpu_per_worker: int) -> None:
    """
    Run MPI-Job
    Args:
        image (str): Image registry for workers (string)
        command (str): Command for workers (string)
        num_worker (int): Number of workers (integer)
        num_worker (int): Number of workers (integer)
        cpu_per_worker (int): CPU allocation per worker (integer: Cores)
        memory_per_worker (int): Memory allocation per worker (integer: GiB)
        gpu_per_worker (int): GPU allocation per worker (integer)
    """
    if not num_worker > 0:
        raise ValueError("num_worker must be greater than 0.")

    # MPI-Job Test
    mpi_job_op = components.load_component_from_file('./mpi_job_component.yaml')
    name: str = 'deepspeed-cnn-dist-job'
    image: str = 'yhjh5302/deepspeed-test:latest'
    command: str = '/usr/sbin/sshd && deepspeed -H /etc/mpi/hostfile deepspeed_train.py --deepspeed --deepspeed_config config.json'
    
    train_task = mpi_job_op(
        name=name,
        namespace=namespace,
        launcher_spec='{ \
          "replicas": 1, \
          "restartPolicy": "Never", \
          "template": { \
            "metadata": { \
              "annotations": { \
                "sidecar.istio.io/inject": "false" \
              }, \
              "labels": { \
                "pod-group.scheduling.x-k8s.io/name": %s \
              } \
            }, \
            "spec": { \
              "containers": [ \
                { \
                  "name": "deepspeed", \
                  "image": "%s", \
                  "command": ["/bin/bash", "-c", "%s"], \
                  "resources": { \
                    "limits": { \
                      "cpu": %s, \
                      "memory": %sGi, \
                      "nvidia.com/gpu": %s \
                    } \
                  }, \
                  "volumeMounts": [ \
                    { \
                        "name": "dshm", \
                        "mountPath": "/dev/shm" \
                    } \
                  ] \
                } \
              ], \
              "volumes": [ \
                { \
                  "name": "dshm", \
                  "emptyDir": { \
                    "medium": "Memory" \
                  } \
                } \
              ] \
            } \
          } \
        }' % (name, image, command, cpu_per_worker, memory_per_worker, gpu_per_worker),
        worker_spec='{ \
          "replicas": %s, \
          "restartPolicy": "Never", \
          "template": { \
            "metadata": { \
              "annotations": { \
                "sidecar.istio.io/inject": "false" \
              }, \
              "labels": { \
                "pod-group.scheduling.x-k8s.io/name": %s \
              } \
            }, \
            "spec": { \
              "containers": [ \
                { \
                  "name": "deepspeed", \
                  "image": "%s", \
                  "command": ["/bin/bash", "-c", "sleep 864000"], \
                  "resources": { \
                    "limits": { \
                      "cpu": %s, \
                      "memory": %sGi, \
                      "nvidia.com/gpu": %s \
                    } \
                  }, \
                  "volumeMounts": [ \
                    { \
                        "name": "dshm", \
                        "mountPath": "/dev/shm" \
                    } \
                  ] \
                } \
              ], \
              "volumes": [ \
                { \
                  "name": "dshm", \
                  "emptyDir": { \
                    "medium": "Memory" \
                  } \
                } \
              ] \
            } \
          } \
        }' % (num_worker, name, image, cpu_per_worker, memory_per_worker, gpu_per_worker),
        delete_after_done=True
    )
    handle_exit = clear_mpijob_op(
      name=name,
      namespace=namespace,
      version='v1'
    )
    handle_exit.after(train_task)


# credentials = auth.ServiceAccountTokenVolumeCredentials(path=None)
cookies = 'authservice_session='
namespace = 'yhjin'
client = kfp_client(
    host=f"http://ml-pipeline.kubeflow:8888",
    namespace=namespace,
    # credentials=credentials,
    cookies=cookies,
)

pipeline_name = 'deepspeed-test'
pipeline_version_name = f'deepspeed-test-{int(time.time())}'
compiler.Compiler().compile(custom_pipeline, 'custom_pipeline.yaml')
try:
  client.upload_pipeline(pipeline_name=pipeline_name, pipeline_package_path='custom_pipeline.yaml')
except ApiException:
  client.upload_pipeline_version(pipeline_name=pipeline_name, pipeline_package_path='custom_pipeline.yaml', pipeline_version_name=pipeline_version_name)
