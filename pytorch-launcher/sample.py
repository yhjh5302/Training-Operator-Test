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


def Clear_PyTorchJob(name, namespace, version="v1"):
    import kubernetes
    api_instance = kubernetes.client.CustomObjectsApi()
    group = "kubeflow.org"
    plural = "pytorchjobs"
    try:
        api_response = api_instance.delete_namespaced_custom_object(
            name=name,
            namespace=namespace,
            group=group,
            version=version,
            plural=plural,
            body=kubernetes.client.models.V1DeleteOptions(
                propagation_policy='Foreground',
                grace_period_seconds=5
            )
        )
        print("PyTorchJob deleted. Status='%s'" % str(api_response.status))
    except kubernetes.client.rest.ApiException as e:
        print("Exception when calling CustomObjectsApi->delete_namespaced_custom_object: %s\n" % e)


clear_pytorchjob_op = components.func_to_container_op(func=Clear_PyTorchJob, packages_to_install=['kubernetes'])


@dsl.pipeline(
    name="launch-kubeflow-pytorchjob",
    description="An example to launch pytorch.",
)
def custom_pipeline(namespace: str, image: str, command: str, num_worker: int, cpu_per_worker: int, memory_per_worker: int, gpu_per_worker: int) -> None:
    """
    Run MPI-Job
    Args:
        image (str): Image registry for workers (string)
        command (str): Command for workers (string)
        num_worker (int): Number of workers (integer)
        cpu_per_worker (int): CPU allocation per worker (integer: Cores)
        memory_per_worker (int): Memory allocation per worker (integer: GiB)
        gpu_per_worker (int): GPU allocation per worker (integer)
    """
    if not num_worker > 0:
        raise ValueError("num_worker must be greater than 0.")

    # PyTorchJob Test
    pytorch_job_op = components.load_component_from_file('./pytorch_job_component.yaml')
    image: str = 'yhjh5302/pytorchjob-test:latest'
    command: str = 'cd /workspace && python3 pytorchjob_train.py --batch_size=1 --backend=gloo'
    train_task = pytorch_job_op(
        name='pytorch-cnn-dist-job',
        namespace=namespace,
        master_spec='{ \
          "replicas": 1, \
          "restartPolicy": "Never", \
          "template": { \
            "metadata": { \
              "annotations": { \
                "sidecar.istio.io/inject": "false", \
                "yunikorn.apache.org/task-group-name": "task-group-example", \
                "yunikorn.apache.org/task-groups": "[{ \
                  \\"name\\": \\"task-group-example\\", \
                  \\"minMember\\": 2, \
                  \\"minResource\\": { \
                    \\"cpu\\": %s, \
                    \\"memory\\": %sGi, \
                    \\"nvidia.com/gpu\\": %s \
                  }, \
                  \\"nodeSelector\\": {}, \
                  \\"tolerations\\": [], \
                  \\"affinity\\": {} \
                }]" \
              } \
            }, \
            "spec": { \
              "containers": [ \
                { \
                  "name": "pytorch", \
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
        }' % (cpu_per_worker, memory_per_worker, gpu_per_worker, image, command, cpu_per_worker, memory_per_worker, gpu_per_worker),
        worker_spec='{ \
          "replicas": %s, \
          "restartPolicy": "Never", \
          "template": { \
            "metadata": { \
              "annotations": { \
                "sidecar.istio.io/inject": "false", \
                "yunikorn.apache.org/task-group-name": "task-group-example", \
                "yunikorn.apache.org/task-groups": "[{ \
                  \\"name\\": \\"task-group-example\\", \
                  \\"minMember\\": 2, \
                  \\"minResource\\": { \
                    \\"cpu\\": %s, \
                    \\"memory\\": %sGi, \
                    \\"nvidia.com/gpu\\": %s \
                  }, \
                  \\"nodeSelector\\": {}, \
                  \\"tolerations\\": [], \
                  \\"affinity\\": {} \
                }]" \
              } \
            }, \
            "spec": { \
              "containers": [ \
                { \
                  "name": "pytorch", \
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
        }' % (num_worker, cpu_per_worker, memory_per_worker, gpu_per_worker, image, command, cpu_per_worker, memory_per_worker, gpu_per_worker),
        delete_after_done=True
    )
    handle_exit = clear_pytorchjob_op(
      name='pytorch-cnn-dist-job',
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
