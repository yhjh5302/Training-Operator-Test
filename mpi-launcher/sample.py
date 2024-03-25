import time

from kfp import Client as kfp_client
from kfp import auth
from kfp import dsl
from kfp import compiler
from kfp import components
from kfp_server_api import ApiException
from kubernetes.client.models import V1EnvFromSource, V1ConfigMapEnvSource, V1EnvVar


def Clear_MPI_Job(name, namespace, num_worker, version="v1"):
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
        api_response = api_instance_custom.delete_namespaced_custom_object(
            name=name,
            namespace=namespace,
            group="scheduling.x-k8s.io",
            version="v1alpha1",
            plural="podgroups",
            body=kubernetes.client.models.V1DeleteOptions(
                propagation_policy='Foreground',
                grace_period_seconds=15
            )
        )
        print("PodGroup %s/%s deleted. Status='%s'" % (name, namespace, str(api_response.get("status", None))))
    except kubernetes.client.rest.ApiException as e:
        print("Exception when calling CustomObjectsApi->delete_namespaced_custom_object: %s\n" % e)

    try:
        api_response = api_instance_core.delete_namespaced_service(name=f"{name}-launcher", namespace=namespace, body={"propagationPolicy": "Foreground"})
        print("Service %s/%s deleted. Status='%s'" % (f"{name}-launcher", namespace, str(api_response.status)))
        for index in range(int(num_worker)):
            api_response = api_instance_core.delete_namespaced_service(name=f"{name}-worker-{index}", namespace=namespace, body={"propagationPolicy": "Foreground"})
            print("Service %s/%s deleted. Status='%s'" % (f"{name}-worker-{index}", namespace, str(api_response.status)))
    except kubernetes.client.rest.ApiException as e:
        print("Exception when calling CoreV1Api->delete_namespaced_service: %s\n" % e)

    try:
        api_response = api_instance_core.delete_namespaced_service_account(name=f"{name}-launcher", namespace=namespace, body={"propagationPolicy": "Foreground"})
        print("ServiceAccount %s/%s deleted. Status='%s'" % (f"{name}-launcher", namespace, str({'conditions': None, 'load_balancer': {'ingress': None}})))
    except kubernetes.client.rest.ApiException as e:
        print("Exception when calling CoreV1Api->delete_namespaced_service_account: %s\n" % e)


clear_mpijob_op = components.func_to_container_op(func=Clear_MPI_Job, packages_to_install=['kubernetes'])


@dsl.pipeline(
    name="launch-kubeflow-mpi-job",
    description="An example to launch deepspeed.",
)
def custom_pipeline(
        namespace: str = "",
        application_id: str = "",
        queue: str = "",
        img: str = "",
        cmd: str = "",
        num_worker: int = 3,
        cpu_per_worker: int = 20,
        memory_per_worker: int = 80,
        gpu_per_worker: int = 1,
        node_group_id: int = 1,
        node_type: str = "",
        public_pvc_nm: str = "",
        public_vol_nm: str = "",
        public_vol_mnt_path: str = "",
        private_pvc_nm: str = "",
        private_vol_nm: str = "",
        private_vol_mnt_path: str = "",
        exp_nm: str = "",
        run_name: str = "",
        config_map_name: str = "",
        device: str = "",
        value: str = "",
    ) -> None:
    if not num_worker > 0:
        raise ValueError("num_worker must be greater than 0.")

    # MPI-Job Test
    mpi_job_op = components.load_component_from_file('./mpi_job_component.yaml')
    sshd_cmd = r"sed -i'' -e's/^#   StrictHostKeyChecking ask/   StrictHostKeyChecking no/' /etc/ssh/ssh_config \
 && sed -i'' -e's/^#PermitRootLogin prohibit-password$/PermitRootLogin yes/' /etc/ssh/sshd_config \
 && sed -i'' -e's/^#PasswordAuthentication yes$/PasswordAuthentication no/' /etc/ssh/sshd_config \
 && sed -i'' -e's/^#PermitUserEnvironment no$/PermitUserEnvironment yes/' /etc/ssh/sshd_config \
 && sed -i'' -e's/^#PermitEmptyPasswords no$/PermitEmptyPasswords no/' /etc/ssh/sshd_config \
 && sed -i'' -e's/^UsePAM yes/UsePAM no/' /etc/ssh/sshd_config \
 && /usr/sbin/sshd"
    # name: str = 'deepspeed-cnn-dist-job'
    # img: str = 'yhjh5302/deepspeed-test:latest'
    # cmd: str = '/usr/sbin/sshd && deepspeed -H /etc/mpi/hostfile deepspeed_train.py --deepspeed --deepspeed_config config.json'
    # cpu_per_worker = 20
    # memory_per_worker = 80
    # gpu_per_worker = 1
    ndr_per_worker = 1

    handle_exit = clear_mpijob_op(
        name=run_name,
        namespace=namespace,
        num_worker=num_worker,
        version='v1'
    )
    with dsl.ExitHandler(handle_exit):
        train_task = mpi_job_op(
            name=run_name,
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
                    "aiplatform/task-parallelism": "multi-node", \
                    "aiplatform/task-type": "deepspeed", \
                    "app": "yunikorn", \
                    "scheduling.x-k8s.io/pod-group": "%s" \
                  } \
                }, \
                "spec": { \
                  "affinity": { \
                    "nodeAffinity": { \
                      "requiredDuringSchedulingIgnoredDuringExecution": { \
                        "nodeSelectorTerms": [{ \
                          "matchExpressions": [ \
                            { \
                              "key": "aiplatform/node-group-id", \
                              "operator": "In", \
                              "values": ["%s"] \
                            }, \
                            { \
                              "key": "aiplatform/node-type", \
                              "operator": "In", \
                              "values": ["%s"] \
                            } \
                          ] \
                        }] \
                      } \
                    } \
                  }, \
                  "containers": [ \
                    { \
                      "name": "deepspeed", \
                      "image": "%s", \
                      "command": ["/bin/bash", "-c", "%s && %s"], \
                      "envFrom": [{ \
                        "configMapRef": { \
                          "name": "%s", \
                          "optional": true \
                        } \
                      }], \
                      "env": [ \
                        { \
                          "name": "node_group_id", \
                          "value": "%s" \
                        }, \
                        { \
                          "name": "node_type", \
                          "value": "%s" \
                        }, \
                        { \
                          "name": "%s", \
                          "value": "%s" \
                        }, \
                        { \
                          "name": "MLFLOW_EXPERIMENT_NAME", \
                          "value": "%s" \
                        }, \
                        { \
                          "name": "MLFLOW_RUN_NAME", \
                          "value": "%s" \
                        } \
                      ], \
                      "resources": { \
                        "limits": { \
                          "cpu": %s, \
                          "memory": %s, \
                          "nvidia.com/gpu": %s, \
                          "rdma/rdma_shared_device_ndr": %s \
                        } \
                      }, \
                      "volumeMounts": [ \
                        { \
                          "name": "%s", \
                          "mountPath": "%s" \
                        }, \
                        { \
                          "name": "%s", \
                          "mountPath": "%s" \
                        }, \
                        { \
                          "name": "dshm", \
                          "mountPath": "/dev/shm" \
                        } \
                      ], \
                      "securityContext": { \
                        "capabilities": { \
                          "add": [ \
                            "IPC_LOCK" \
                          ] \
                        } \
                      } \
                    } \
                  ], \
                  "volumes": [ \
                    { \
                      "name": "%s", \
                      "persistentVolumeClaim": { "claimName": "%s" } \
                    }, \
                    { \
                      "name": "%s", \
                      "persistentVolumeClaim": { "claimName": "%s" } \
                    }, \
                    { \
                      "name": "dshm", \
                      "emptyDir": { \
                        "medium": "Memory" \
                      } \
                    } \
                  ], \
                  "schedulerName": "scheduler-plugins-scheduler" \
                } \
              } \
            }' % (run_name, node_group_id, node_type, img, sshd_cmd, cmd, config_map_name, node_group_id, node_type, device, value, exp_nm, run_name, cpu_per_worker, memory_per_worker, gpu_per_worker, ndr_per_worker, public_vol_nm, public_vol_mnt_path, private_vol_nm, private_vol_mnt_path, public_vol_nm, public_pvc_nm, private_vol_nm, private_pvc_nm),
            worker_spec='{ \
              "replicas": %s, \
              "restartPolicy": "Never", \
              "template": { \
                "metadata": { \
                  "annotations": { \
                    "sidecar.istio.io/inject": "false" \
                  }, \
                  "labels": { \
                    "aiplatform/task-parallelism": "multi-node", \
                    "aiplatform/task-type": "deepspeed", \
                    "app": "yunikorn", \
                    "scheduling.x-k8s.io/pod-group": "%s" \
                  } \
                }, \
                "spec": { \
                  "affinity": { \
                    "nodeAffinity": { \
                      "requiredDuringSchedulingIgnoredDuringExecution": { \
                        "nodeSelectorTerms": [{ \
                          "matchExpressions": [ \
                            { \
                              "key": "aiplatform/node-group-id", \
                              "operator": "In", \
                              "values": ["%s"] \
                            }, \
                            { \
                              "key": "aiplatform/node-type", \
                              "operator": "In", \
                              "values": ["%s"] \
                            } \
                          ] \
                        }] \
                      } \
                    } \
                  }, \
                  "containers": [ \
                    { \
                      "name": "deepspeed", \
                      "image": "%s", \
                      "command": ["/bin/bash", "-c", "%s && sleep infinity"], \
                      "envFrom": [{ \
                        "configMapRef": { \
                          "name": "%s", \
                          "optional": true \
                        } \
                      }], \
                      "env": [ \
                        { \
                          "name": "node_group_id", \
                          "value": "%s" \
                        }, \
                        { \
                          "name": "node_type", \
                          "value": "%s" \
                        }, \
                        { \
                          "name": "%s", \
                          "value": "%s" \
                        }, \
                        { \
                          "name": "MLFLOW_EXPERIMENT_NAME", \
                          "value": "%s" \
                        }, \
                        { \
                          "name": "MLFLOW_RUN_NAME", \
                          "value": "%s" \
                        } \
                      ], \
                      "resources": { \
                        "limits": { \
                          "cpu": %s, \
                          "memory": %s, \
                          "nvidia.com/gpu": %s, \
                          "rdma/rdma_shared_device_ndr": %s \
                        } \
                      }, \
                      "volumeMounts": [ \
                        { \
                          "name": "%s", \
                          "mountPath": "%s" \
                        }, \
                        { \
                          "name": "%s", \
                          "mountPath": "%s" \
                        }, \
                        { \
                          "name": "dshm", \
                          "mountPath": "/dev/shm" \
                        } \
                      ], \
                      "securityContext": { \
                        "capabilities": { \
                          "add": [ \
                            "IPC_LOCK" \
                          ] \
                        } \
                      } \
                    } \
                  ], \
                  "volumes": [ \
                    { \
                      "name": "%s", \
                      "persistentVolumeClaim": { "claimName": "%s" } \
                    }, \
                    { \
                      "name": "%s", \
                      "persistentVolumeClaim": { "claimName": "%s" } \
                    }, \
                    { \
                      "name": "dshm", \
                      "emptyDir": { \
                        "medium": "Memory" \
                      } \
                    } \
                  ], \
                  "schedulerName": "scheduler-plugins-scheduler" \
                } \
              } \
            }' % (num_worker, run_name, node_group_id, node_type, img, sshd_cmd, config_map_name, node_group_id, node_type, device, value, exp_nm, run_name, cpu_per_worker, memory_per_worker, gpu_per_worker, ndr_per_worker, public_vol_nm, public_vol_mnt_path, private_vol_nm, private_vol_mnt_path, public_vol_nm, public_pvc_nm, private_vol_nm, private_pvc_nm),
            delete_after_done=True
        )

        logging_task = dsl.ContainerOp(
            name="logging-op",
            image="docker.io/jomi0330/mlflow-logging:prod",
            command=["sh", "-c", "python mlflow_run_detail.py"],
            output_artifact_paths={"mlpipeline-ui-metadata": "/mlpipeline-ui-metadata.json"}
        )
        logging_task.add_env_variable(V1EnvVar(name="MLFLOW_EXPERIMENT_NAME", value=exp_nm))
        logging_task.add_env_variable(V1EnvVar(name="MLFLOW_RUN_NAME", value=run_name))
        config_map_ref = V1ConfigMapEnvSource(name=config_map_name, optional=True)
        logging_task.add_env_from(V1EnvFromSource(config_map_ref=config_map_ref))
        logging_task.after(train_task)


# credentials = auth.ServiceAccountTokenVolumeCredentials(path=None)
# cookies = 'authservice_session='
# namespace = 'yhjin'
# client = kfp_client(
#     host=f"http://ml-pipeline.kubeflow:8888",
#     namespace=namespace,
#     # credentials=credentials,
#     cookies=cookies,
# )

pipeline_name = 'deepspeed-test'
pipeline_version_name = f'deepspeed-test-{int(time.time())}'
compiler.Compiler().compile(custom_pipeline, 'custom_pipeline.yaml')
# try:
#   client.upload_pipeline(pipeline_name=pipeline_name, pipeline_package_path='custom_pipeline.yaml')
# except ApiException:
#   client.upload_pipeline_version(pipeline_name=pipeline_name, pipeline_package_path='custom_pipeline.yaml', pipeline_version_name=pipeline_version_name)
