import time, uuid

from kfp import Client as kfp_client
from kfp import auth
from kfp import dsl
from kfp import compiler
from kfp import components
from kfp_server_api import ApiException
from kubernetes.client.models import V1EnvFromSource, V1ConfigMapEnvSource, V1EnvVar


def Clear_PyTorchJob(name, namespace, version="v1"):
    import kubernetes
    kubernetes.config.load_incluster_config()
    api_instance_custom = kubernetes.client.CustomObjectsApi()
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
                grace_period_seconds=15
            )
        )
        print("PyTorchJob deleted. Status='%s'" % str(api_response.get("status", None)))
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


clear_pytorchjob_op = components.func_to_container_op(func=Clear_PyTorchJob, packages_to_install=['kubernetes'])


@dsl.pipeline(
    name="launch-kubeflow-pytorchjob",
    description="An example to launch pytorch.",
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
        nccl_conf: str = "",
        email: str = "",
        username: str = "",
    ) -> None:
    if not num_worker > 0:
        raise ValueError("num_worker must be greater than 0.")

    # PyTorchJob Test
    # name: str = 'pytorch-cnn-dist-job'
    # img: str = 'yhjh5302/pytorchjob-test:latest'
    # cmd: str = 'cd /workspace && python3 pytorchjob_train.py --batch_size=1 --backend=gloo'
    # cpu_per_worker = 20
    # memory_per_worker = 80
    # gpu_per_worker = 1

    # Settings
    pytorch_ssh_key = str(username) + "-ssh-key"
    pytorch_job_op = components.load_component_from_file('./pytorch_job_component.yaml')
    job_name = "violet-run-pipeline-ddp-" + str(uuid.uuid4().hex)[:16]
    ndr_per_worker = 1

    handle_exit = clear_pytorchjob_op(
        name=job_name,
        namespace=namespace,
        version='v1'
    )
    with dsl.ExitHandler(handle_exit):
        train_task = pytorch_job_op(
            name=job_name,
            namespace=namespace,
            master_spec='{ \
              "replicas": 1, \
              "restartPolicy": "Never", \
              "template": { \
                "metadata": { \
                  "annotations": { \
                    "sidecar.istio.io/inject": "false", \
                    "aiplatform/owner": "%s" \
                  }, \
                  "labels": { \
                    "aiplatform/task-parallelism": "multi-node", \
                    "aiplatform/task-type": "pytorch-ddp", \
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
                      "name": "pytorch", \
                      "image": "%s", \
                      "command": ["/bin/bash", "-c", "%s"], \
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
                        }, \
                        { \
                          "name": "nccl-conf", \
                          "subPath": "nccl.conf", \
                          "mountPath": "/etc/nccl.conf" \
                        }, \
                        { \
                          "name": "ssh-public-key", \
                          "subPath": "authorized_keys", \
                          "mountPath": "/root/.ssh/authorized_keys" \
                        }, \
                        { \
                          "name": "ssh-private-key", \
                          "subPath": "id_rsa_violet", \
                          "mountPath": "/root/.ssh/id_rsa" \
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
                    }, \
                    { \
                      "name": "nccl-conf", \
                      "configMap": { \
                        "name": "%s", \
                        "defaultMode": 420 \
                      } \
                    }, \
                    { \
                      "name": "ssh-public-key", \
                      "secret": { \
                        "secretName": "%s", \
                        "defaultMode": 384, \
                        "items": [ \
                          { \
                            "key": "public_key", \
                            "path": "authorized_keys" \
                          } \
                        ] \
                      } \
                    }, \
                    { \
                      "name": "ssh-private-key", \
                      "secret": { \
                        "secretName": "%s", \
                        "defaultMode": 384, \
                        "items": [ \
                          { \
                            "key": "private_key", \
                            "path": "id_rsa_violet" \
                          } \
                        ] \
                      } \
                    } \
                  ], \
                  "schedulerName": "scheduler-plugins-scheduler" \
                } \
              } \
            }' % (email, job_name, node_group_id, node_type, img, cmd, config_map_name, node_group_id, node_type, device, value, exp_nm, run_name, cpu_per_worker, memory_per_worker, gpu_per_worker, ndr_per_worker, public_vol_nm, public_vol_mnt_path, private_vol_nm, private_vol_mnt_path, public_vol_nm, public_pvc_nm, private_vol_nm, private_pvc_nm, nccl_conf, pytorch_ssh_key, pytorch_ssh_key),
            worker_spec='{ \
              "replicas": %s, \
              "restartPolicy": "Never", \
              "template": { \
                "metadata": { \
                  "annotations": { \
                    "sidecar.istio.io/inject": "false", \
                    "aiplatform/owner": "%s" \
                  }, \
                  "labels": { \
                    "aiplatform/task-parallelism": "multi-node", \
                    "aiplatform/task-type": "pytorch-ddp", \
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
                      "name": "pytorch", \
                      "image": "%s", \
                      "command": ["/bin/bash", "-c", "%s"], \
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
                        }, \
                        { \
                          "name": "nccl-conf", \
                          "subPath": "nccl.conf", \
                          "mountPath": "/etc/nccl.conf" \
                        }, \
                        { \
                          "name": "ssh-public-key", \
                          "subPath": "authorized_keys", \
                          "mountPath": "/root/.ssh/authorized_keys" \
                        }, \
                        { \
                          "name": "ssh-private-key", \
                          "subPath": "id_rsa_violet", \
                          "mountPath": "/root/.ssh/id_rsa" \
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
                    }, \
                    { \
                      "name": "nccl-conf", \
                      "configMap": { \
                        "name": "%s", \
                        "defaultMode": 420 \
                      } \
                    }, \
                    { \
                      "name": "ssh-public-key", \
                      "secret": { \
                        "secretName": "%s", \
                        "defaultMode": 384, \
                        "items": [ \
                          { \
                            "key": "public_key", \
                            "path": "authorized_keys" \
                          } \
                        ] \
                      } \
                    }, \
                    { \
                      "name": "ssh-private-key", \
                      "secret": { \
                        "secretName": "%s", \
                        "defaultMode": 384, \
                        "items": [ \
                          { \
                            "key": "private_key", \
                            "path": "id_rsa_violet" \
                          } \
                        ] \
                      } \
                    } \
                  ], \
                  "schedulerName": "scheduler-plugins-scheduler" \
                } \
              } \
            }' % (num_worker, email, job_name, node_group_id, node_type, img, cmd, config_map_name, node_group_id, node_type, device, value, exp_nm, run_name, cpu_per_worker, memory_per_worker, gpu_per_worker, ndr_per_worker, public_vol_nm, public_vol_mnt_path, private_vol_nm, private_vol_mnt_path, public_vol_nm, public_pvc_nm, private_vol_nm, private_pvc_nm, nccl_conf, pytorch_ssh_key, pytorch_ssh_key),
            delete_after_done=True
        )

        logging_task = dsl.ContainerOp(
            name="logging-op",
            image="docker.io/yhjh5302/mlflow-logging:v1",
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
