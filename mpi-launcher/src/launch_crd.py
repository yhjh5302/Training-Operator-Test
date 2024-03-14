# Copyright 2019 kubeflow.org.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import datetime
import json
import logging
import multiprocessing
import time

from kubernetes import client as k8s_client
from kubernetes.client import rest

logger = logging.getLogger(__name__)

class K8sCR(object):
  def __init__(self, group, plural, version, client):
    self.group = group
    self.plural = plural
    self.version = version
    self.custom_client = k8s_client.CustomObjectsApi(client)
    self.core_client = k8s_client.CoreV1Api(client)

  def print_pod_logs(self, namespace, pod_name):
    stream = []
    max_retry = 5760 # 24h
    for _ in range(max_retry):
      try:
        pod = self.core_client.read_namespaced_pod(name=pod_name, namespace=namespace)
        pod_phase = pod.status.phase
        if pod_phase in ['Pending', 'Init', 'PodInitializing']:
          logger.info(f"{namespace}/{pod_name} is {pod_phase}... retry")
          time.sleep(15)
        else:
          break
      except rest.ApiException as e:
        if e.status == 404:
          logger.info(f"{namespace}/{pod_name} not found... retry")
          time.sleep(15)

    stream = self.core_client.read_namespaced_pod_log(name=pod_name, namespace=namespace, follow=True, _preload_content=False)
    for line in stream:
      logger.info(f"{line.decode().strip()}")
    logger.info(f"Stop reading {namespace}/{pod_name} logs")

  def wait_for_condition(self,
                         namespace,
                         name,
                         expected_conditions=[],
                         timeout=datetime.timedelta(days=365),
                         polling_interval=datetime.timedelta(seconds=15),
                         status_callback=None,
                         delete_after_done=False):
    """Waits until any of the specified conditions occur.
    Args:
      namespace: namespace for the CR.
      name: Name of the CR.
      expected_conditions: A list of conditions. Function waits until any of the
        supplied conditions is reached.
      timeout: How long to wait for the CR.
      polling_interval: How often to poll for the status of the CR.
      status_callback: (Optional): Callable. If supplied this callable is
        invoked after we poll the CR. Callable takes a single argument which
        is the CR.
    """
    end_time = datetime.datetime.now() + timeout
    try:
      while True:
        try:
          results = self.custom_client.get_namespaced_custom_object(
            self.group, self.version, namespace, self.plural, name)
        except Exception as e:
          logger.error("There was a problem waiting for %s/%s %s in namespace %s; Exception: %s [%s]",
                        self.group, self.plural, name, namespace, e, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
          raise

        if results:
          if status_callback:
            status_callback(results)
          expected, condition = self.is_expected_conditions(results, expected_conditions)
          if expected:
            logger.info("%s/%s %s in namespace %s has reached the expected condition: %s. [%s]",
                        self.group, self.plural, name, namespace, condition, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            return results
          else:
            if condition:
              logger.info("Current condition of %s/%s %s in namespace %s is %s. [%s]",
                    self.group, self.plural, name, namespace, condition, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        if datetime.datetime.now() + polling_interval > end_time:
          raise Exception(
            "Timeout waiting for {0}/{1} {2} in namespace {3} to enter one of the "
            "conditions {4}.".format(self.group, self.plural, name, namespace, expected_conditions))

        time.sleep(polling_interval.seconds)
    except:
      if delete_after_done:
        self.delete(name, namespace)
      raise

  def is_expected_conditions(self, inst, expected_conditions):
      conditions = inst.get("status", {}).get("conditions")
      if not conditions:
          return False, ""
      if conditions[-1]["type"] in expected_conditions and conditions[-1]["status"] == "True":
          return True, conditions[-1]["type"]
      else:
          return False, conditions[-1]["type"]

  def create(self, spec):
    """Create a CR.
    Args:
      spec: The spec for the CR.
    """
    try:
      # Create a Resource
      namespace = spec["metadata"].get("namespace", "default")
      logger.info("Creating %s/%s %s in namespace %s.",
        self.group, self.plural, spec["metadata"]["name"], namespace)
      api_response = self.custom_client.create_namespaced_custom_object(
        self.group, self.version, namespace, self.plural, spec)
      logger.info("Created %s/%s %s in namespace %s.",
        self.group, self.plural, spec["metadata"]["name"], namespace)
      return api_response
    except rest.ApiException as e:
      self._log_and_raise_exception(e, "create")

  def delete(self, name, namespace):
    try:
      body = {
        # Set garbage collection so that CR won't be deleted until all
        # owned references are deleted.
        "propagationPolicy": "Foreground",
      }
      logger.info("Deleteing %s/%s %s in namespace %s.",
        self.group, self.plural, name, namespace)
      api_response = self.custom_client.delete_namespaced_custom_object(
        group=self.group,
        version=self.version,
        namespace=namespace,
        plural=self.plural,
        name=name,
        body=body)
      logger.info("Deleted %s/%s %s in namespace %s.",
        self.group, self.plural, name, namespace)
      return api_response
    except rest.ApiException as e:
      self._log_and_raise_exception(e, "delete")

  def _log_and_raise_exception(self, ex, action):
    message = ""
    if getattr(ex, "message", None):
      message = ex.message
    if getattr(ex, "body", None):
      try:
        body = json.loads(ex.body)
        message = body.get("message")
      except ValueError:
        logger.error("Exception when %s %s/%s: %s", action, self.group, self.plural, ex.body)
        raise

    logger.error("Exception when %s %s/%s: %s", action, self.group, self.plural, message)
    raise ex


class K8sPodGroup(object):
  def __init__(self, client):
    self.custom_client = k8s_client.CustomObjectsApi(client)
    self.core_client = k8s_client.CoreV1Api(client)
    self.group = "scheduling.x-k8s.io"
    self.version = "v1alpha1"
    self.plural = "podgroups"

  def create(self, name, namespace, num_pod, schedule_timeout_seconds):
    body = {
      "kind": "PodGroup",
      "metadata": {
        "name": name,
        "namespace": namespace
      },
      "spec": {
        "scheduleTimeoutSeconds": schedule_timeout_seconds,
        "minMember": num_pod
      }
    }
    try:
      logger.info("Creating podgroup %s in namespace %s.", name, namespace)
      api_response = self.custom_client.create_namespaced_custom_object(
        group=self.group,
        version=self.version,
        namespace=namespace,
        plural=self.plural,
        body=body)
      logger.info("Created podgroup %s in namespace %s.", name, namespace)
      return api_response
    except rest.ApiException as e:
      self._log_and_raise_exception(e, "create")

  def delete(self, name, namespace):
    try:
      body = {
        # Set garbage collection so that CR won't be deleted until all
        # owned references are deleted.
        "propagationPolicy": "Foreground",
      }
      logger.info("Deleteing podgroup %s in namespace %s.", name, namespace)
      api_response = self.custom_client.delete_namespaced_custom_object(
        group=self.group,
        version=self.version,
        namespace=namespace,
        plural=self.plural,
        name=name,
        body=body)
      logger.info("Deleted podgroup %s in namespace %s.", name, namespace)
      return api_response
    except rest.ApiException as e:
      self._log_and_raise_exception(e, "delete")

  def _log_and_raise_exception(self, ex, action):
    message = ""
    if getattr(ex, "message", None):
      message = ex.message
    if getattr(ex, "body", None):
      try:
        body = json.loads(ex.body)
        message = body.get("message")
      except ValueError:
        logger.error("Exception when %s podgroup: %s", action, ex.body)
        return

    logger.error("Exception when %s podgroup: %s", action, message)
    return


class K8sService(object):
  def __init__(self, client):
    self.custom_client = k8s_client.CustomObjectsApi(client)
    self.core_client = k8s_client.CoreV1Api(client)

  def create(self, name, namespace, labels=None, ports=None):
    metadata = k8s_client.V1ObjectMeta(name=name, labels=labels)
    spec = k8s_client.V1ServiceSpec(selector=labels, ports=ports)
    body = k8s_client.V1Service(api_version="v1", kind="Service", metadata=metadata, spec=spec)
    try:
      logger.info("Creating service %s in namespace %s.", name, namespace)
      api_response = self.core_client.create_namespaced_service(namespace, body)
      logger.info("Created service %s in namespace %s.", name, namespace)
      return api_response
    except rest.ApiException as e:
      self._log_and_raise_exception(e, "create")

  def delete(self, name, namespace):
    try:
      logger.info("Deleteing service %s in namespace %s.", name, namespace)
      api_response = self.core_client.delete_namespaced_service(name=name, namespace=namespace, body={"propagationPolicy": "Foreground"})
      logger.info("Deleted servicee %s in namespace %s.", name, namespace)
      return api_response
    except rest.ApiException as e:
      self._log_and_raise_exception(e, "delete")

  def _log_and_raise_exception(self, ex, action):
    message = ""
    if getattr(ex, "message", None):
      message = ex.message
    if getattr(ex, "body", None):
      try:
        body = json.loads(ex.body)
        message = body.get("message")
      except ValueError:
        logger.error("Exception when %s service: %s", action, ex.body)
        return

    logger.error("Exception when %s service: %s", action, message)
    return


class K8sServiceAccount(object):
  def __init__(self, client):
    self.custom_client = k8s_client.CustomObjectsApi(client)
    self.core_client = k8s_client.CoreV1Api(client)

  def delete(self, name, namespace):
    try:
      logger.info("Deleteing service account %s in namespace %s.", name, namespace)
      api_response = self.core_client.delete_namespaced_service_account(name=name, namespace=namespace, body={"propagationPolicy": "Foreground"})
      logger.info("Deleted servicee account %s in namespace %s.", name, namespace)
      return api_response
    except rest.ApiException as e:
      self._log_and_raise_exception(e, "delete")

  def _log_and_raise_exception(self, ex, action):
    message = ""
    if getattr(ex, "message", None):
      message = ex.message
    if getattr(ex, "body", None):
      try:
        body = json.loads(ex.body)
        message = body.get("message")
      except ValueError:
        logger.error("Exception when %s service account: %s", action, ex.body)
        return

    logger.error("Exception when %s service account: %s", action, message)
    return
