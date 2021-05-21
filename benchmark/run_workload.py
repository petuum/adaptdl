#!/usr/bin/env python3

import argparse
import copy
import itertools
import os
import pandas
import subprocess
import time
import yaml
from kubernetes import client, config, watch


def build_images(models, repository):
    # Build image for each model and upload to registry.
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
    templates = {}
    for model in models:
        with open(os.path.join(models_dir, model, "adaptdljob.yaml")) as f:
            template = yaml.load(f)
        dockerfile = os.path.join(models_dir, model, "Dockerfile")
        image = repository + ":" + model
        subprocess.check_call(["docker", "build", "-t", image, project_root, "-f", dockerfile])
        subprocess.check_call(["docker", "push", image])
        repodigest = subprocess.check_output(
                ["docker", "image", "inspect", image, "--format={{index .RepoDigests 0}}"])
        repodigest = repodigest.decode().strip()
        template["spec"]["template"]["spec"]["containers"][0]["image"] = repodigest
        templates[model] = template
    return templates


def cache_images(templates):
    # Cache job images on all nodes in the cluster.
    daemonset = {
        "apiVersion": "apps/v1",
        "kind": "DaemonSet",
        "metadata": {"name": "images"},
        "spec": {
            "selector": {"matchLabels": {"name": "images"}},
            "template": {
                "metadata": {"labels": {"name": "images"}},
                "spec": {
                    "containers": [],
                    "imagePullSecrets": [{"name": "stagingsecret"}],
                }
            }
        }
    }
    for name, template in templates.items():
        daemonset["spec"]["template"]["spec"]["containers"].append({
            "name": name,
            "image": template["spec"]["template"]["spec"]["containers"][0]["image"],
            "command": ["sleep", "1000000000"],
        })
    apps_api = client.AppsV1Api()
    namespace = config.list_kube_config_contexts()[1]["context"].get("namespace", "default")
    apps_api.create_namespaced_daemon_set(namespace, daemonset)
    while True:
        # Wait for DaemonSet to be ready.
        obj = apps_api.read_namespaced_daemon_set("images", namespace)
        ready = obj.status.number_ready
        total = obj.status.desired_number_scheduled
        print("caching images on all nodes: {}/{}".format(ready, total))
        if total > 0 and ready >= total: break
        time.sleep(10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("workload", type=str, help="path to workload csv")
    parser.add_argument("repository", type=str, help="url to docker repository")
    args = parser.parse_args()

    workload = pandas.read_csv(args.workload)

    config.load_kube_config()

    templates = build_images(["bert", "cifar10", "deepspeech2", "imagenet", "ncf", "yolov3"], args.repository)
    cache_images(templates)

    objs_api = client.CustomObjectsApi()
    namespace = config.list_kube_config_contexts()[1]["context"].get("namespace", "default")
    obj_args = ("adaptdl.petuum.com", "v1", namespace, "adaptdljobs")

    print("start workload")
    start = time.time()
    for row in workload.sort_values(by="time").itertuples():
        while time.time() - start < row.time:
            time.sleep(1)
        print("submit job {} at time {}".format(row, time.time() - start))
        job = copy.deepcopy(templates[row.application])
        job["metadata"].pop("generateName")
        job["metadata"]["name"] = row.name
        job["spec"].update({
            "application": row.application,
            "targetNumReplicas": row.num_replicas,
            "targetBatchSize": row.batch_size,
        })
        if row.num_replicas:
            job["spec"]["minReplicas"] = job["spec"]["maxReplicas"] = row.num_replicas
        volumes = job["spec"]["template"]["spec"].setdefault("volumes", [])
        volumes.append({
            "name": "pollux",
            "persistentVolumeClaim": { "claimName": "pollux" },
        })
        mounts = job["spec"]["template"]["spec"]["containers"][0].setdefault("volumeMounts", [])
        mounts.append({
            "name": "pollux",
            "mountPath": "/pollux/checkpoint",
            "subPath": "pollux/checkpoint/" + row.name,
        })
        mounts.append({
            "name": "pollux",
            "mountPath": "/pollux/tensorboard",
            "subPath": "pollux/tensorboard/" + row.name,
        })
        env = job["spec"]["template"]["spec"]["containers"][0].setdefault("env", [])
        env.append({"name": "ADAPTDL_CHECKPOINT_PATH", "value": "/pollux/checkpoint"})
        env.append({"name": "ADAPTDL_TENSORBOARD_LOGDIR", "value": "/pollux/tensorboard"})
        env.append({"name": "APPLICATION", "value": row.application})
        env.append({"name": "TARGET_NUM_REPLICAS", "value": str(row.num_replicas)})
        env.append({"name": "TARGET_BATCH_SIZE", "value": str(row.batch_size)})
        env.append({"name": "TRACE_EFFICIENCY", "value": "true"})
        print(yaml.dump(job))
        objs_api.create_namespaced_custom_object(*obj_args, job)
