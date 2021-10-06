import copy
import ray
from adaptdl.goodput import GoodputFunction, GradParams, PerfParams


def optimize(hints, existing_ips, job_resources, max_cluster_size):
    if not hints:
        return ["virtual_node_0"]
    print(f"optimizer ips: {existing_ips}")
    existing_ips = [ip for (ip, running) in existing_ips.items() if running]
    print(f"existing ips: {existing_ips}")
    virtual_nodes = [
        f"virtual_node_{i}"
        for i in range(max_cluster_size - len(existing_ips))]

    nodes = existing_ips + virtual_nodes
    print(f"Nodes: {nodes}")

    existing_ips = set(existing_ips)
    node_resources = {
        node["NodeManagerAddress"]: node["Resources"]
        for node in ray.nodes()
        if (node["NodeManagerAddress"] in existing_ips
            and node["Resources"])}

    replicas_per_node = [0 for node in nodes]
    for index, node in enumerate(nodes):
        resources = node_resources.get(node, copy.deepcopy(job_resources))
        replicas_per_node[index] = int(
            min(resources.get(resource_type, 0.0) / value
                for resource_type, value in job_resources.items()))

    max_workers = sum(replicas_per_node)

    if "norm" in hints["gradParams"]:
        hints["gradParams"]["sqr"] = hints["gradParams"]["norm"]
        del hints["gradParams"]["norm"]
    perf_params = PerfParams(**hints["perfParams"])
    grad_params = GradParams(**hints["gradParams"])

    goodput_fn = GoodputFunction(perf_params, grad_params,
                                 hints["initBatchSize"])

    max_batch_size = hints["maxBatchSize"]
    atomic_bsz_range = hints["localBszBounds"]
    accumulation = hints["gradientAccumulation"]

    goodputs = [(num_actors,
                 goodput_fn.optimize(
                     1, num_actors, max_batch_size=max_batch_size,
                     atomic_bsz_range=atomic_bsz_range,
                     accumulation=accumulation))
                for num_actors in range(1, max_workers + 1)]

    base_goodput = goodputs[0][1][0]
    optimal_goodputs = [
        base_goodput * num_actors
        for (num_actors, (goodput, atomic_bsz, accum_steps)) in goodputs]
    max_goodput = float("-inf")
    best_replicas = 1
    for (num_actors, (goodput, atomic_bsz, accum_steps)), optimal_goodput in \
            zip(goodputs, optimal_goodputs):
        if goodput > max_goodput and (goodput >= optimal_goodput * 0.5):
            best_replicas = num_actors
            max_goodput = goodput
    print(base_goodput, goodputs, optimal_goodputs)

    result = []
    while best_replicas > 0:
        count = (min(replicas_per_node[0], best_replicas))
        if count:
            result += [(nodes[0]) * count]
            if count == replicas_per_node[0]:
                nodes = nodes[1:]
                replicas_per_node = replicas_per_node[1:]
            best_replicas -= count
        else:
            nodes = nodes[1:]
            replicas_per_node = replicas_per_node[1:]

    print(f"rescaling to {result}")
    return result
