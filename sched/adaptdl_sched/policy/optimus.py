import collections
import copy
import math


class OptimusPolicy(object):
    def __init__(self):
        pass

    def optimize(self, jobs, nodes, prev_allocations, node_template):
        allocations = {k: v for k, v in prev_allocations.items() if k in jobs}
        for job in jobs.values():
            completion_epoch = job.application.get_completion_epoch(
                    job.target_batch_size)
            if completion_epoch <= job.epoch:
                job.remaining = 1
            else:
                job.remaining = (job.application.get_iteration(job.target_batch_size, completion_epoch) -
                                 job.application.get_iteration(job.target_batch_size, job.epoch))
        min_replicas = {}
        for key, job in jobs.items():
            min_replicas[key] = 1  # math.ceil(job.target_batch_size / job.application.max_local_bsz)
        num_gpus = sum(node.resources["nvidia.com/gpu"] for node in nodes.values())
        num_replicas = {}
        gain = {}
        for key, job in sorted(jobs.items(), key=lambda item: min_replicas[item[0]]):
            if min_replicas[key] > num_gpus:
                num_replicas[key] = 0
                gain[key] = 0
                continue
            num_replicas[key] = min_replicas[key]
            num_gpus -= min_replicas[key]
            if num_replicas[key] + 1 > job.max_replicas or num_gpus < 1:
                gain[key] = 0
            else:
                gain[key] = (self.predict_step_time(job, num_replicas[key]) -
                             self.predict_step_time(job, num_replicas[key] + 1)) * job.remaining
        # Add resources in order of maximum marginal gain.
        while num_gpus > 0 and max(gain.values()) > 0:
            key = max(gain, key=lambda k: gain[k])
            job = jobs[key]
            num_replicas[key] += 1
            if num_replicas[key] + 1 > job.max_replicas or num_gpus < 1:
                gain[key] = 0
            else:
                gain[key] = (self.predict_step_time(job, num_replicas[key]) -
                             self.predict_step_time(job, num_replicas[key] + 1)) * job.remaining
            num_gpus -= 1
        # Placements.
        allocations = {k: v for k, v in allocations.items() if len(v) == num_replicas[k]}
        job_keys = sorted(jobs, key=lambda k: num_replicas[k])
        total_gpus = {idx: int(node.resources['nvidia.com/gpu']) for idx, node in nodes.items()}
        free_gpus = collections.Counter(total_gpus) - collections.Counter(sum(allocations.values(), []))
        for key in job_keys:
            if num_replicas[key] > 0 and not allocations.get(key):
                # Allocate resources.
                allocations[key] = []
                while len(allocations[key]) < num_replicas[key]:
                    node_idx, count = free_gpus.most_common(1)[0]
                    num = min(count, num_replicas[key] - len(allocations[key]))
                    allocations[key].extend([node_idx] * num)
                    free_gpus[node_idx] -= num
        return allocations, len(nodes)

    def predict_step_time(self, job, num_replicas):
        placement = ()
        while sum(placement) < num_replicas:
            placement = (*placement, min(num_replicas - sum(placement), 4))
        local_bsz = math.ceil(job.target_batch_size / num_replicas - 1e-8)
        accum_steps = math.ceil(local_bsz / job.application.max_local_bsz - 1e-8) - 1
        if num_replicas == 1:
            accum_steps = max(1, accum_steps)
        atomic_bsz = math.ceil(local_bsz / (accum_steps + 1) - 1e-8)
        count = num_replicas * (accum_steps + 1)
        atomic_bsz = min(atomic_bsz, int(job.application.max_batch_size / count))
        #throughput = job.speedup_fn._goodput_fn.throughput(len(placement), num_replicas, atomic_bsz, accum_steps)
        #return atomic_bsz * count / throughput
        step_time, sync_time = job.application.get_throughput(placement, atomic_bsz)
        return step_time + (step_time - sync_time) * accum_steps
