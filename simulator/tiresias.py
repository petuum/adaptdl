import collections
import copy


class TiresiasPolicy(object):
    def __init__(self, time_fn):
        self._time_fn = time_fn
        self._queue_threshold = 3600 * 16
        self._solve_starvation = 0
        self._queue_0 = []
        self._queue_1 = []
        self._status = {}
        self._last_check_time = collections.Counter()
        self._total_executed_time = collections.Counter()
        self._executed_time = collections.Counter()
        self._last_pending_time = collections.Counter()
        self._pending_time = collections.Counter()

    def optimize(self, jobs, nodes, prev_allocations, node_template):
        event_time = int(self._time_fn())
        # Remove completed jobs.
        self._queue_0 = [key for key in self._queue_0 if key in jobs]
        self._queue_1 = [key for key in self._queue_1 if key in jobs]
        self._status = {key: val for key, val in self._status.items() if key in jobs}
        allocations = {key: val for key, val in prev_allocations.items() if key in jobs}
        # Add new jobs to pending.
        for key, job in jobs.items():
            if key not in self._status:
                self._status[key] = 'PENDING'
                self._queue_0.append(key)
        # Update queues.
        for key, job in jobs.items():
            assert self._status[key] in ('RUNNING', 'PENDING')
            if self._status[key] == 'RUNNING':  # Job is running.
                tmp = int(event_time - self._last_check_time[key]) 
                self._total_executed_time[key] = int(self._total_executed_time[key] + tmp)
                self._executed_time[key] = int(self._executed_time[key] + tmp) # decide job priority queue
                self._last_check_time[key] = event_time
                # check demotion
                gputime = int(self._executed_time[key] * job.max_replicas)
                if key in self._queue_0 and gputime >= self._queue_threshold:
                    self._queue_0.pop(self._queue_0.index(key))
                    self._queue_1.append(key)
                    print("job {} demote to Q1".format(key))
            elif self._status[key] == 'PENDING':
                tmp = int(event_time - self._last_check_time[key]) 
                self._last_check_time[key] = event_time
                self._pending_time[key] = int(self._pending_time[key] + tmp) #this is the total pending_time
                if self._executed_time[key] > 0: # if not started yet, job is always in Q0 and no need to push_back
                    self._last_pending_time[key] = int(self._last_pending_time[key] + tmp) #this is the total pending_time
                #Q0 job no need to push_back, and must be a runned 
                if self._solve_starvation > 0 and key not in self._queue_0 and \
                        self._total_executed_time[key] > 0 and self._executed_time[key] > 0:
                    if self._last_pending_time[key] >= int(self._executed_time[key] * self._solve_starvation):
                        self._executed_time[key] = 0
                        self._last_pending_time[key] = 0
                        self._queue_0.append(key)
                        self._queue_1.pop(self._queue_1.index(key))
        # Update statuses.
        total_gpus = {idx: int(node.resources['nvidia.com/gpu']) for idx, node in nodes.items()}
        num_gpus = sum(total_gpus.values())
        for queue in (self._queue_0, self._queue_1):
            for idx in queue:
                if jobs[idx].max_replicas <= num_gpus:
                    self._status[idx] = 'RUNNING'
                    num_gpus -= jobs[idx].max_replicas
                else:
                    self._status[idx] = 'PENDING'
                    allocations.pop(idx, None)
        # Update allocations.
        free_gpus = collections.Counter(total_gpus) - collections.Counter(sum(allocations.values(), []))
        for queue in (self._queue_0, self._queue_1):
            for idx in queue:
                if self._status[idx] == 'RUNNING' and not allocations.get(idx):
                    # Allocate resources.
                    allocations[idx] = []
                    while len(allocations[idx]) < jobs[idx].max_replicas:
                        node_idx, count = free_gpus.most_common(1)[0]
                        num = min(count, jobs[idx].max_replicas - len(allocations[idx]))
                        allocations[idx].extend([node_idx] * num)
                        free_gpus[node_idx] -= num
        # Objective values, allocations, active nodes.
        return allocations, len(nodes)
