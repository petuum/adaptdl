import copy
import logging
import numpy as np
import pymoo.model.crossover
import pymoo.model.mutation
import pymoo.model.problem
import pymoo.model.repair
import pymoo.optimize

from collections import OrderedDict
from pymoo.algorithms.nsga2 import NSGA2
from pymoo.operators.crossover.util import crossover_mask
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


class JobInfo(object):
    def __init__(self, resources, speedup_fn, creation_timestamp, progress,
                 max_replicas, preemptible):
        """
        Args:
            resources (dict): Requested resources (eg. GPUs) of each replica.
            speedup_fn (SpeedupFunction): Speedup function for this job.
            creation_timestamp (datetime): Time when this job was created.
            progress (float): Total service attained by this job.
            max_replicas (int): Maximum number of replicas.
            preemptible (bool): Whether this job is preemptible.
        """
        assert progress >= 0 and max_replicas > 0
        self.resources = resources
        self.speedup_fn = speedup_fn
        self.creation_timestamp = creation_timestamp
        self.progress = progress
        self.max_replicas = max_replicas
        self.preemptible = preemptible


class NodeInfo(object):
    def __init__(self, resources, cost, eta=0):
        """
        Args:
            resources (dict): Available resources (eg. GPUs) on this node.
            cost (float): Cost per time unit of using this node.
            eta (int): Seconds until this node becomes available.
        """
        assert cost >= 0 and eta >= 0
        self.resources = resources
        self.cost = cost
        self.eta = eta


class AdaptDLPolicy(object):
    def __init__(self):
        self._prev_xs = None
        self._prev_jobs = None
        self._prev_nodes = None

    def _jobs_sort_key(self, item):
        job = item[1]
        level = np.ceil(np.log10(max(job.progress / 720, 1)))
        # Put all preemptible jobs as highest priority.
        return (job.preemptible, level, job.creation_timestamp)

    def optimize(self, jobs, nodes, prev_allocations, generations=100):
        jobs = OrderedDict(sorted(jobs.items(), key=self._jobs_sort_key))
        nodes = OrderedDict(sorted(nodes.items()))
        jobs_index = {key: idx for idx, key in enumerate(jobs)}
        nodes_index = {key: idx for idx, key in enumerate(nodes)}
        # Map previous allocations to the current jobs and nodes.
        allocations = np.zeros((len(jobs), len(nodes)), dtype=np.int)
        for job_idx, job_key in enumerate(jobs):
            for node_key in prev_allocations.get(job_key, []):
                if node_key in nodes:
                    allocations[job_idx, nodes_index[node_key]] += 1
        problem = Problem(list(jobs.values()), list(nodes.values()),
                          allocations)
        pop_size = 100
        if self._prev_xs is None:
            # Initialize population randomly.
            xs = np.random.randint(0, np.iinfo(np.int16).max,
                                   (pop_size, problem.n_var))
            xs = xs % (problem.xu + 1)
        else:
            # Carry over the previous generation.
            xs = np.zeros((len(self._prev_xs), problem.n_var), dtype=np.int)
            prev_plan = self._prev_xs.reshape(self._prev_xs.shape[0],
                                              len(self._prev_jobs), -1)
            plan = xs.reshape(prev_plan.shape[0], len(jobs), -1)
            for job_idx, job_key in enumerate(self._prev_jobs):
                for node_idx, node_key in enumerate(self._prev_nodes):
                    if job_key in jobs and node_key in nodes:
                        plan[:, jobs_index[job_key], nodes_index[node_key]] = \
                                prev_plan[:, job_idx, node_idx]
            xs = plan.reshape(plan.shape[0], -1)
        algorithm = NSGA2(
            pop_size=pop_size,
            sampling=xs,
            crossover=Crossover(),
            mutation=Mutation(),
            repair=Repair(),
            # pymoo (as of 0.3.2) has a bug which makes it crash if duplicates
            # are eliminated. TODO: re-evaluate on a future version of pymoo.
            eliminate_duplicates=False,
        )
        result = pymoo.optimize.minimize(problem, algorithm,
                                         ("n_gen", generations))
        # Get the unique individuals in the pareto front.
        front = NonDominatedSorting().do(
                result.F, only_non_dominated_front=True)
        _, idx = np.unique(result.X[front], axis=0, return_index=True)
        X = result.X[front][idx]
        F = result.F[front][idx]
        A = problem._evaluate(X, {})
        # Construct return values.
        ret = []
        for x, f, a in zip(X, F, A):
            plan = x.reshape(len(jobs), -1)
            allocations = {job_key: [] for job_key in jobs}
            for job_idx, job_key in enumerate(jobs):
                for node_idx, node_key in enumerate(nodes):
                    allocations[job_key].extend(
                            [node_key] * a[job_idx, node_idx])
            active_nodes = []
            for node_idx, node_key in enumerate(nodes):
                if np.any(plan[:, node_idx]):
                    active_nodes.append(node_key)
            ret.append((tuple(f), allocations, active_nodes))
        self._prev_xs = copy.deepcopy(result.X)
        self._prev_jobs = copy.deepcopy(jobs)
        self._prev_nodes = copy.deepcopy(nodes)
        return ret


class Problem(pymoo.model.problem.Problem):
    def __init__(self, jobs, nodes, prev_allocations):
        self.jobs = jobs
        self.nodes = nodes
        self._prev_allocations = prev_allocations
        # Find which resource types are requested by at least one job.
        rtypes = sorted(set.union(*[set(job.resources) for job in jobs]))
        # Build array of job resources.
        self._job_resources = np.zeros((len(jobs), len(rtypes)), np.int64)
        for j, job in enumerate(jobs):
            for r, rtype in enumerate(rtypes):
                self._job_resources[j, r] = job.resources.get(rtype, 0)
        # Build array of node resources.
        self._node_resources = np.zeros((len(nodes), len(rtypes)), np.int64)
        for n, node in enumerate(nodes):
            for r, rtype in enumerate(rtypes):
                self._node_resources[n, r] = node.resources.get(rtype, 0)
        # Upper bound each job: <replicas on node 0> <replicas on node 1> ...
        xu = np.zeros((len(jobs), len(nodes)), dtype=np.int)
        for j, job in enumerate(jobs):
            for n, node in enumerate(nodes):
                xu[j, n] = min(
                    node.resources.get(rtype, 0) // job.resources[rtype]
                    for rtype in rtypes if job.resources.get(rtype, 0) > 0)
        assert np.all(np.any(xu, axis=1))  # All jobs can be scheduled.
        xu = xu.flatten()
        super().__init__(n_var=xu.size, n_obj=2, xl=0, xu=xu, type_var=np.int)

    def _evaluate(self, xs, out, *args, **kwargs):
        # Evaluate each individual with full plan-ahead. Resources are assigned
        # to jobs in FIFO order, and each job can start using any subset of its
        # requested resources. Assumes each job ends when it completes its
        # estimated remaining progress, and its resources will become free to
        # be allocated to other jobs.
        #     xs shape: (population_size, num_jobs * num_nodes)
        plan = xs.reshape(len(xs), len(self.jobs), -1)
        # Set remaining = progress to emulate Least-Attained-Service (LAS).
        remaining = np.tile([j.progress for j in self.jobs], (len(xs), 1))
        remaining = np.maximum(remaining, 1.0)  # Guarantee some remaining.
        # Iterate forwards through time until no jobs are remaining.
        jct_sum = np.zeros(len(xs))
        node_time = np.zeros((len(xs), len(self.nodes)))
        node_eta = np.tile([float(n.eta) for n in self.nodes], (len(xs), 1))
        current_allocations = None
        while np.any(remaining > 0):  # Will loop roughly len(self.jobs) times.
            assert np.all(remaining >= 0)
            assert np.all(node_eta >= 0)
            # Get desired allocations for all remaining jobs.
            plan_remaining = plan * np.expand_dims(remaining > 0, axis=-1)
            # Get actual allocations for all jobs.
            allocations = self._get_allocations(plan_remaining, node_eta)
            # Get the speedup for all jobs.
            speedup = self._get_speedup(allocations)
            # Get the time duration for these allocations.
            duration = self._get_duration(speedup, remaining, node_eta)
            assert np.count_nonzero(duration)
            # Update the objective values.
            jct_sum += np.count_nonzero(remaining > 0, axis=1) * duration
            node_time += (np.any(plan_remaining, axis=1) *
                          np.expand_dims(duration, axis=1))
            # Update remaining job progress and node eta.
            remaining -= speedup * np.expand_dims(duration, axis=-1)
            node_eta -= (node_eta > 0) * np.expand_dims(duration, axis=-1)
            # Account for floating point errors.
            remaining[np.isclose(remaining, 0.0)] = 0.0
            node_eta[np.isclose(node_eta, 0.0)] = 0.0
            # First loop computes the current allocations, subsequent loops
            # compute future allocations.
            if current_allocations is None:
                current_allocations = allocations
        restart = np.any(current_allocations != self._prev_allocations, axis=2)
        jct_sum += np.sum(restart, axis=1) * 30.0  # Penalize job restarts.
        average_jct = jct_sum / len(self.jobs)
        total_cost = np.sum(node_time * [n.cost for n in self.nodes], axis=1)
        out["F"] = np.column_stack([average_jct, total_cost])
        return current_allocations  # Return allocations made at time 0.

    def _get_allocations(self, plan_remaining, node_eta):
        # Computes the actual allocations for each job, given their desired
        # allocations according to the plan.
        #     plan_remaining shape: [population_size, num_jobs, num_nodes]
        #     node_eta shape: [population_size, num_nodes]
        job_resources = np.expand_dims(self._job_resources, 1)
        # plan_resources: [population_size, num_jobs, num_nodes, num_rtypes]
        plan_resources = np.expand_dims(plan_remaining, -1) * job_resources
        # node_resources: [population_size, num_nodes, num_rtypes]
        node_resources = \
            self._node_resources * np.expand_dims(node_eta <= 0, -1)
        plan_resources = np.minimum(plan_resources.cumsum(axis=1),
                                    np.expand_dims(node_resources, 1))
        plan_resources = np.diff(plan_resources, axis=1, prepend=0)
        bignum = 999999
        allocations = np.floor_divide(plan_resources, job_resources,
                                      where=(job_resources > 0),
                                      out=np.full_like(plan_resources, bignum))
        allocations = np.amin(allocations, axis=-1)
        assert np.all(allocations < bignum)
        # Enforce at most one distributed job per node.
        distributed = np.count_nonzero(plan_remaining, axis=2) > 1
        mask = plan_remaining * np.expand_dims(distributed, axis=-1) > 0
        mask = mask.cumsum(axis=1) - 1 > 0
        allocations[mask] = 0
        return allocations  # shape: [population_size, num_jobs, num_nodes]

    def _get_speedup(self, allocations):
        # Computes the speedup of each job given their allocations.
        #     allocations shape: [population_size, num_jobs, num_nodes]
        speedup = []
        num_nodes = np.count_nonzero(allocations, axis=2)
        num_replicas = np.sum(allocations, axis=2)
        for idx, job in enumerate(self.jobs):
            speedup.append(job.speedup_fn(
                num_nodes[:, idx], num_replicas[:, idx]))
        return np.stack(speedup, axis=1)  # shape: [population_size, num_jobs]

    def _get_duration(self, speedup, remaining, node_eta):
        # Computes the duration of time until the next job completes or the
        # next node becomes available, using the given job speedup.
        #     speedup shape: [population_size, num_jobs]
        #     remaining shape: [population_size, num_jobs]
        #     node_eta shape: [population_size, num_nodes]
        # Compute next job completion time, ignore division by zero.
        with np.errstate(divide="ignore", invalid="ignore"):
            jct = np.where(remaining > 0, remaining / speedup, 0.0)
        next_job = np.amin(np.where(jct > 0, jct, np.inf),
                           initial=np.inf, axis=1)
        # Compute next node arrival time.
        next_node = np.amin(np.where(node_eta > 0, node_eta, np.inf),
                            initial=np.inf, axis=1)
        # Fast-forward until next job completes or next node arrives.
        duration = np.minimum(next_job, next_node)  # shape = (population)
        duration[np.isinf(duration)] = 0.0  # Zero if no next job or node.
        return duration  # shape: [population_size]


class Crossover(pymoo.model.crossover.Crossover):
    def __init__(self):
        super().__init__(n_parents=2, n_offsprings=2)

    def _do(self, problem, xs, **kwargs):
        # Single-point crossover over all jobs.
        n_parents, n_matings, _ = xs.shape
        n_jobs = len(problem.jobs)
        points = np.random.randint(n_jobs, size=n_matings)
        mask = np.arange(n_jobs) < np.expand_dims(points, -1)
        ys = crossover_mask(xs.reshape(n_parents, n_matings, n_jobs, -1), mask)
        return ys.reshape(n_parents, n_matings, -1)


class Mutation(pymoo.model.mutation.Mutation):
    def _do(self, problem, xs, **kwargs):
        # Select variables randomly, then assign each of them a random value
        # within their upper bounds.
        plan = xs.reshape(xs.shape[0], len(problem.jobs), -1)
        num_nonzero = np.count_nonzero(plan, axis=2)
        num_zero = len(problem.nodes) - num_nonzero
        # Try to balance the number of mutations to zero/nonzero elements.
        prob = 0.5 / np.where(plan > 0,
                              np.expand_dims(np.maximum(num_nonzero, 1), 2),
                              np.expand_dims(np.maximum(num_zero, 1), 2))
        prob = prob.reshape(xs.shape[0], -1)
        m = np.random.random(xs.shape) < prob
        r = np.random.randint(np.iinfo(np.int16).max, size=xs.shape)
        return (xs + m * r) % (problem.xu + 1)


class Repair(pymoo.model.repair.Repair):
    def _do(self, problem, pop, **kwargs):
        xs = pop.get("X")
        plan = xs.reshape(xs.shape[0], len(problem.jobs), -1)
        # Enforce at least one replica per job.
        cumsum = np.cumsum(problem.xu.reshape(len(problem.jobs), -1), axis=-1)
        cumsum = np.broadcast_to(cumsum, plan.shape)
        sample = np.random.uniform(high=cumsum[:, :, [-1]])
        sample = np.diff(cumsum > sample, axis=-1, prepend=0)
        mask = np.all(plan == 0, axis=-1)
        plan[mask, :] = sample[mask, :]
        # Enforce no more than max replicas per job.
        max_replicas = np.array([j.max_replicas for j in problem.jobs])
        while True:
            overflow = np.maximum(np.sum(plan, axis=2) - max_replicas, 0)
            if np.all(overflow == 0):
                break
            fuzzy = plan + np.random.random(plan.shape)
            fuzzy[fuzzy < 1] = np.inf
            mask = fuzzy == np.amin(fuzzy, axis=2, keepdims=True)
            plan -= mask * np.expand_dims(overflow, 2)
            plan = np.maximum(plan, 0)
        return pop.new("X", plan.reshape(xs.shape[0], -1))
