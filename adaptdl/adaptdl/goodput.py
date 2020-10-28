# Copyright 2020 Petuum, Inc. All Rights Reserved.
#
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


import autograd
import numpy as np
import collections
import scipy.optimize
import scipy.stats


# Parameters for a performance model which predicts the per-step time of
# distributed SGD using all-reduce. At a high level, models compute time and
# network time separately, and combines them with some degree of overlap.
# Compute time is modeled as a linear function of the local batch size.
# Network time is modeled using different parameters depending on if the job
# is inter-node (there exists a pair of replicas on different nodes), or
# intra-node (all replicas are on the same node). For both cases, network time
# is modeled as a constant term plus a retrogression term which increases
# linearly with the total number of replicas.
Params = collections.namedtuple("Params", [
    # T_compute ~ alpha_c + beta_c * local_bsz +
    #             (alpha_a + beta_a * local_bsz) * accumulation_steps
    "alpha_c",  # Constant term of compute time
    "beta_c",   # Multiplicative factor of compute time
    # If inter-node: T_network ~ alpha_n + beta_n * replicas
    "alpha_n",  # Constant term of inter-node network time
    "beta_n",   # Retrogression factor of inter-node network time
    # If intra-node: T_network ~ alpha_r + beta_r * replicas
    "alpha_r",  # Constant term of intra-node network time
    "beta_r",   # Retrogression factor of intra-node network time
    # T_step ~ (T_compute ^ gamma + T_network ^ gamma) ^ (1 / gamma)
    # Essentially is a p-norm where p = gamma. When p ~ 1 then
    # T_step ~ T_compute + T_network, indicating no overlap between compute
    # and network. When p -> infinity then T_step = max(T_compute, T_network),
    # indicating perfect overlap. We limit gamma to [1, 10] since 10 is close
    # enough to approximate the max function for our purposes.
    "gamma",    # Models the degree of overlap between compute and network
])


class GoodputFunction(object):

    def __init__(self, perf_params, grad_params, init_batch_size):
        self._perf_params = perf_params
        self._grad_params = grad_params
        self._init_batch_size = init_batch_size

    def __call__(self, num_nodes, num_replicas, atomic_bsz, accum_steps):
        return self.evaluate(num_nodes, num_replicas, atomic_bsz, accum_steps)

    def evaluate(self, num_nodes, num_replicas, atomic_bsz, accum_steps):
        return (self.throughput(num_nodes, num_replicas,
                                atomic_bsz, accum_steps) *
                self.efficiency(num_replicas * atomic_bsz * (accum_steps + 1)))

    def throughput(self, num_nodes, num_replicas, atomic_bsz, accum_steps):
        step_time, _, _ = _predict(self._perf_params, num_nodes,
                                   num_replicas, atomic_bsz, accum_steps)
        batch_size = num_replicas * atomic_bsz * (accum_steps + 1)
        return batch_size / step_time

    def efficiency(self, batch_size):
        grad_sqr = self._grad_params['norm']
        grad_var = self._grad_params['var']
        scale = batch_size * self._init_batch_size
        denom = grad_var / scale + grad_sqr
        gain = np.where(denom > 0, (grad_var + grad_sqr) / denom, 1.0)
        return gain / scale

    def optimize(self, num_nodes, num_replicas, max_batch_size=None,
                 atomic_bsz_range=None, accumulation=False):
        assert np.all(np.less_equal(1, num_nodes))
        assert np.all(np.less_equal(num_nodes, num_replicas))
        if max_batch_size is None:
            max_batch_size = self._init_batch_size
        assert self._init_batch_size <= max_batch_size
        atomic_bsz_range = atomic_bsz_range or (None, None)
        min_atomic_bsz = atomic_bsz_range[0] or 1
        max_atomic_bsz = atomic_bsz_range[1] or max_batch_size
        # Remember what the output shape/format should be and flatten inputs.
        output_shape = np.broadcast(num_nodes, num_replicas).shape
        output_scalar = np.isscalar(num_nodes) or np.isscalar(num_replicas)
        num_nodes = np.broadcast_to(num_nodes, output_shape).flatten()
        num_replicas = np.broadcast_to(num_replicas, output_shape).flatten()
        # Samples 50 different total batch sizes in geometric space.
        batch_size = np.geomspace(self._init_batch_size, max_batch_size)
        local_bsz = np.expand_dims(batch_size, -1) / num_replicas
        if accumulation:
            # If local_bsz size exceeds the max atomic batch size, split it
            # into a number of batches to form (atomic_bsz, accum_steps) such
            # that (atomic_bsz * (accum_steps + 1)) is close to local_bsz.
            #
            # If num_replicas == 1 and local_bsz > self._init_batch_size, then
            # always set accum_steps to 1. This is because (1) the gradient
            # statistics used for scaling up the learning rate are inaccurate
            # when there is only one atomic minibatch to estimate them from,
            # and (2) using a lower accum_steps should always yield a higher
            # goodput when there is only a single replica.
            accum_steps = np.ceil(local_bsz / max_atomic_bsz) - 1
            accum_steps = np.where(
                (num_replicas == 1) & (local_bsz > self._init_batch_size),
                np.maximum(accum_steps, 1), accum_steps).astype(np.int)
        else:
            accum_steps = np.zeros_like(local_bsz, dtype=np.int)
        atomic_bsz = np.ceil(local_bsz / (accum_steps + 1)).astype(np.int)
        # Evaluate the goodput of all candidate configurations.
        goodput = self.evaluate(num_nodes, num_replicas,
                                atomic_bsz, accum_steps)
        # Set the goodput of invalid configurations to 0.0.
        goodput = np.where((min_atomic_bsz <= atomic_bsz) &
                           (atomic_bsz <= max_atomic_bsz), goodput, 0.0)
        # Find the indices of the best configurations.
        indices = np.argmax(goodput, axis=0), np.arange(goodput.shape[1])
        # Restore the correct output shape and return results.
        goodput = goodput[indices].reshape(output_shape)
        atomic_bsz = atomic_bsz[indices].reshape(output_shape)
        accum_steps = accum_steps[indices].reshape(output_shape)
        if output_scalar:
            goodput = goodput.item()
            atomic_bsz = atomic_bsz.item()
            accum_steps = accum_steps.item()
        return goodput, atomic_bsz, accum_steps


def fit_perf_params(nodes, replicas, local_bsz, accumulation_steps,
                    step_time, step_time_compute, accumulation_time):
    # Fit the performance model given step time and compute time measurements
    # for different configurations of nodes, replicas, local_bsz, and
    # accumulation_steps.

    # HACK: We want to use the original numpy module for calls from the
    # SpeedupFunction for performance reasons, but also need those functions to
    # use autograd.numpy when we want to differentiate them. We patch the
    # global np reference only for the code invoked rom this function.
    global np  # Replace numpy from autograd.
    orig_np = np
    np = autograd.numpy

    replicas = np.array(replicas)
    local_bsz = np.array(local_bsz)
    step_time = np.array(step_time)
    step_time_compute = np.array(step_time_compute)
    accumulation_time = np.array(accumulation_time)

    # Set initial params to reasonable values.
    params = [1e-1, 1e-2] * 3 + [1.0 + 1e-3]
    # Set lower/upper bounds for each parameter. Add a small slack to lower
    # bounds to avoid numerical instability issues.
    lower = [1e-8, 1e-8] * 3 + [1.0]
    upper = [np.inf, np.inf] * 3 + [10.0]
    if len(np.unique(local_bsz)) == 1:
        # Fix alpha_c if only observed a single local batch size.
        # This makes the speedup model optimistic with respect to
        # scaling up the batchsize. This will assign equal weight
        # to the constant and multplicative factors for compute time
        # if there is only a single datapoint (which is by far the
        # most likely case for this scenario)
        params[0] = upper[0] = lower[0] = np.mean(step_time_compute) / 2
    if not any(nodes > 1):
        # Fix alpha_n and beta_n if no multi-node observations.
        params[2] = upper[2] = lower[2]
        params[3] = upper[3] = lower[3]
    if not any(np.logical_and(nodes == 1, replicas > 1)):
        # Fix alpha_r and beta_r if no single-node/multi-replica observations.
        params[4] = upper[4] = lower[4]
        params[5] = upper[5] = lower[5]
    if not any(replicas > 2):
        # Fix beta_n and beta_r if no replicas > 2.
        params[3] = upper[3] = lower[3]
        params[5] = upper[5] = lower[5]
    bounds = scipy.optimize.Bounds(lower, upper, keep_feasible=True)
    args = (nodes, replicas, local_bsz, accumulation_steps,
            step_time, step_time_compute, accumulation_time)
    # FIXME: need to handle optimization failures and propagate to the Trainer.
    grad_fn = autograd.grad(_obj_fn)
    result = scipy.optimize.minimize(_obj_fn, params, args=args,
                                     jac=grad_fn, bounds=bounds)
    params = result.x
    np = orig_np  # Restore original numpy.
    return Params(*params)


def _predict(params, nodes, replicas, local_bsz, accumulation_steps):
    params = Params(*params)
    step_time_compute = _predict_compute(params, local_bsz)
    step_time_network = _predict_network(params, nodes, replicas)
    step_time_accumulation = step_time_compute
    gamma = params.gamma
    # Return predicted total step time in log-space to avoid numerical issues
    # in autograd and optimization.
    step_time = (
        ((step_time_compute) ** gamma + step_time_network ** gamma) **
        (1 / gamma)
        + step_time_accumulation * accumulation_steps)
    return (
        step_time,
        step_time_compute,
        step_time_network)


def _predict_compute(params, local_bsz):
    params = Params(*params)
    # Forward/backward passes should scale linearly with the batch size.
    return params.alpha_c + params.beta_c * local_bsz


def _predict_network(params, nodes, replicas):
    params = Params(*params)
    # Select the most significant link between replicas, currently either
    # inter-node (nodes > 1) or intra-node (replicas > 1). Note that if
    # replicas == 1 then neither of these two conditions are matched.
    conds = [nodes > 1, replicas > 1]
    # Bandwidth is bottlenecked by the most significant link, alpha models
    # the overhead of transferring data across that link.
    bottleneck = np.select(conds, [params.alpha_n, params.alpha_r], 1e-8)
    # Assuming ring all-reduce, communication happens in a number of rounds
    # equal to the number of replicas. beta models the performance
    # retrogression from increasing the number of replicas beyond 2.
    retrogress = np.select(conds, [params.beta_n, params.beta_r], 1e-8)
    retrogress = retrogress * np.maximum(replicas - 2, 1e-8)
    return (bottleneck + retrogress)


def _rmse(pred, true):
    return np.sqrt(((pred - true) ** 2).mean())


def _obj_fn(params, nodes, replicas, local_bsz, accumulation_steps,
            step_time, step_time_compute, accumulation_time):
    params = Params(*params)
    pred_step_time, pred_step_time_compute, _ = \
        _predict(params, nodes, replicas, local_bsz, accumulation_steps)
    # Error of total step time predictions.
    err1 = _rmse(np.log(pred_step_time),
                 np.log(step_time + accumulation_time * accumulation_steps))
    # Error of compute time predictions.
    err2 = _rmse(np.log(pred_step_time_compute), np.log(step_time_compute))
    # L2 regularization towards a smaller gamma, because it's easier to
    # optimize the alpha and beta parameters when gamma is smaller.
    reg1 = 1e-3 * (params.gamma - 1) ** 2
    # Penalize retrogression terms to prefer a more optimistic model.
    reg2 = 1e-2 * ((params.beta_n / params.alpha_n) ** 2 +
                   (params.beta_r / params.alpha_r) ** 2)
    return err1 + err2 + reg1 + reg2
