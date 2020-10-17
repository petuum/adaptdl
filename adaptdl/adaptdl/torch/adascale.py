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


import functools

import numpy as np
import torch.distributed
import torch.optim
from torch.autograd import Variable


__all__ = ["AdaScale"]


def _normsq(params):
    """
    Returns the square of the L2 norm for each elem of params
    as an np array
    """
    return np.asarray([p.pow(2).sum().item() for p in params])


class AdaScale(object):
    """
    Implements the AdaScale_ algorithm for scaling the learning rate for
    distributed and large batch size training. Can be used in combination with
    ``torch.nn.parallel.DistributedDataParallel`` and ``torch.optim.SGD``.

    .. code-block:: python

        optim = torch.optim.SGD(model, lr=0.001)
        model = DistributedDataParallel(model)
        adascale = AdaScale(optim)

        for epoch in ...:
            for batch in ...:
                optim.zero_grad()
                loss = ...
                loss.backward()
                adascale.step()

    Arguments:
        optimizer (torch.optim.Optimizer): Optimizer to apply AdaScale to.
        scale (float): Scaling factor of the batch size, e.g. using a 10x
            larger batch size (summed across all replicas) means a scale of
            10. If None, defaults to ``num_replicas``.
        num_replicas (int): Number of replicas for distributed training. If
            None, defaults to ``torch.distributed.get_world_size()``.
        patch_optimizer (bool): If True, monkey-patches the ``step`` method of
            the optimizer with the AdaScale ``step`` method.

    .. _AdaScale: https://proceedings.icml.cc/static/paper_files/icml/2020/4682-Supplemental.pdf
    """  # noqa: E501
    def __init__(self, optimizer, scale=None,
                 num_replicas=None, patch_optimizer=False):
        self._optimizer = optimizer
        self._optimizer_step = optimizer.step
        self._sum_local_norm = None
        self._norms_future = None
        self._num_replicas = (num_replicas if num_replicas is not None
                              else torch.distributed.get_world_size())
        self._num_params = \
            int(sum(len(pg["params"]) for pg in optimizer.param_groups))
        self._prev_full_grad = None
        self._norms = None
        self._made_step = False
        self._accumulation_steps = 1
        self._current_accumulation_step = 0

        self._optimizer.state.setdefault("adascale", {
            "norm": np.zeros(self._num_params),
            "replicas": 0.0,

            # Averages of n and v
            "norm_avg": np.ones(self._num_params),
            "var_avg": np.zeros(self._num_params),
        })

        self.set_scale(self._num_replicas if scale is None else scale)

        idx = 0
        for param_group in self._optimizer.param_groups:
            for param in param_group["params"]:
                param.register_hook(
                    functools.partial(self._backward_hook, idx))
                idx += 1

        if patch_optimizer:
            self.patch_optimizer()
        self._smoothing = 0.997

    @property
    def _state(self):
        return self._optimizer.state["adascale"]

    @property
    def scale(self):
        """
        The scaling factor of the current batch size, relative to the baseline
        batch size when training with a single replica. For example, if the
        baseline batch size is 32, but using a scaled-up batch size of 80, then
        then the scaling factor is 2.5.
        """
        return self._scale

    def set_scale(self, scale):
        """
        Set the scaling factor of the current batch size. It is up to the
        application to invoke this function to make sure that AdaScale's
        scaling factor matches the actual batch size used during training.

        Arguments:
            scale (float): New scaling factor to be applied to AdaScale.
        """
        if self._state['replicas'] == 1 and self._num_replicas > 1:
            # TODO: when to reset running averages should be decided outside of
            # the AdaScale object.
            self._reset_avg("norm")
            self._reset_avg("norm_avg")
            self._reset_avg("var_avg")
        self._scale = scale
        self._state['replicas'] = self._num_replicas

    def set_accumulation_steps(self, accumulation_steps):
        """
        Set the number of batches sampled before performing an optimizer
        step for gradient accumulation. Also resets the current step number
        for gradient accumulation to 0

        Arguments:
            accumulation_steps (int): new number of batches sampled before
                                  stepping.
        """
        accumulation_steps = accumulation_steps + 1
        if accumulation_steps != self._accumulation_steps:
            self._current_accumulation_step = 0
            self._accumulation_steps = int(accumulation_steps)
            self._norms = None

    def is_accumulation_step(self):
        return self._current_accumulation_step != self._accumulation_steps - 1

    def norm_avg(self):
        """
        Current estimate of the squared l2-norm of the true gradient (sigma
        squared).

        Returns (float): Estimate of squared l2-norm.
        """
        return np.sum(self._state["norm_avg"])

    def var_avg(self):
        """
        Current estimate of the trace of the covariance of the true gradient
        (mu squared).

        Returns (float): Estimate of trace of the covariance.
        """
        return np.sum(self._state["var_avg"])

    def get_progress(self, scale=None):
        if self._made_step:
            return self.gain(scale)
        else:
            return 0.0

    def gain(self, scale=None):
        """
        Current estimate of the AdaScale gain ratio.

        Arguments:
            scale (float): The batch size scale to estimate the gain ratio for.

        Returns (float): Estimate of gain ratio.
        """
        scale = self._scale if scale is None else scale
        var = self.var_avg()
        norm = self.norm_avg()
        return (var + norm) / (var / scale + norm)

    def _update_avg(self, param_name, value, factor):
        biased = self._state.get(param_name + "_biased", 0.0)
        unbias = self._state.get(param_name + "_unbias", 0.0)
        biased = factor * biased + (1.0 - factor) * value
        unbias = factor * unbias + (1.0 - factor)
        self._state[param_name + "_biased"] = biased
        self._state[param_name + "_unbias"] = unbias
        self._state[param_name] = biased / unbias

    def _reset_avg(self, param_name):
        self._state.pop(param_name + "_biased", None)
        self._state.pop(param_name + "_unbias", None)

    def _backward_hook(self, idx, grad):
        # This method should be invoked once for each parameter during the
        # backward pass, before gradients are synchronized between replicas.
        if self._norms is None:
            self._norms = torch.zeros((self._accumulation_steps,
                                       self._num_params),
                                      device=grad[0].device)
        self._norms[self._current_accumulation_step][idx] = \
            grad.pow(2).sum()
        grad /= self._accumulation_steps
        self._final_callback_queued = False
        Variable._execution_engine.queue_callback(self._queue_callback)

    def _queue_callback(self):
        # This method should be invoked after the entire backward pass. We want
        # to make sure self._final_callback is invoked once, only after all
        # gradients have been synchronized between each replica. However, the
        # synchronization code in DistributedDataParallel is also done in a
        # callback, which might not yet be executed. Therefore, we enqueue
        # self._final_callback from this method, which should ensure it is
        # invoked after the gradient synchronization callback.
        if self._final_callback_queued:
            return
        self._final_callback_queued = True
        Variable._execution_engine.queue_callback(self._final_callback)

    def _final_callback(self):
        # This method should be invoked once for each backward pass, after
        # gradients have been synchronized between each replica.
        self._final_callback_queued = False
        if (self.is_accumulation_step()):
            return
        grad = []
        total_steps = self._accumulation_steps
        for group in self._optimizer.param_groups:
            grad.extend([p.grad.detach().clone()
                         if p.grad is not None else
                         torch.zeros([1], dtype=torch.float64)
                         for p in group["params"]])

        theta = self._smoothing ** self._scale
        has_previous_step = not (self._norms_future is None)

        if has_previous_step:
            if self._num_replicas > 1:
                self._norms_future[0].wait()
            norms_sum, samples = self._norms_future[1]
            norms = norms_sum.cpu().numpy()
        if self._num_replicas > 1:
            norms_sum = torch.sum(self._norms, 0)
            self._norms_future = (
                torch.distributed.all_reduce(
                    norms_sum, async_op=True),
                (norms_sum,
                 self._state['replicas'] * total_steps))
        else:
            self._norms_future = (
                None, (torch.sum(self._norms, 0),
                       self._state['replicas'] * total_steps))

        if has_previous_step:
            if samples > 1:
                # DistributedDataParallel averages gradients across replica,
                # but we also need to average the gradients across the
                # gradient accumulation steps manually
                n = _normsq(self._prev_full_grad)
                var = norms / (samples - 1)
                var -= n * (samples / (samples - 1))
                var *= (self._scale / samples)
                var = np.maximum(var, 1e-6)
                norm = n - var / self._scale
                norm = np.maximum(norm, 0.0)
                self._update_avg('norm_avg', norm, theta)
                self._update_avg('var_avg', var, theta)
            # Single gradient datapoint, use difference estimation.
            else:
                prev_grad = self._prev_full_grad
                n = _normsq([(g1 + g2) / 2 for g1, g2 in zip(prev_grad, grad)])
                var = np.array([(g1.pow(2).sum() + g2.pow(2).sum()).item()
                                for g1, g2 in zip(prev_grad, grad)])
                var -= 2 * n
                var *= self._scale
                var = np.maximum(var, 1e-6)
                norm = n - var / (2 * self._scale)
                norm = np.maximum(norm, 0.0)
                self._update_avg('norm_avg', norm, theta)
                self._update_avg('var_avg', var, theta)
        self._norms = None
        self._prev_full_grad = grad

    def step(self, *args, **kwargs):
        """
        Run one optimizer step using Adascale. Essentially just invokes
        ``optimizer.step(*args, **kwargs)`` with a scaled learning rate.

        Arguments:
            args: Positional arguments passed to ``optimizer.step``.
            kwargs: Keyword arguments passed to ``optimizer.step``.
        """
        if (self.is_accumulation_step()):
            self._current_accumulation_step += 1
            self._made_step = False
            return

        self._current_accumulation_step = 0
        self._made_step = True
        initial_lr = [pg["lr"] for pg in self._optimizer.param_groups]
        offset = 0
        for param_group in self._optimizer.param_groups:
            size = len(param_group["params"])
            grad_sqr = self._state["norm_avg"][offset:offset + size].sum()
            grad_var = self._state["var_avg"][offset:offset + size].sum()
            gain = (grad_var + grad_sqr) / (grad_var / self.scale + grad_sqr)
            param_group["lr"] = gain * param_group["lr"]
            offset += size
        self._optimizer_step(*args, **kwargs)
        for lr, param_group in zip(initial_lr, self._optimizer.param_groups):
            param_group["lr"] = lr

    def patch_optimizer(self):
        """
        Monkey-patch the optimizer's step function with :meth:`AdaScale.step`.
        """
        # TODO: detect if the optimizer has already been patched.

        @functools.wraps(self._optimizer.step)
        def wrapper(*args, **kwargs):
            return self.step(*args, **kwargs)
        old_zero_grad = self._optimizer.zero_grad

        @functools.wraps(self._optimizer.zero_grad)
        def zero_wrapper(*args, **kwargs):
            if self._made_step:
                return old_zero_grad(*args, **kwargs)
            return None
        self._optimizer.step = wrapper
        self._optimizer.zero_grad = zero_wrapper
