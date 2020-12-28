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
import logging
import math
import warnings

import numpy as np
import torch.distributed
import torch.optim
from torch.autograd import Variable
from types import MethodType

from adaptdl.torch.data import current_dataloader

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

__all__ = ["ScalingRuleBase", "AdaScale", "LinearScale", "SqrtScale",
           "LEGWScale"]


def _average_groups(grads1, grads2):
    ret = []
    for group1, group2 in zip(grads1, grads2):
        ret.append([])
        for g1, g2 in zip(group1, group2):
            if g1 is None:
                ret[-1].append(g2)
            elif g2 is None:
                ret[-1].append(g1)
            else:
                ret[-1].append((g1 + g2) / 2)
    return ret


def _normsqr_groups(grads):
    ret = []
    for group in grads:
        normsqr = [g.pow(2).sum(dtype=torch.float64)
                   for g in group if g is not None]
        ret.append(sum(normsqr).item() if normsqr else 0.0)
    return np.array(ret)


class GradientNoiseScale(object):
    """This class tracks gradient related stats and takes care of gradient
    accumulation."""
    def __init__(self, adp, optimizer,
                 mp_scaler=None,
                 num_replicas=None,
                 accum_scale=None):
        self._adp = adp
        self.optimizer = optimizer
        self._orig_optimizer_step = optimizer.step
        self._orig_optimizer_zero_grad = optimizer.zero_grad
        self._should_zero_grad = True
        self._mp_scaler = mp_scaler
        self._local_sqr = None
        self._num_replicas = (num_replicas if num_replicas is not None
                              else torch.distributed.get_world_size())
        self._accum_scale = accum_scale or self._num_replicas
        self._prev_grads = None

        self._reset_accumulation()

        self.optimizer.state.setdefault("scaling_rule", {
            "progress": 0.0,
            "prev_scale": 0.0,
            # Averages of n and v
            "sqr_avg": np.ones(len(optimizer.param_groups)),
            "var_avg": np.zeros(len(optimizer.param_groups)),
            # Whether estimates are biased (using differenced estimator).
            "biased": False,
        })

        for idx, param_group in enumerate(self.optimizer.param_groups):
            for param in param_group["params"]:
                param.register_hook(
                    functools.partial(self._backward_hook, idx, param))
        self._callback_queued = False
        self._smoothing = 0.999

    @property
    def state(self):
        return self.optimizer.state["scaling_rule"]

    def _reset_accumulation(self):
        self._orig_optimizer_zero_grad()
        self._local_sqr = None
        self._accum_count = 0

    @property
    def accum_scale(self):
        return self._accum_scale

    def set_accum_scale(self, accum_scale):
        if not np.isclose(self._accum_scale, accum_scale):
            self._reset_accumulation()
            self._accum_scale = accum_scale

    def sqr_avg(self):
        """
        Current estimate of the squared l2-norm of the true gradient (sigma
        squared).

        Returns (float): Estimate of squared l2-norm.
        """
        return float(np.sum(np.maximum(self.state["sqr_avg"], 0.0)))

    def var_avg(self):
        """
        Current estimate of the trace of the covariance of the true gradient
        (mu squared).

        Returns (float): Estimate of trace of the covariance.
        """
        return float(np.sum(np.maximum(self.state["var_avg"], 1e-6)))

    def get_progress(self):
        return self.state["progress"]

    def gain(self, scale):
        """
        Current estimate of the AdaScale gain ratio.

        Arguments:
            scale (float): The total scale to estimate the gain ratio for.

        Returns (float): Estimate of gain ratio.
        """
        var = self.var_avg()
        norm = self.sqr_avg()
        return (var + norm) / (var / scale + norm)

    def _update_avg(self, param_name, value, factor):
        biased = self.state.get(param_name + "_biased", 0.0)
        unbias = self.state.get(param_name + "_unbias", 0.0)
        biased = factor * biased + (1.0 - factor) * value
        unbias = factor * unbias + (1.0 - factor)
        self.state[param_name + "_biased"] = biased
        self.state[param_name + "_unbias"] = unbias
        self.state[param_name] = biased / unbias

    def _reset_avg(self, param_name):
        self.state.pop(param_name + "_biased", None)
        self.state.pop(param_name + "_unbias", None)

    def _backward_hook(self, idx, param, grad):
        # This method should be invoked once for each parameter during the
        # backward pass, before gradients are synchronized between replicas.
        if self._local_sqr is None:
            self._local_sqr = torch.zeros(len(self.optimizer.param_groups),
                                          device=grad.device,
                                          dtype=torch.float64)
        # Update the local gradient square sum
        self._local_sqr[idx] += grad.detach().pow(2).sum(dtype=torch.float64)
        if not self._callback_queued:
            Variable._execution_engine.queue_callback(self._queue_callback)
        self._callback_queued = True

    def _queue_callback(self):
        # This method should be invoked after the entire backward pass. We want
        # to make sure self._final_callback is invoked once, only after all
        # gradients have been synchronized between each replica. However, the
        # synchronization code in DistributedDataParallel is also done in a
        # callback, which might not yet be executed. Therefore, we enqueue
        # self._final_callback from this method, which should ensure it is
        # invoked after the gradient synchronization callback.
        self._callback_queued = False
        self._accum_count += 1
        if self._adp.require_backward_grad_sync:
            # Asynchronously sum the local squared-gradient statistics. The
            # actual gradient averaging should also be happening at the same
            # time, until self._final_callback is invoked.
            if self._num_replicas > 1:
                self._async_op = torch.distributed.all_reduce(self._local_sqr,
                                                              async_op=True)
            Variable._execution_engine.queue_callback(self._final_callback)
            self._should_zero_grad = True
        else:
            # Keep on accumulating gradients, should not zero grad.
            self._should_zero_grad = False

    def _final_callback(self):
        # This method should be invoked once the gradients have been
        # synchronized between all replicas and accumulation steps.
        if self._num_replicas > 1:
            self._async_op.wait()

        grads = []
        if self._mp_scaler is not None:
            mixed_precision_scale = self._mp_scaler.get_scale()
        else:
            mixed_precision_scale = 1.0
        for group in self.optimizer.param_groups:
            grads.append([])
            for param in group["params"]:
                if param.grad is None:
                    grads[-1].append(None)
                    continue
                grad = param.grad.detach().float()
                grads[-1].append(
                    grad / mixed_precision_scale / self._accum_count)

        # Note: mixed precision can result in nan/inf gradients,
        # which propogate into our norm and variance estimates.
        # Mixed precision autoscaling skips the skip where
        # there are nan/inf, so we also skip the update here
        grads_normsqr = _normsqr_groups(grads)
        if not np.all(np.isfinite(grads_normsqr)):
            LOG.warning("AdaScale detected invalid gradient! Skipping step.")
            return

        count = self._num_replicas * self._accum_count
        scale = self._accum_scale * self._accum_count
        if count > 1:
            # Average local squared-norm samples.
            local_sqr = self._local_sqr.cpu().numpy() / count
            # Gradient is squared in local_sqr, so need to square the
            # mixed precision scale as well
            local_sqr = (local_sqr / mixed_precision_scale ** 2)
            total_sqr = grads_normsqr
            if self.state["biased"]:
                self._reset_avg("sqr_avg")
                self._reset_avg("var_avg")
            self.state["biased"] = False
            self._prev_grads = None
        else:
            # Single gradient datapoint, use difference estimation.
            if self._prev_grads is not None:
                local_sqr = (_normsqr_groups(self._prev_grads) +
                             grads_normsqr) / 2
                avg_grads = _average_groups(grads, self._prev_grads)
                total_sqr = _normsqr_groups(avg_grads)
                count = 2
                scale = 2 * self._accum_scale
            self.state["biased"] = True
            self._prev_grads = [[g.clone() if g is not None else None
                                 for g in group] for group in grads]

        if count > 1:
            grad_sqr = (count * total_sqr - local_sqr) / (count - 1)
            grad_var = (local_sqr - total_sqr) * scale / (count - 1)
            theta = self._smoothing ** scale
            self._update_avg('sqr_avg', grad_sqr, theta)
            self._update_avg('var_avg', grad_var, theta)

    def zero_grad(self, *args, **kwargs):
        if self._should_zero_grad:
            self._reset_accumulation()
        else:
            warnings.warn("skipping zero_grad for accumulated gradient")

    def patch_optimizer(self, step_fn):
        """
        Monkey-patch the optimizer's step function with
        :meth:`ScalingRuleBase.step`.
        Arguments:
            step_fn (function): Scaling rule's step function.
        """
        @functools.wraps(self.optimizer.step)
        def step_wrapper(optim, *args, **kwargs):
            return step_fn(*args, **kwargs)

        @functools.wraps(self.optimizer.zero_grad)
        def zero_wrapper(optim, *args, **kwargs):
            return self.zero_grad(*args, **kwargs)
        self.optimizer.step = MethodType(step_wrapper, self.optimizer)
        self.optimizer.zero_grad = MethodType(zero_wrapper, self.optimizer)

    def run_step(self, calculate_lr_factors_fn, *args, **kwargs):
        """
        Run one optimizer step. Essentially just invokes
        ``optimizer.step(*args, **kwargs)`` with a scaled learning rate.

        Arguments:
            calculate_lr_factors_fn (function): Function used to calculate
            lr factors/multipliers.
        """
        scale = self._accum_scale * self._accum_count
        initial_lr = [pg["lr"] for pg in self.optimizer.param_groups]
        lr_factors = calculate_lr_factors_fn(scale)
        for lr_factor, pg in zip(lr_factors, self.optimizer.param_groups):
            pg["lr"] = lr_factor * pg["lr"]
        self._orig_optimizer_step(*args, **kwargs)
        for lr, pg in zip(initial_lr, self.optimizer.param_groups):
            pg["lr"] = lr
        self.state["progress"] += self.gain(scale)
        self._reset_accumulation()


class ScalingRuleBase(object):
    """
    Base class for scaling rules that has the ability to track gradient noise
    scale calculations. Its subclasses can be used in combination with
    ``adaptdl.torch.parallel.AdaptiveDataParallel`` and ``torch.optim.SGD``.

    .. code-block:: python

        optim = torch.optim.SGD(model, lr=0.001)
        adascale = AdaScale()
        model = AdaptiveDataParallel(model, optim, adascale)

        for epoch in ...:
            for batch in ...:
                optim.zero_grad()
                loss = ...
                loss.backward()
                adascale.step()
    """
    def __init__(self):
        # instance of AdaptiveDataParallel, needs to be set before any of the
        # methods can be used
        self.adp = None

    def calculate_lr_factors(self, scale):
        raise NotImplementedError

    def step(self, *args, **kwargs):
        """
        Run one optimizer step. Essentially just invokes
        ``optimizer.step(*args, **kwargs)`` with a scaled learning rate.

        Arguments:
            args: Positional arguments passed to ``optimizer.step``.
            kwargs: Keyword arguments passed to ``optimizer.step``.
        """
        if not self.adp:
            raise ValueError("AdaptiveDataParallel instance is not set!")
        if not self.adp.require_backward_grad_sync:
            return
        self.adp.gns.run_step(self.calculate_lr_factors, *args, **kwargs)

    def patch_optimizer(self):
        """
        Monkey-patch the optimizer's step function with
        :meth:`ScalingRuleBase.step`.
        """
        # TODO: detect if the optimizer has already been patched.
        self.adp.gns.patch_optimizer(self.step)


class AdaScale(ScalingRuleBase):
    """
    Implements the AdaScale_ algorithm for scaling the learning rate for
    distributed and large batch size training.

    .. _AdaScale: https://proceedings.icml.cc/static/paper_files/icml/2020/4682-Supplemental.pdf
    """  # noqa: E501

    def calculate_lr_factors(self, scale):
        """Calculate factors to be applied to lr for each parameter group."""
        var = self.adp.gns.state["var_avg"]
        sqr = self.adp.gns.state["sqr_avg"]
        var = np.maximum(var, 1e-6)
        sqr = np.maximum(sqr,  0.0)
        return (var + sqr) / (var / scale + sqr)


class LinearScale(ScalingRuleBase):

    def calculate_lr_factors(self, scale):
        return np.full(len(self.adp.gns.optimizer.param_groups), scale)


class SqrtScale(ScalingRuleBase):

    def calculate_lr_factors(self, scale):
        return np.full(len(self.adp.gns.optimizer.param_groups),
                       math.sqrt(scale))


class LEGWScale(ScalingRuleBase):
    """
    Implements the LEGWScale algorithm for scaling the learning rate.

    Essentially, with LEGWScale, lr_factor is calculated based on
    training progress as follows:
    - when current_step < base_warmup_epoch * scale * steps_per_epoch:
      `lr_factor = sqrt(scale) * progress_ratio` where
      `progress_ratio = current_step /
                        (scale * base_warmup_epochs * steps_per_epoch)`
    - when current_step >= base_warmup_epoch * scale * steps_per_epoch:
      `lr_factor = sqrt(scale)`

    In order to adapt LEGWScale to AdaptDL, `progress_ratio` is
    calculated differently as:
    `progress / (scale * base_warmup_epochs * steps_per_epoch)` where
    `progress` is the effective steps trained based on AdaptDL's
    estimation.

    Argmuents:
        base_warmup_epochs: Base warmup epochs
        data_size: total number of samples in the dataset

    .. _LEGWScale: https://arxiv.org/pdf/1901.08256.pdf
    """

    def __init__(self, base_warmup_epochs, data_size):
        super().__init__()
        self._base_warmup_epochs = base_warmup_epochs
        self._data_size = data_size

    def _legw_lr_factor(self, scale):
        dataloader = current_dataloader()
        # total training steps for warm up
        total_steps = self._base_warmup_epochs * scale * \
            self._data_size / dataloader.batch_size
        max_lr_multiplier = math.sqrt(scale)
        # effective training steps taken
        progress = self.adp.gns.get_progress()
        if progress < total_steps:
            lr_factor = max_lr_multiplier * (progress / total_steps)
        else:
            lr_factor = max_lr_multiplier
        return lr_factor

    def calculate_lr_factors(self, scale):
        lr_factor = self._legw_lr_factor(scale)
        return np.full(len(self.adp.gns.optimizer.param_groups), lr_factor)