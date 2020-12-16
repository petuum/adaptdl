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
import typing
import math

import numpy as np
import torch.distributed
import torch.optim
from torch.autograd import Variable

__all__ = ["TBDBase", "AdaScale", "LinearScale", "SqrtScale"]


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
        normsqr = [g.pow(2).sum() for g in group if g is not None]
        ret.append(sum(normsqr).item() if normsqr else 0.0)
    return np.array(ret)


class _ScalingRuleBase(object):
    """
    Base class for scaling rules.

    Arguments:
        optimizer (torch.optim.Optimizer): Optimizer to apply this scaling rule
        to.
        scale (float): Scaling factor of the batch size, e.g. using a 10x
            larger batch size (summed across all replicas) means a scale of
            10. If None, defaults to ``num_replicas``.
        num_replicas (int): Number of replicas for distributed training. If
            None, defaults to ``torch.distributed.get_world_size()``.
        patch_optimizer (bool): If True, monkey-patches the ``step`` method of
            the optimizer with this scaling rule's ``step`` method.
    """

    def __init__(self, optimizer, scale=None, num_replicas=None,
                 patch_optimizer=False):
        self._optimizer = optimizer
        self._optimizer_step = optimizer.step
        self._accumulation_steps = 1
        self._current_accumulation_step = 0
        self._num_replicas = (num_replicas if num_replicas is not None
                              else torch.distributed.get_world_size())
        self._scale = 1.0
        self.set_scale(self._num_replicas if scale is None else scale)

        self._made_step = False

        if patch_optimizer:
            self.patch_optimizer()

    @property
    def scale(self):
        """
        The scaling factor of the current batch size, relative to the baseline
        batch size when training with a single replica. For example, if the
        baseline batch size is 32, but using a scaled-up batch size of 80, then
        the scaling factor is 2.5.
        """
        return self._scale

    def set_scale(self, scale):
        """
        Set the scaling factor of the current batch size.

        Arguments:
            scale (float): New scaling factor to be applied.
        """
        self._scale = scale

    def is_accumulation_step(self):
        return self._current_accumulation_step != self._accumulation_steps - 1

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

    def _calculate_scale_factors(self):
        """
        Calculate a list of multipliers to be applied to the current learning
        rate.
        """
        raise NotImplementedError

    def step(self, *args, **kwargs):
        """
        Run one optimizer step. Essentially just invokes
        ``optimizer.step(*args, **kwargs)`` with a scaled learning rate.

        Arguments:
            args: Positional arguments passed to ``optimizer.step``.
            kwargs: Keyword arguments passed to ``optimizer.step``.
        """
        if self.is_accumulation_step():
            self._current_accumulation_step += 1
            self._made_step = False
            return

        self._current_accumulation_step = 0
        self._made_step = True
        initial_lr = [pg["lr"] for pg in self._optimizer.param_groups]
        lr_factors = self._calculate_scale_factors()
        for idx, param_group in enumerate(self._optimizer.param_groups):
            param_group["lr"] = lr_factors[idx] * param_group["lr"]
        self._optimizer_step(*args, **kwargs)
        for lr, param_group in zip(initial_lr, self._optimizer.param_groups):
            param_group["lr"] = lr

    def get_progress(self):
        raise NotImplementedError

    def to_tensorboard(self, writer, global_step, tag_prefix=""):
        """
        Output some useful metrics to TensorBoard.

        Arguments:
            writer (torch.utils.tensorboard.SummaryWriter): ``SummaryWriter``
                object to output metrics to.
            global_step (int): Global step value to record.
            tag_prefix (str): Prefix added to each metric's tag.
        """
        pass

    def patch_optimizer(self):
        """
        Monkey-patch the optimizer's step function with :meth:`self.step`.
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


# Surpass type checking errors while making sure that `object` is inherited.
_Base = _ScalingRuleBase if typing.TYPE_CHECKING else object


class _GradientNoiseScaleMixin(_Base):
    """
    This Mixin class that makes ScalingRuleBase able to track gradient related
    stats.
    """

    def __init__(self, optimizer, scale=None,
                 num_replicas=None, patch_optimizer=False):
        super().__init__(optimizer, scale, num_replicas, patch_optimizer)
        self._optimizer.state.setdefault("gradient_noise_scale", {
            "replicas": self._num_replicas,

            # Averages of n and v
            "norm_avg": np.ones(len(optimizer.param_groups)),
            "var_avg": np.zeros(len(optimizer.param_groups)),
        })

        self._local_sqr = None
        self._prev_grads = None
        self._smoothing = 0.999

        for idx, param_group in enumerate(self._optimizer.param_groups):
            for param in param_group["params"]:
                param.register_hook(
                    functools.partial(self._backward_hook, idx))

    @property
    def _state(self):
        return self._optimizer.state["gradient_noise_scale"]

    def set_accumulation_steps(self, accumulation_steps):
        """
        Override _ScalingRuleBase.set_accumulation_steps to reset `_local_sqr`.
        """
        accumulation_steps = accumulation_steps + 1
        if accumulation_steps != self._accumulation_steps:
            self._current_accumulation_step = 0
            self._accumulation_steps = int(accumulation_steps)
            self._local_sqr = None

    def set_scale(self, scale=None):

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

    def reset_avg_if_necessary(self):
        if self._state['replicas'] == 1 and self._num_replicas > 1:
            # TODO: when to reset running averages should be decided outside of
            #       the scaling rule object.
            self._reset_avg("norm_avg")
            self._reset_avg("var_avg")
        self._state['replicas'] = self._num_replicas

    def _backward_hook(self, idx, grad):
        # This method should be invoked once for each parameter during the
        # backward pass, before gradients are synchronized between replicas.
        if self._local_sqr is None:
            self._local_sqr = torch.zeros(
                (self._accumulation_steps,
                 len(self._optimizer.param_groups)),
                device=grad.device)
        self._local_sqr[self._current_accumulation_step][idx] \
            += grad.pow(2).sum()
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
        if self.is_accumulation_step() or self._final_callback_queued:
            return
        self._final_callback_queued = True
        Variable._execution_engine.queue_callback(self._final_callback)
        # Asynchronously sum the local squared-gradient statistics. The actual
        # gradient averaging should also be happening at the same time, until
        # self._final_callback is invoked.
        if self._num_replicas > 1:
            self._local_sqr = (self._local_sqr,
                               torch.distributed.all_reduce(self._local_sqr,
                                                            async_op=True))

    def _final_callback(self):
        # This method should be invoked once the gradients have been
        # synchronized between all replicas and accumulation steps.
        self._final_callback_queued = False

        grads = [[p.grad.detach() for p in group["params"]]
                 for group in self._optimizer.param_groups]

        # Average local squared-norm samples across replicas.
        if self._num_replicas > 1:
            self._local_sqr[1].wait()
            self._local_sqr = self._local_sqr[0] / self._num_replicas
        count = self._num_replicas * self._accumulation_steps
        scale = self.scale
        if count > 1:
            # Average local squared-norm samples across accumulation steps.
            local_sqr = self._local_sqr.cpu().numpy().mean(axis=0)
            total_sqr = _normsqr_groups(grads)
            self._prev_grads = None
        # Single gradient datapoint, use difference estimation.
        else:
            if self._prev_grads is not None:
                local_sqr = (_normsqr_groups(self._prev_grads) +
                             _normsqr_groups(grads)) / 2
                total_sqr = \
                    _normsqr_groups(_average_groups(grads, self._prev_grads))
                count = 2
                scale = 2 * self.scale
            self._prev_grads = [[g.clone() if g is not None else None
                                 for g in group] for group in grads]

        if count > 1:
            grad_sqr = (count * total_sqr - local_sqr) / (count - 1)
            grad_var = (local_sqr - total_sqr) * scale / (count - 1)
            grad_sqr = np.maximum(grad_sqr, 0.0)
            grad_var = np.maximum(grad_var, 1e-6)
            theta = self._smoothing ** scale
            self._update_avg('norm_avg', grad_sqr, theta)
            self._update_avg('var_avg', grad_var, theta)
        self._local_sqr = None


class TBDBase(_GradientNoiseScaleMixin, _ScalingRuleBase):
    """Base class for scaling rules that brings in gradient
    noise scale calculations."""
    subclasses = []

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.subclasses.append(cls)

    def get_progress(self, scale=None):
        if self._made_step:
            return self.gain(scale)
        else:
            return 0.0

    def _calculate_scale_factors(self):
        raise NotImplementedError


class AdaScale(TBDBase):
    """
    Implements the AdaScale_ algorithm for scaling the learning rate for
    distributed and large batch size training. Can be used in combination with
    ``torch.nn.parallel.DistributedDataParallel`` and ``torch.optim.SGD``.

    .. code-block:: python

        optim = torch.optim.SGD(model, lr=0.001)
        model = DistributedDataParallel(model, optim, scaling_rule="AdaScale")
        adascale = model.scaling_rule

        for epoch in ...:
            for batch in ...:
                optim.zero_grad()
                loss = ...
                loss.backward()
                adascale.step()

    .. _AdaScale: https://proceedings.icml.cc/static/paper_files/icml/2020/4682-Supplemental.pdf
    """  # noqa: E501

    def _calculate_scale_factors(self) -> float:
        grad_sqr = self._state["norm_avg"]
        grad_var = self._state["var_avg"]
        gain = (grad_var + grad_sqr) / (grad_var / self._scale + grad_sqr)
        return gain

    def to_tensorboard(self, writer, global_step, tag_prefix=""):
        writer.add_scalar(tag_prefix + "Learning_Rate_Factor",
                          self.gain(), global_step)


class LinearScale(TBDBase):

    def _calculate_scale_factors(self) -> float:
        return self.scale * np.ones(len(self._optimizer.param_groups))

    def to_tensorboard(self, writer, global_step, tag_prefix=""):
        writer.add_scalar(tag_prefix + "Learning_Rate_Factor",
                          self.scale, global_step)


class SqrtScale(TBDBase):

    def _calculate_scale_factors(self, pg_index=None) -> float:
        return math.sqrt(self.scale) \
               * np.ones(len(self._optimizer.param_groups))

    def to_tensorboard(self, writer, global_step, tag_prefix=""):
        writer.add_scalar(tag_prefix + "Learning_Rate_Factor",
                          math.sqrt(self.scale), global_step)
