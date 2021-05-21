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
import warnings

import math
import numpy as np
import os
import torch.distributed
import torch.optim
from torch.autograd import Variable
from types import MethodType


__all__ = ["AdaScale"]


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


def _normsqr_groups(grads, pinvs=None):
    ret = []
    if pinvs is None:
        for group in grads:
            normsqr = [g.pow(2).sum(dtype=torch.float64) for g in group if g is not None]
            ret.append(sum(normsqr).item() if normsqr else 0.0)
    else:
        assert len(pinvs) > 0
        for group, pinv_group in zip(grads, pinvs):
            normsqr = [(g / pinv).pow(2).sum(dtype=torch.float64) for g, pinv in \
                       zip(group, pinv_group) if g is not None]
            ret.append(sum(normsqr).item() if normsqr else 0.0)
    return np.array(ret)


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
    def __init__(self, ddp_model, optimizer, num_replicas=None,
                 batch_scale=None, patch_optimizer=False):
        self._ddp_model = ddp_model
        self._optimizer = optimizer
        self._optimizer_step = None
        self._optimizer_zero_grad = None
        self._num_replicas = (num_replicas if num_replicas is not None
                              else torch.distributed.get_world_size())
        self._batch_scale = batch_scale or self._num_replicas
        self._prev_grads = None
        self._is_adam = optimizer.__class__.__name__ in ['Adam', 'AdamW']
        self._betas = []
        self._epss = []

        self._optimizer.state.setdefault("adascale", {
            "progress": 0.0,
            "prev_scale": 0.0,
            # Averages of n and v
            "sqr_avg": np.ones(len(optimizer.param_groups)),
            "var_avg": np.zeros(len(optimizer.param_groups)),
            # Whether estimates are biased (using differenced estimator).
            "biased": False,
        })

        for idx, param_group in enumerate(self._optimizer.param_groups):
            if self._is_adam:
                self._betas.append(param_group['betas'][1])
                self._epss.append(param_group['eps'])
            for param in param_group["params"]:
                param.register_hook(
                    functools.partial(self._backward_hook, idx, param))
        self._callback_queued = False

        if patch_optimizer:
            self.patch_optimizer()

        self._smoothing = 0.999
        self.loss_scale = 1.0

        self.raw_grad_sqr = np.ones(len(optimizer.param_groups))
        self.raw_grad_var = np.zeros(len(optimizer.param_groups))

        self._reset_accumulation()

    def _reset_accumulation(self):
        if self._optimizer_zero_grad is None:
            self._optimizer.zero_grad()
        else:
            self._optimizer_zero_grad()
        self._local_sqr = None
        self._accum_count = 0

    def _reset_adam_state(self, step=0):
        for group in self._optimizer.param_groups:
            beta1, beta2 = group["betas"]
            for param in group["params"]:
                state = self._optimizer.state[param]
                if state.get("step", 0) > 0:
                    state["exp_avg"].mul_(
                        (1 - beta1 ** step) / (1 - beta1 ** state["step"]))
                    state["exp_avg_sq"].mul_(
                        (1 - beta2 ** step) / (1 - beta2 ** state["step"]))
                    state["step"] = step

    @property
    def _state(self):
        return self._optimizer.state["adascale"]

    @property
    def batch_scale(self):
        return self._batch_scale

    def set_batch_scale(self, batch_scale):
        if not np.isclose(self._batch_scale, batch_scale):
            self._reset_accumulation()
            self._batch_scale = batch_scale

    def sqr_avg(self):
        """
        Current estimate of the squared l2-norm of the true gradient (sigma
        squared).

        Returns (float): Estimate of squared l2-norm.
        """
        return float(np.sum(np.maximum(self._state["sqr_avg"], 0.0)))

    def var_avg(self):
        """
        Current estimate of the trace of the covariance of the true gradient
        (mu squared).

        Returns (float): Estimate of trace of the covariance.
        """
        return float(np.sum(np.maximum(self._state["var_avg"], 1e-6)))

    def get_progress(self):
        return self._state["progress"]

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

    def _backward_hook(self, idx, param, grad):
        # This method should be invoked once for each parameter during the
        # backward pass, before gradients are synchronized between replicas.
        if self._local_sqr is None:
            self._local_sqr = torch.zeros(len(self._optimizer.param_groups),
                                          device=grad.device, dtype=torch.float64)

        # Get the preconditioning matrix for the optimizer
        pinv = 1
        state = self._optimizer.state[param]
        if self._is_adam and state.get("step", 0) > 0:
            exp_avg_sq = state["exp_avg_sq"].clone()
            beta2 = self._betas[idx]
            eps = self._epss[idx]
            correction = 1 - beta2 ** state['step']
            pinv = (exp_avg_sq.sqrt() / math.sqrt(correction)).add_(eps)
            pinv = pinv if state["step"] > 5 else 1

        # Update the local gradient square sum
        self._local_sqr[idx] += (grad.float() / (pinv * self.loss_scale)).pow(2).sum(dtype=torch.float64)
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
        if self._ddp_model.require_backward_grad_sync:
            # Asynchronously sum the local squared-gradient statistics. The actual
            # gradient averaging should also be happening at the same time, until
            # self._final_callback is invoked.
            self._async_op = torch.distributed.all_reduce(self._local_sqr,
                                                          async_op=True)
            Variable._execution_engine.queue_callback(self._final_callback)

    def _final_callback(self):
        # This method should be invoked once the gradients have been
        # synchronized between all replicas and accumulation steps.
        self._async_op.wait()

        grads = []
        for group in self._optimizer.param_groups:
            grads.append([])
            for param in group["params"]:
                if param.grad is None:
                    grads[-1].append(None)
                    continue
                param.grad.div_(self._accum_count)
                grads[-1].append(param.grad.detach().float() / self.loss_scale)

        pinvs = None
        if self._is_adam:
            pinvs = self.get_pinvs()

        check = [g.sum() for group in grads for g in group]
        if any(c != c or c in (float('inf'), -float('inf')) for c in check):
            print("AdaScale detected invalid gradient! Skipping step.")
            return

        count = self._num_replicas * self._accum_count
        scale = self._batch_scale * self._accum_count
        if count > 1:
            # Average local squared-norm samples.
            local_sqr = self._local_sqr.cpu().numpy() / count
            total_sqr = _normsqr_groups(grads, pinvs)
            if self._state["biased"]:
                self._reset_avg("sqr_avg")
                self._reset_avg("var_avg")
            self._state["biased"] = False
            self._prev_grads = None
        else:
            # Single gradient datapoint, use difference estimation.
            if self._prev_grads is not None:
                local_sqr = (_normsqr_groups(self._prev_grads, pinvs) +
                             _normsqr_groups(grads, pinvs)) / 2
                total_sqr = _normsqr_groups(
                    _average_groups(grads, self._prev_grads), pinvs
                )
                count = 2
                scale = 2 * self._batch_scale
            self._state["biased"] = True
            self._prev_grads = [[g.clone() if g is not None else None
                                 for g in group] for group in grads]

        if count > 1:
            grad_sqr = (count * total_sqr - local_sqr) / (count - 1)
            grad_var = (local_sqr - total_sqr) * scale / (count - 1)
            self.raw_grad_sqr = grad_sqr
            self.raw_grad_var = grad_var
            #grad_sqr = np.maximum(grad_sqr, 0.0)
            #grad_var = np.maximum(grad_var, 1e-6)
            theta = self._smoothing ** scale
            self._update_avg('sqr_avg', grad_sqr, theta)
            self._update_avg('var_avg', grad_var, theta)

    def step(self, *args, **kwargs):
        """
        Run one optimizer step using Adascale. Essentially just invokes
        ``optimizer.step(*args, **kwargs)`` with a scaled learning rate.

        Arguments:
            args: Positional arguments passed to ``optimizer.step``.
            kwargs: Keyword arguments passed to ``optimizer.step``.
        """
        if not self._ddp_model.require_backward_grad_sync:
            return
        scale = self._batch_scale * self._accum_count
        if self._is_adam and not np.isclose(scale, self._state["prev_scale"]):
            self._reset_adam_state()
            self._state["prev_scale"] = scale
        initial_lr = [pg["lr"] for pg in self._optimizer.param_groups]
        for sqr, var, pg in zip(self._state["sqr_avg"], self._state["var_avg"],
                                self._optimizer.param_groups):
            sqr, var = max(float(sqr), 0.0), max(float(var), 1e-6)
            factor = scale ** 0.5 if self._is_adam \
                                  else (var + sqr) / (var / scale + sqr)
            pg["lr"] = factor * pg["lr"]
            if "TRACE_THROUGHPUT" in os.environ:
                # Don't take actual step when tracing throughput.
                pg["lr"] = 0.0
        if self._optimizer_step is None:
            self._optimizer.step(*args, **kwargs)
        else:
            self._optimizer_step(*args, **kwargs)
        for lr, pg in zip(initial_lr, self._optimizer.param_groups):
            pg["lr"] = lr
        self._state["progress"] += self.gain(scale)
        self._reset_accumulation()

    def zero_grad(self, *args, **kwargs):
        warnings.warn("zero_grad has no effect with AdaScale")

    def patch_optimizer(self):
        """
        Monkey-patch the optimizer's step function with :meth:`AdaScale.step`.
        """
        # TODO: detect if the optimizer has already been patched.

        @functools.wraps(self._optimizer.step)
        def step_wrapper(optim, *args, **kwargs):
            return self.step(*args, **kwargs)

        @functools.wraps(self._optimizer.zero_grad)
        def zero_wrapper(optim, *args, **kwargs):
            return self.zero_grad(*args, **kwargs)
        self._optimizer_step = self._optimizer.step
        self._optimizer.step = MethodType(step_wrapper, self._optimizer)
        self._optimizer_zero_grad = self._optimizer.zero_grad
        self._optimizer.zero_grad = MethodType(zero_wrapper, self._optimizer)

    def get_pinvs(self):
        state = self._optimizer.state
        out = []

        for group in self._optimizer.param_groups:
            pinvs = []
            for p in group["params"]:
                step = state[p].get('step', 0)
                if step < 1:
                    return None
                elif step > 5:
                    exp_avg_sq = state[p]['exp_avg_sq'].clone()
                    bias_correction = 1 - group['betas'][1] ** state[p]['step']
                    pinv = exp_avg_sq.sqrt() / math.sqrt(bias_correction)
                    pinvs.append(pinv.add_(group['eps']))
                else:
                    pinvs.append(1)
            out.append(pinvs)
        return out
