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
import math
import numpy as np
import warnings

from types import MethodType

from adaptdl.torch.data import current_dataloader


__all__ = ["ScalingRuleBase", "AdaScale", "LinearScale", "SqrtScale",
           "LEGWScale"]


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
        self._optimizer = None
        self._orig_optimizer_step = None

    def scale_lr(self, scale):
        raise NotImplementedError

    def zero_grad(self, *args, **kwargs):
        if self.adp.gns.should_zero_grad:
            self.adp.gns.reset_accumulation(*args, **kwargs)
        else:
            warnings.warn("skipping zero_grad for accumulated gradient")

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
        scale = self.adp.gns.accum_scale * self.adp.gns.accum_count
        initial_lr = [pg["lr"] for pg in self._optimizer.param_groups]
        scaled_lr = np.multiply(self.scale_lr(scale), initial_lr)
        for lr, pg in zip(scaled_lr, self._optimizer.param_groups):
            pg["lr"] = lr
        self._orig_optimizer_step(*args, **kwargs)
        for lr, pg in zip(initial_lr, self._optimizer.param_groups):
            pg["lr"] = lr
        self.adp.gns.set_progress(self.adp.gns.get_progress()
                                  + self.adp.gns.gain(scale))

    def _patch_optimizer(self):
        """
        Monkey-patch the optimizer's step function with
        :meth:`ScalingRuleBase.step`.
        """
        @functools.wraps(self._optimizer.step)
        def step_wrapper(optim, *args, **kwargs):
            return self.step(*args, **kwargs)

        @functools.wraps(self._optimizer.zero_grad)
        def zero_wrapper(optim, *args, **kwargs):
            return self.zero_grad(*args, **kwargs)
        self._optimizer.step = MethodType(step_wrapper, self._optimizer)
        self._optimizer.zero_grad = MethodType(zero_wrapper, self._optimizer)

    def initialize(self, adp, optimizer, patch_optimizer=False):
        self.adp = adp
        self._optimizer = optimizer
        self._orig_optimizer_step = optimizer.step
        if patch_optimizer:
            self._patch_optimizer()


class AdaScale(ScalingRuleBase):
    """
    Implements the AdaScale_ algorithm for scaling the learning rate for
    distributed and large batch size training.

    .. _AdaScale: https://proceedings.icml.cc/static/paper_files/icml/2020/4682-Supplemental.pdf
    """  # noqa: E501

    def scale_lr(self, scale):
        """Calculate factors to be applied to lr for each parameter group."""
        var = self.adp.gns.raw_var_avg
        sqr = self.adp.gns.raw_sqr_avg
        var = np.maximum(var, 1e-6)
        sqr = np.maximum(sqr,  0.0)
        return (var + sqr) / (var / scale + sqr)


class LinearScale(ScalingRuleBase):

    def scale_lr(self, scale):
        return scale


class SqrtScale(ScalingRuleBase):

    def scale_lr(self, scale):
        return math.sqrt(scale)


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

    def scale_lr(self, scale):
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
