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
import time
import warnings
from typing import Optional

import torch
import torch.cuda
import torch.distributed
from torch.autograd import Variable
from torch.nn.parallel import DistributedDataParallel

import adaptdl.checkpoint
import adaptdl.env
from adaptdl.torch.data import current_dataloader
from adaptdl.torch.scaling_rules import AdaScale, ScalingRuleBase
from adaptdl.torch.gradient_noise_scale import GradientNoiseScale
from adaptdl.torch._metrics import profile_sync_time, update_grad_params,\
    update_progress


class AdaptiveDataParallel(DistributedDataParallel):
    """
    This class extends PyTorch DistributedDataParallel with support for
    adaptive batch sizes and checkpoint-restart elasticity. It automatically
    saves the given model, optimizer, and (optionally) LR scheduler whenever a
    checkpoint is triggered, and restores their states after restart. The
    optimizer is automatically patched with the chosen scaling rule.

    Arguments:
        model (torch.nn.Module): Model to be distributed.
        optimizer (torch.optim.Optimizer): Optimizer used to update the given
        model's parameters, will be patched using subclass of
        :class:`adaptdl.torch.scaling_rules.ScalingRuleBase`.
        scaling_rule (ScalingRuleBase): Scaling rule used to
        patch the given optimizer, default to AdaScale.
        lr_scheduler (torch.optim.lr_scheduler._LRScheduler): LR scheduler used
        to anneal the learning rate for the given optimizer.
        name (string): Unique name for each instance of this class, needed only
        if multiple instances exist.
    """
    def __init__(self, model, optimizer, lr_scheduler=None, mp_scaler=None,
                 scaling_rule: Optional[ScalingRuleBase] = None,
                 name="adaptdl-dataparallel", **kwargs):
        super().__init__(model, **kwargs)
        self._key = id(self)
        # Register backward hooks on model parameters. Depends on these hooks
        # being invoked before gradients are averaged. This is technically an
        # internal behavior of DistributedDataParallel, but seems to be abused
        # pretty widely so there should be little chance of it changing.
        # https://discuss.pytorch.org/t/59291
        for param in model.parameters():
            param.register_hook(functools.partial(self._backward_hook, param))

        # Setup for the scaling_rule, must be after registering backward hooks
        # because some of them need to register their own backward hooks.
        self.gns = GradientNoiseScale(self, optimizer, mp_scaler=mp_scaler)
        self.scaling_rule = scaling_rule or AdaScale()
        self.scaling_rule.initialize(self, optimizer, patch_optimizer=True)

        self._state = _AdaptiveDataParallelState(
            model, optimizer, lr_scheduler, mp_scaler, name)
        adaptdl.checkpoint.load_state(self._state)

        self._sync_start = None

    def forward(self, *args, **kwargs):
        # Do not do gradient synchronization during gradient accumulation.
        dataloader = current_dataloader()
        if dataloader is not None and dataloader.training:
            self.require_backward_grad_sync = dataloader.is_optim_step()
            accum_scale = (dataloader.current_local_bsz *
                           adaptdl.env.num_replicas() / dataloader.batch_size)
            self.gns.set_accum_scale(accum_scale)
        return super().forward(*args, **kwargs)

    def _backward_hook(self, param, grad):
        # This method should be invoked once for each parameter during the
        # backward pass, before gradients are synchronized between replicas.
        if grad.device.type.startswith("cuda"):
            self._sync_start = torch.cuda.Event(enable_timing=True)
            self._sync_start.record()
        else:
            self._sync_start = time.time()
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
        # self._sync_start should mark the last time any local gradient
        # from this module was produced. We assume the duration until now was
        # primarily spent waiting for gradient synchronization.
        # TODO: Depends on the internal behavior of DistributedDataParallel,
        #       which might break with future versions of PyTorch. Any better
        #       and well-supported way to measure the synchronization time?
        if isinstance(self._sync_start, torch.cuda.Event):
            sync_end = torch.cuda.Event(enable_timing=True)
            sync_end.record()
            sync_end.synchronize()
            profile_sync_time(self._sync_start.elapsed_time(sync_end) / 1e3)
        else:
            profile_sync_time(time.time() - self._sync_start)

        dataloader = current_dataloader()
        if dataloader is None:
            # Don't allow backpropagation outside of a dataloader loop, because
            # the batch size would be unknown.
            raise RuntimeError("backpropagation outside AdaptiveDataLoader")
        dataloader.train()

        scale = dataloader.current_batch_size / dataloader.batch_size
        self._state.gain = self.gns.gain(scale)
        self._state.lr_factor = \
            np.average(self.scaling_rule.scale_lr(scale))
        update_progress(self.gns.get_progress())
        if dataloader.max_batch_size and \
                dataloader.max_batch_size > dataloader.batch_size:
            update_grad_params(self._key, self.gns.sqr_avg(),
                               self.gns.var_avg())
        self._sync_start = None

    def zero_grad(self, *args, **kwargs):
        warnings.warn("zero_grad has no effect with AdaptiveDataParallel")

    @property
    def gain(self):  # TODO: should be tracked in the metrics module instead.
        """
        Current estimate of the AdaScale gain (r_t) value.
        """
        return self._state.gain

    def to_tensorboard(self, writer, global_step, tag_prefix=""):
        """
        Output some useful metrics to TensorBoard.

        Arguments:
            writer (torch.utils.tensorboard.SummaryWriter): ``SummaryWriter``
                object to output metrics to.
            global_step (int): Global step value to record.
            tag_prefix (str): Prefix added to each metric's tag.
        """
        if tag_prefix and not tag_prefix.endswith("/"):
            tag_prefix += "/"
        writer.add_scalar(tag_prefix + "Gradient_Norm_Sqr",
                          self.gns.sqr_avg(), global_step)
        writer.add_scalar(tag_prefix + "Gradient_Variance",
                          self.gns.var_avg(), global_step)
        writer.add_scalar(tag_prefix + "Gain",
                          self._state.gain, global_step)
        writer.add_scalar(tag_prefix + "Learning_Rate_Factor",
                          self._state.lr_factor, global_step)


class _AdaptiveDataParallelState(adaptdl.checkpoint.State):
    def __init__(self, model, optimizer, lr_scheduler, mp_scaler,
                 name="adaptdl-dataparallel"):
        super().__init__(name)
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.mp_scaler = mp_scaler
        # TODO: Gain/goodput should be tracked in the metrics module instead.
        self.gain = 1.0
        # lr_factor summary
        self.lr_factor = 1.0

    def save(self, fileobj):
        state_dicts = [self.model.state_dict(), self.optimizer.state_dict()]

        if self.lr_scheduler is not None:
            state_dicts.append(self.lr_scheduler.state_dict())
        else:
            state_dicts.append(None)

        if self.mp_scaler is not None:
            state_dicts.append(self.mp_scaler.state_dict())
        else:
            state_dicts.append(None)
        torch.save((state_dicts, self.gain, self.lr_factor), fileobj)

    def load(self, fileobj):
        state_dicts, self.gain, self.lr_factor = torch.load(fileobj)
        self.model.load_state_dict(state_dicts[0])
        self.optimizer.load_state_dict(state_dicts[1])
        if state_dicts[2] is not None:
            self.lr_scheduler.load_state_dict(state_dicts[2])
        if state_dicts[3] is not None:
            self.mp_scaler.load_state_dict(state_dicts[3])
