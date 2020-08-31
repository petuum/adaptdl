
Usage
=====

Enabling elastic batchsize
--------------------------

To enable elastic batchsize, provide a non-\ ``None`` bounds for either ``max_batch_size`` or ``local_bsz_bounds``

Batchsize bounds parameters
---------------------------


* 
  ``max_batch_size: Int`` is a (positive) integer that provides an upper bound on the global batchsize across all replicas. For example, if ``max_batch_size`` is ``80``\ , and the number of replicas is ``8``\ , ``10`` would be a valid local batchsize on the replicas, but ``11`` would not be. Generally ``max_batch_size`` should be dependent on the model and training parameters.

* 
  ``local_bsz_bounds: (lower_bound: Int, upper_bound: Int)`` are bounds on the local batchsize. ``lower_bound`` is the minimum batchsize allowed on any node, and ``upper_bound`` is the maximum batchsize allowed on any node. For obvious reasons, ``lower_bound`` must not be greater than ``upper_bound``. Generally, ``local_bsz_bounds`` should depend on the model size and local memory resources. GPU memory should be the primary limit on ``upper_bound``.

Note that these bounds may interact. The maximum local batchsize is limited by both ``max_batch_size`` / ``replicas`` and upper bound. Certain number of replicas may be invalid if ``max_batch_size`` / ``replicas`` becomes smaller than ``local_bsz_bounds``. For example, if ``lower_bound`` is ``100``\ , and ``max_batch_size`` is ``1000``\ , the job will not be scheduleable on ``11`` replicas. 

Leaving one of ``max_batch_size`` or ``local_bsz_bounds`` as ``None`` will cause the batchsize to only be bounded by the other. It is advisable that if ``local_bsz_bounds`` is ``None``\ , that a batchsize of ``max_batch_size`` can fit on a few replicas.

In all cases, the batchsize will be identical for all replicas.

Using the elastic batchsize
---------------------------

To get the elastic batchsize at each epoch, call ``adaptdlTrainer.get_local_bsz()``. Please note that this function may return a different batchsize as training progresses due to $\mathtt{gain}$ changing (See the description of AdascaleSGD for more details). The canonical use-case is to use the value from a new call to ``adaptdlTrainer.get_local_bsz()`` at the start of each epoch. Currently, there is no safeguard in place to prevent stale batchsizes from being used.

To ensure that you are using up-to-date batchsizes, ensure that your code resembles

.. code-block::

   for epoch in epochs:
       batchsize = adaptdlTrainer.get_local_bsz()
       ... Training for 1 epoch using batchsize...
       adaptdlTrainer.step(epoch, ...)

and **NOT**

.. code-block::

   # DO NOT DO THIS
   batchsize = adaptdlTrainer.get_local_bsz()
   for epoch in epochs:
       ... Training for 1 batch using batchsize...
       adaptdlTrainer.step(epoch, ...)

In the second case, the batchsize will only be updated when the job is rescheduled to a different allocation. 

Interoperability/Restrictions/Limititations
-------------------------------------------

Elastic batchsize is enabled by monkeypatching the provided optimizer. This allows that the passed-in optimizer be used as expected, with up-to-date parameters, etc. Enabling elastic batchsize will overwrite the optimizer's step function. It will scale the learning rate of the optimizer directly, using the following pattern in the optimizer object, where ``base_step`` is the original step function:

.. code-block::

   def step(self, ...):
       initial_lr = [group['lr'] for group in self.param_groups]
       for (g, group) in zip(gain, self.param_groups):
           group['lr'] *= g
       base_step(*args, **kwargs)
       for (lr, group) in zip(initial_lr, self.param_groups):
           group['lr'] = lr

This allows for interoperability between elastic batchsize and standard learning rate schedulers, including both changing the learning rate directly and the schedulers in ``torch.optim.lr_scheduler``. For example, the following is valid code for enabling a multistep scheduler with elastic batchsize scaling:

.. code-block::

   #non-None max_batch_size, local_bsz_bounds
   adaptdlTrainer = AdaptDLTorchTrainer(args.bs, max_batch_size, local_bsz_bounds)
   optimizer = optim.SGD(model.parameters(), lr=args.lr)
   lr_scheduler = MultiStepLR(optimizer, [30,45], 0.1)
   adaptdl_state = adaptdlTrainer.program_reentry(model, optimizer, lr_scheduler)

Step is the only function overwritten, but other functions, including ``predict_gain(scale)`` and ``update_scale(scale, replicas)`` are added (where ``scale`` is the ratio of the new global batchsize to the initial batchsize).

Currently, only optimizers of the class ``torch.optim.SGD`` are compatible with elastic optimization.

Because the learning rate can be increased by as much as $\frac{\mathtt{max_batch_size}}{\mathtt{initial_batch_size}}$, compatability with functionality that requires a hard limit on the step size, such as gradient clipping, is not supported. If this is a blocker, please let us know.

Other usage notes:
------------------


* Elastic Batchsize is not applied when the number of current replicas is 1. In this case, the default batchsize is used. However, gain is still computed to know when and how to rescale.
* Instability may occur when too large of a batchsize is used.
* Local batchsize will be the same accross all replicas
* Expect the gain to start out near $1$ and to increase as the model converges.
* When observing the raw gain values, remember that it depends heavily on the scale. It may be useful to look at the ratio of $\frac{\mathtt{gain}}{\mathtt{scale}}$
