Standalone Training
===================

This tutorial shows how to run AdaptDL training code in a standalone setting,
outside of an AdaptDL-scheduled cluster. Standalone training has no dependency
on deploying Kubernetes or the AdaptDL scheduler. It can be useful for:

1.  Distributed training with adaptive batch sizes in a dedicated cluster.
2.  Local testing the training code before submitting to an AdaptDL cluster.


Local Training
--------------

Any training code that uses AdaptDL APIs can be run locally as a single
process. All that's needed is to install the ``adaptdl`` package, and run the
code as a regular python program.

.. code-block:: shell

   $ python3 -m pip install adaptdl

As an example, we shall run the simple MNIST training script
(:download:`mnist_step_5.py <tutorial/mnist_step_5.py>`).

.. code-block:: shell

   $ python3 mnist.py

Output:

.. code-block::

   WARNING:adaptdl.reducer:Could not connect to root, trying again...
   INFO:adaptdl.reducer:Master waiting for connections on 0
   INFO:adaptdl.reducer:rank 0 connecting to 0.0.0.0 on port 36405
   INFO:adaptdl.torch:Initializing torch.distributed using tcp://0.0.0.0:39345?rank=0&world_size=1
   INFO:adaptdl.torch:torch.distributed initialized
   INFO:adaptdl.torch.epoch:starting at epoch 0
   Train Epoch: 0 [0/60000 (0%)]   Loss: 2.318445
   Train Epoch: 0 [640/60000 (1%)] Loss: 1.647522
   ...
   ...
   ...
   Train Epoch: 13 [58880/60000 (98%)]  Loss: 0.003577
   Train Epoch: 13 [59520/60000 (99%)]  Loss: 0.034688

   Test set: Average loss: 0.0267, Accuracy: 9911/10000 (99%)


Manual Checkpoint-Restart
-------------------------

When a training program is running locally, a checkpoint can be triggered by
sending an interrupt (CTRL-C in most terminals). The environment variable
``ADAPTDL_CHECKPOINT_PATH`` specifies where the checkpoint should be located.

.. code-block:: shell

   $ mkdir mnist-checkpoint
   $ ADAPTDL_CHECKPOINT_PATH=mnist-checkpoint python3 mnist.py

Output (after sending CTRL-C during training):

.. code-block::

   WARNING:adaptdl.reducer:Could not connect to root, trying again...
   INFO:adaptdl.reducer:Master waiting for connections on 0
   INFO:adaptdl.reducer:rank 0 connecting to 0.0.0.0 on port 51067
   INFO:adaptdl.torch:Initializing torch.distributed using tcp://0.0.0.0:24997?rank=0&world_size=1
   INFO:adaptdl.torch:torch.distributed initialized
   INFO:adaptdl.torch.epoch:starting at epoch 0
   Train Epoch: 0 [0/60000 (0%)]    Loss: 2.318445
   Train Epoch: 0 [640/60000 (1%)]  Loss: 1.647522
   ...
   ...
   ...
   Train Epoch: 7 [30080/60000 (50%)]      Loss: 0.009690
   Train Epoch: 7 [30720/60000 (51%)]      Loss: 0.010559
   ^CINFO:adaptdl._signal:Got SIGINT, exiting gracefully... Send signal again to force exit.
   INFO:adaptdl._signal:Got SIGINT, exiting gracefully... Send signal again to force exit.

Training can be resumed by running the script with the same checkpoint path.

.. code-block:: shell

   $ ADAPTDL_CHECKPOINT_PATH=mnist-checkpoint python3 mnist.py

Output:

.. code-block::

   WARNING:adaptdl.reducer:Could not connect to root, trying again...
   INFO:adaptdl.reducer:Master waiting for connections on 0
   INFO:adaptdl.reducer:rank 0 connecting to 0.0.0.0 on port 45371
   INFO:adaptdl.torch:Initializing torch.distributed using tcp://0.0.0.0:23678?rank=0&world_size=1
   INFO:adaptdl.torch:torch.distributed initialized
   INFO:adaptdl.torch.epoch:starting at epoch 7
   Train Epoch: 7 [0/60000 (0%)]   Loss: 0.070648
   Train Epoch: 7 [640/60000 (2%)] Loss: 0.068212
   ...
   ...
   ...
   Train Epoch: 13 [58880/60000 (98%)]     Loss: 0.081517
   Train Epoch: 13 [59520/60000 (99%)]     Loss: 0.006973

   Test set: Average loss: 0.0281, Accuracy: 9913/10000 (99%)

Whenever possible, it's recommended to test the training code locally in this
way before submitting it to an AdaptDL-scheduled cluster.


Distributed Training
--------------------

Training code that uses AdaptDL APIs can also be run on a distributed cluster,
without requiring the AdaptDL scheduler. In this setting, the training job will
run using the same number of replicas until it finishes, or until a checkpoint
is manually triggered. Although the number of replicas is fixed, standalone
distributed training can still benefit from the automatic batch size and
learning rate scaling offered by AdaptDL.

The following environment variables need to be set for every replica:

- ``ADAPTDL_MASTER_ADDR``: network address of the node running the rank 0
  replica, must be accessible from all other replicas.
- ``ADAPTDL_MASTER_PORT``: available port on the node running the rank 0
  replica, must be accessible from all other replicas.
- ``ADAPTDL_NUM_REPLICAS``: total number of replicas.
- ``ADAPTDL_REPLICA_RANK``: integer rank from 0 .. K-1 for each replica, where
  K is the total number of replicas.

Assuming two nodes with hostnames ``node-0`` and ``node-1``, on ``node-0``:

.. code-block:: shell

   $ ADAPTDL_MASTER_ADDR=node-0 ADAPTDL_MASTER_PORT=47000 \
     ADAPTDL_NUM_REPLICAS=2 ADAPTDL_REPLICA_RANK=0 python3 mnist.py

And on ``node-1``:

.. code-block:: shell

   $ ADAPTDL_MASTER_ADDR=node-0 ADAPTDL_MASTER_PORT=47000 \
     ADAPTDL_NUM_REPLICAS=2 ADAPTDL_REPLICA_RANK=1 python3 mnist.py

A checkpoint can be triggered by sending an interrupt to any of the replicas.
The replica with rank 0 will save the checkpoint to the path specified by the
``ADAPTDL_CHECKPOINT_PATH`` environment variable, and then all replicas will
exit.

Training can be resumed from the checkpoint using any number of replicas.
However, each replica will need to be able to access the saved checkpoint. This
means the checkpoint should be saved to a shared distributed filesystem such as
NFS, or be manually copied to each node before resuming training.
