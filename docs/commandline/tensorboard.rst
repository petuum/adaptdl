Integrating your AdaptDL job with TensorBoard
=============================================

`Tensorboard <https://www.tensorflow.org/tensorboard>`__ provides a simple way to collect and visualize model performance, statistics, and weights. AdaptDL provides integration with tensorboard across replicas via AdaptDL's command line interface. 

AdaptDL provides a way to deploy a Tensorboard instance on your kubernetes cluster that your AdaptDL jobs can interact with. This tutorial demonstrates how to have your AdaptDL jobs write to Tensorboard and how to access the Tensorboard UI.

Modifying Your Code
-------------------

The AdaptDL CLI provides the environment variable
``ADAPTDL_TENSORBOARD_LOGDIR`` as the log directory for AdaptDL
TensorBoard deployments. Use
``$ADAPTDL_TENSORBOARD_LOGDIR/<job-name>`` for the particular job you
are running via the following code:

::

   os.path.join(os.getenv("ADAPTDL_TENSORBOARD_LOGDIR"),
                          adaptdl.env.get_job_id())

Following :doc:`AdaptDL with PyTorch <../adaptdl-pytorch>`, the following changes are made to the ``test`` function
to write to tensorboard:

::

    with stats.synchronized():
        test_loss = stats["test_loss"] / len(test_loader.dataset)
        correct = stats["correct"]
        tensorboard_dir = os.path.join(os.getenv("ADAPTDL_TENSORBOARD_LOGDIR", "/tmp"),
                                       adaptdl.env.get_job_id())
        with SummaryWriter(tensorboard_dir) as writer:
            writer.add_scalar("Test/Loss", test_loss, epoch)
            writer.add_scalar("Test/Accuracy", 100. * correct / len(test_loader.dataset), epoch)

See :download:`mnist_tensorboard.py <../tutorial/mnist_tensorboard.py>` for more context.

Deploying Tensorboard
---------------------

Launch the AdaptDL TensorBoard deployment with

::

   adaptdl tensorboard create tensorboard-deployment --nodeport

This will create a deployment running tensorboard and a service to
expose tensorboard’s port.

Attaching TensorBoard
---------------------

When creating your AdaptDL job via the adaptdl cli, use the flag
``--tensorboard tensorboard-deployment``. This will attach the necessary
persistent volume claims and environment variables to your AdaptDL job.

For example, to launch the Tensorboard MNIST example from above, run the following in your command line.

::

    adaptdl submit . --tensorboard tensorboard-deployment -d tutorial/Dockerfile -f tutorial/adaptdljob.yaml

Accessing the TensorBoard GUI
-----------------------------

Using an external IP and the nodeport service
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you have an external IP address for your cluster, you may use the nodeport service
created in part 2 by the ``--nodeport`` flag. To access this, run

::

   kubectl describe services | grep tensorboard

Once you have located the tensorboard service, look for the ``Port`` parameter. Tensorboard
will be available on ``http://<external-ip>:<port>``. Follow `this guide <https://kubernetes.io/docs/tutorials/stateless-application/expose-external-ip-address/>`__
for more detailed information.

Using port-forwarding
^^^^^^^^^^^^^^^^^^^^^

If you do not have any external IP addresses for your cluster, you may
use ``kubectl port-forward`` to gain access to your TensorBoard UI.

The deployment created in part 2 of this guide has a single tensorboard
pod, with the name in the format of
``adaptdl-tensorboard-tensorboard-deployment-<some-unique-id>``. Find out the
exact name of the pod with
``kubectl get pods | grep tensorboard-deployment``.

Once you have the name of your pod, run
``kubectl port-forward <pod-name> 6006:6006``. This will
start a long running process which will serve the TensorBoard UI on
``http://localhost:6006``
