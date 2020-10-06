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

   adaptdl tensorboard create my-tensorboard

This will create a deployment running tensorboard and a service to
expose tensorboard’s port.

Attaching TensorBoard
---------------------

When creating your AdaptDL job via the adaptdl cli, use the flag
``--tensorboard my-tensorboard``. This will attach the necessary
persistent volume claims and environment variables to your AdaptDL job.

For example, to launch the Tensorboard MNIST example from above, run the following in your command line.

::

    adaptdl submit . --tensorboard my-tensorboard -d tutorial/Dockerfile -f tutorial/adaptdljob.yaml

Accessing TensorBoard
---------------------

To access the GUI of your TensorBoard instance running in Kubernetes, you can
start a proxy to it locally:

::

    $ adaptdl tensorboard proxy my-tensorboard -p 8080
    Proxying to TensorBoard instance my-tensorboard at http://localhost:8080

The proxy will keep running until you manually stop it by sending an interrupt.
Now, you can view your TensorBoard instance by pointing your favorite browser
to ``http://localhost:8080``.
