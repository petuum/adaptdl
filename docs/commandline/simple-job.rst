Submitting a Simple Job
=======================

This page is an introduction to running AdaptDL jobs using a simple "Hello,
world!" program. The goal is to show the basics of creating and interacting
with AdaptDL jobs. For an introduction to modifying existing PyTorch code to
use AdaptDL, please see :doc:`AdaptDL with PyTorch<../adaptdl-pytorch>`.

Installation
------------

::

   python3 -m pip install adaptdl-cli

Writing a Simple Program
------------------------

For the purpose of this guide, you will want a simple python script that
produces output to ``adaptdl.env.share_path()``, the directory used for
your job for storing general files.

For example, you may copy the following code (into ``hello_world/hello_world.py``):

::

   import adaptdl.env
   import os
   import time

   print("Hello, world!")

   with open(os.path.join(adaptdl.env.share_path(), "foo.txt"), "w") as f:
       f.write("Hello, world!")

   time.sleep(100)

Please note that stdout is only accessible while a job is still running.
Therefore, the ``time.sleep(100)`` call is important for this tutorial.

Writing a Dockerfile
--------------------

In order to run your application code, the job containers need access to
the code directly. A simple method is to create a docker image containing
the application.

Currently the ``adaptdl`` cli requires you to be able to push
to and the cluster to be able to pull from a docker registry. This may
be dockerhub, or it may be your own private docker registry. Please
ensure that that is set up before proceeding.

Copy the following docker file into ``hello_world/Dockerfile``:

::

    FROM python:3.7-slim
    RUN python3 -m pip install adaptdl

    COPY hello_world.py /root/hello_world.py

    ENV PYTHONUNBUFFERED=true

.. tip::

   If the Dockerfile is not written carefully, the Docker build step can take a
   long time. Make sure to follow the best practices when writing your
   Dockerfile so your builds are as fast as possible:

   #. `Exploiting caching in Dockerfile to re-use layers and speed up builds <https://pythonspeed.com/articles/docker-caching-model/>`_
   #. `Using .dockerignore to minimize the size of your docker context. <https://devopsheaven.com/docker/dockerignore/2018/04/25/using-dockerignore.html>`_

   In particular, you should (almost) always have a ``.dockerignore`` file that
   contains ``.git`` and other large files/directories which are not used in
   your containers.

Configuring the Job
-------------------

AdaptDL jobs are specified as Kuberenetes Resource. The following yaml file defines
the job specification for your hello world application:

Example (in ``hello_world/adaptdljob.yaml``):

::

   apiVersion: adaptdl.petuum.com/v1
   kind: AdaptDLJob
   metadata:
     generateName: hello-world-
   spec:
     template:
       spec:
         containers:
         - name: main
           command:
           - python3
           - /root/hello_world.py

Submitting the Job
------------------

Run the following AdaptDL cli command from your client.

::

   adaptdl submit hello_world

.. note::

   If you are using Docker for Mac with AdaptDL's built-in insecure registry, the first run of
   ``adaptdl submit`` may fail with an error similar to:

   ::

      Get https://host.docker.internal:59283/v2/: x509: certificate signed by unknown authority

   You may need to restart Docker, and ``adaptdl submit`` should work thereafer.

This will create the AdaptDL Kubernetes job object for your application. Once this is created,
the AdaptDL scheduler will recognize the job and schedule it for execution. Please note that for
this command to work, the docker file created in step 3 must be located in `hello_world/Dockerfile`
and the yaml created in step 4 must be located in `hello_world/adaptdljob.yaml`. 

Monitoring the Job
------------------

Once the job object has been created, you can find more information about the job using

::

    adaptdl ls

This should produce some output similar to 

::

    Name                                                             Status     Start(UTC)    Runtime  Rplc  Rtrt
    hello-world-kgjsc                                                Running    Aug-24 18:47  1 min    1     0

Once the ``Status`` is listed as ``Running`` and not ``Pending``, then the AdaptDL scheduler has
created pods for your AdaptDL job. Use the following command to find out more details about the pods:

::

    kubectl get pods

This should produce an output that looks like

::

    NAME                                                         READY   STATUS     RESTARTS   AGE
    adaptdl-adaptdl-sched-856cc685c4-hhdks                       3/3     Running    0          8h
    hello-world-kgjsc-a7fe6b49-e673-11ea-a27e-061e69fb5c39-0-0   1/1     Running    0          20s

Note that this gets all of the pods in the default namespace, including the scheduler. To restrict this to just the pods
created for your job, use ``kubectl get pods | grep hello-world``.

When the phase is listed as ``Running``, as opposed to ``ContainerCreating``, then you can get the stdout and stderr logs
via the following, (replacing ``<pod-name>`` with the name value you got from ``kubectl get pods``):

::

    kubectl logs <pod-name>

This should produce output of ``Hello, world!``.

Please note that this method of getting stdout and stderr output requires the pod to still exist. However,
when an AdaptDL job finishes or rescales, the worker pods are deleted. For more durable logging, it is advised to
write to a file.

Retrieving Output Files
-----------------------

Use the following to copy result files to your client machine. Please replace ``<adaptdl-job>`` with the name
value from the output of ``adaptdl ls`` in step 10:

::

    adaptdl cp <adaptdl-job>:/adaptdl/share/foo.txt foo.txt

``foo.txt`` on your local client should then contain ``hello world``

Deleting the Job
----------------

Delete the job with kubectl: ``kubectl delete adaptdljob <adaptdl-job>``. Again, replace the name parameter with the one
from before. This will delete the AdaptDL kubernetes object from your job, which will also delete any running pods or other
attached resources. Please note that this may cause files the job has written to to no longer be available.

(Advanced) External Registry
----------------------------------

If possible, we recommend using a secure external Docker registry instead of
the default insecure registry installed along with the AdaptDL scheduler. To do
this, you'll need to export two environment variables to let AdaptDL know the
full reponame to use, say ``registry.example.com/adaptdl-submit``, along with
registry credentials ``mysecret``. Refer to `this website
<https://kubernetes.io/docs/tasks/configure-pod-container/pull-image-private-registry/#create-a-secret-by-providing-credentials-on-the-command-line>`_
for how to create one.

.. code-block:: shell

   export ADAPTDL_SUBMIT_REPO=registry.example.com/adaptdl-submit
   export ADAPTDL_SUBMIT_REPO_CREDS=mysecret

Then do ``docker login`` in with the registry credentials.
