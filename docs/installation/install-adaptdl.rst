Installing the AdaptDL Scheduler
================================

This page shows how to install the AdaptDL scheduler on an existing Kubernetes
instance. If you do not have a running Kubernetes instance, you may refer to
other pages to :doc:`deploy a single-node MicroK8s instance<deploy-microk8s>`,
or to :doc:`provision an auto-scaling cluster on EKS<deploy-eks>`.

.. note::

   The following instructions assume ``kubectl`` and ``helm`` are installed
   locally and configured with administrator access to an existing Kubernetes
   instance.

Install the AdaptDL Helm Chart
------------------------------

The AdaptDL scheduler can be installed in just a few commands using Helm.
First, add the AdaptDL Helm repository to you local environment:

.. code-block::

   $ helm repo add adaptdl https://github.com/petuum/adaptdl/raw/helm-repo

Next, update the repository with the latest charts:

.. code-block::

   $ helm repo update

Finally, install the AdaptDL scheduler chart:

.. code-block::

   $ helm install adaptdl adaptdl/adaptdl --namespace adaptdl --create-namespace \
     --set docker-registry.enabled=true

The above command installs Kubernetes deployments for the AdaptDL scheduler
service, as well as a Docker registry. The Docker registry is used to store
intermediate Docker images when submitting jobs with the AdaptDL CLI.

.. danger::

   The Docker registry installed with the AdaptDL scheduler is *insecure*.
   Please install an alternative secure registry for serious use!
   The included Docker registry may be disabled by omitting the
   ``--set docker-registry.enabled=true`` option, and then the AdaptDL CLI may
   be configured to use the alternative secure registry.

Check that the AdaptDL scheduler and Docker registry are running:

.. code-block::

   $ kubectl get pods -n adaptdl

Example output:

.. code-block::

   adaptdl-adaptdl-sched-7d8b689f45-9ds8h   3/3     Running            0          2m37s
   adaptdl-registry-7f45598964-t8df6        1/1     Running            0          2m37s

Next Steps
----------

Once the AdaptDL scheduler is installed and running, you may :doc:`run AdaptDL
jobs using the AdaptDL CLI<../commandline/index>`.
