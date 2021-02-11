Deploying MicroK8s for AdaptDL
==============================

This page describes how to deploy a single-node MicroK8s Kubernetes instance
on which AdaptDL can be run. Refer to other pages if you want to run AdaptDL
on :doc:`an existing Kubernetes cluster<install-adaptdl>`, or on :doc:`an
auto-scaling cluster with EKS<deploy-eks>`.

.. note::

   The instructions on this page assume Ubuntu 18.04 or above with sudo access.

Installing MicroK8s
-------------------

First, install MicroK8s using Snap:

.. code-block:: shell

   $ sudo snap install microk8s --classic --channel=1.18/stable

The above command should install a barebones MicroK8s instance locally. Next,
enable dns:

.. code-block:: shell

   $ sudo microk8s enable dns

Enable gpu and storage:

.. code-block:: shell

   $ sudo microk8s enable gpu storage

The above command enables pods to utilize GPUs if available, and allows local
storage to be used for AdaptDL training checkpoints.

Initialize Helm, which is a package manager that can later be used to deploy
the AdaptDL scheduler:

.. code-block:: shell

   $ sudo microk8s enable helm
   $ sudo microk8s helm init --stable-repo-url=https://charts.helm.sh/stable
   $ sudo helm repo add stable https://charts.helm.sh/stable

Interacting with MicroK8s
-------------------------

Once MicroK8s is installed, you can interact with it via ``microk8s.kubectl``,
in the same way as using ``kubectl`` to interact with other Kubernetes
instances:

.. code-block:: shell

   $ sudo microk8s.kubectl get nodes

Example output:

.. code-block::

   NAME       STATUS     ROLES    AGE    VERSION
   gpu00100   Ready      <none>   10m    v1.18.8

If you prefer to omit ``sudo``, add your user to the ``microk8s`` group, and
then re-login to your shell:

.. code-block:: shell

   $ sudo usermod -a -G microk8s $USER

If you prefer to use ``kubectl`` rather than ``microk8s.kubectl``:

.. code-block:: shell

   $ mkdir -p $HOME/.kube
   $ sudo microk8s kubectl config view --raw > $HOME/.kube/config
   $ sudo chown -f -R $USER ~/.kube

The above step is recommended when later deploying AdaptDL onto MicroK8s.

Next Steps
----------

Once your MicroK8s instance is installed and running, you can :doc:`deploy the
AdaptDL scheduler<install-adaptdl>`.
