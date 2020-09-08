Provisioning EKS for AdaptDL
============================

This page describes how to setup an AWS EKS cluster that auto-scales according
to cluster load. Refer to other pages if you want to run AdaptDL on :doc:`an
existing Kubernetes cluster<install-adaptdl>`, or on :doc:`an a single node
with MicroK8s<deploy-microk8s>`.

.. note::

   The instruction on this page assume ``eksctl``, ``kubectl``, ``helm`` and
   ``awscli`` are installed locally. You can follow `this guide
   <https://docs.aws.amazon.com/eks/latest/userguide/getting-started-eksctl.html>`_
   to install all the tools needed.

.. attention::

   This guide will provision AWS resources which will cost money. As of August
   2020, you pay $0.10 per hour for each Amazon EKS cluster that you create.
   $0.30 GB-Month for the EFS storage and $0.526 per hour per ``g4dn.xlarge``
   instance that you will end up using, starting with one. Note because the
   cluster is auto-scaling, additional instances will be spawned only when
   needed and you will be charged only for the duration of their lifetimes.

Provisioning the Cluster
------------------------

You may use the provided manifest to create the cluster. Some configurations
may be changed as per your preferences by downloading and modifying the file.

.. code-block:: shell

   eksctl create cluster -f https://raw.githubusercontent.com/petuum/adaptdl/master/deploy/eks/adaptdl-eks-cluster-on-demand.yaml

This will provision an elastic EKS cluster with name ``adaptdl-eks-cluster``
with 1 minimum and 4 maximum nodes in the ``us-west-2`` region. All nodes are
on-demand ``g4dn.xlarge`` instances with a single GPU each. You can change the
instance type and auto-scaling limits by changing ``nodeGroups.instanceType``,
``nodeGroups.minSize``, and ``nodeGroups.maxSize``, respectively. You can also
change the cluster ``name``, AWS ``region`` of your choice.

Make sure the ``CLUSTER_NAME`` and ``AWS_REGION`` environment variables reflect
the correct values after this step, for example:

.. code-block::

   export CLUSTER_NAME=adaptdl-eks-cluster
   export AWS_REGION=us-west-2

Provisioning EFS
----------------

AdaptDL depends on a distributed filesystem like EFS to save and load
checkpoints during training. You may follow the instructions from `this website
<https://www.eksworkshop.com/beginner/190_efs/launching-efs/>`_ to provision an
EFS volume for your cluster.

Next, install the EFS provisioner Helm chart. Make sure you have set the
``FILE_SYSTEM_ID`` environment variable according to the linked instructions.

.. code-block:: shell

   helm repo add stable https://kubernetes-charts.storage.googleapis.com/

   helm repo update

   helm install stable/efs-provisioner \
   --set efsProvisioner.efsFileSystemId=$FILE_SYSTEM_ID \
   --set efsProvisioner.awsRegion=$AWS_REGION \
   --generate-name


Installing the Cluster Autoscaler
---------------------------------

.. code-block:: shell

   helm repo add autoscaler https://kubernetes.github.io/autoscaler

   helm repo update

   helm install autoscaler/cluster-autoscaler-chart \
   --set autoDiscovery.clusterName=$CLUSTER_NAME \
   --set awsRegion=$AWS_REGION \
   --generate-name

To verify that cluster-autoscaler has started, run:

.. code-block:: shell

   kubectl --namespace=default get pods -l "app.kubernetes.io/name=aws-cluster-autoscaler-chart"

Should show the Cluster Autoscaler pod as ``Running``

Installing the NVIDIA Plugin
----------------------------

.. code-block:: shell

   kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.6.0/nvidia-device-plugin.yml

(Optional) Registry Access
--------------------------

If you will be using AdaptDL's insecure registry, you will need to add a new
rule to the security group associated with the nodes of the cluster. You may
need help from your AWS administrator to perform this step.

.. code-block:: shell

   SECURITY_GROUP=$(aws cloudformation describe-stack-resources --stack-name \
   eksctl-$CLUSTER_NAME-nodegroup-ng-1 --query \
   'StackResources[?LogicalResourceId == `SG`].[PhysicalResourceId]' --output text)

   aws ec2 authorize-security-group-ingress --group-id $SECURITY_GROUP \
   --protocol tcp --port 32000 --cidr 0.0.0.0/0

Cleaning Up
-----------

Once you are done with the cluster, you can clean up all AWS resources with:

.. code-block:: shell

   eksctl delete cluster --name $CLUSTER_NAME

   for target in `aws efs describe-mount-targets --file-system-id $FILE_SYSTEM_ID --query 'MountTargets[].MountTargetId' --output text`; \
   do aws efs delete-mount-target --mount-target-id $target; done

   aws efs delete-file-system --file-system-id $FILE_SYSTEM_ID

Next Steps
----------

Once your EKS cluster is provisioned and running, you can :doc:`deploy the
AdaptDL scheduler<install-adaptdl>`.
