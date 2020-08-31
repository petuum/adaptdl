.. image:: _static/img/AdaptDLHorizLogo.png
  :align: center

.. image:: https://readthedocs.org/projects/adaptdl/badge/?version=latest
  :target: https://adaptdl.readthedocs.io/en/latest/?badge=latest
  :alt: Documentation Status

`Documentation <https://adaptdl.readthedocs.org>`_ |
`Examples <https://github.com/petuum/adaptdl/tree/master/examples>`_

.. include-start-after

AdaptDL is a *resource-adaptive* deep learning (DL) training and scheduling
framework. The goal of AdaptDL is to make distributed DL easy and efficient in
dynamic-resource environments such as shared clusters and the cloud.

Some core features offered by AdaptDL are:

*  Elastically schedule distributed DL training jobs in shared clusters.
*  Cost-aware resource auto-scaling in cloud computing environments (e.g. AWS).
*  Automatic batch size and learning rate scaling for distributed training.

AdaptDL supports PyTorch training programs. TensorFlow support coming soon!

Why AdaptDL?
------------

AdaptDL's state-of-the-art scheduling algorithm directly optimizes cluster-wide
training performance and resource utilization. By elastically re-scaling jobs,
co-adapting batch sizes and learning rates, and avoiding network interference,
AdaptDL improves shared-cluster training compared with alternative schedulers.

.. image:: _static/img/scheduling-performance.png
  :align: center

In the cloud (e.g. AWS), AdaptDL auto-scales the size of the cluster based on
how well those cluster resources are utilized. AdaptDL automatically
provisions spot instances when available to reduce cost by up to 80%.

Efficient distributed training requires careful selection of the batch size and
learning rate, which can be tricky to find manually. AdaptDL offers automatic
batch size and learning rate scaling, which enables efficient distributed
training without requiring manual effort.

.. image:: _static/img/autobsz-performance.png
  :align: center

.. include-end-before

Getting Started
---------------

AdaptDL consists of a *Kubernetes job scheduler* and a *distributed training
library*. They can be used in two ways:

1.  Scheduling multiple training jobs on a shared cluster or the cloud
    (`Scheduler Installation <https://adaptdl.readthedocs.io/en/latest/installation/index.html>`_).
2.  Adapting the batch size and learning rate for a single training job
    (`Standalone Training <https://adaptdl.readthedocs.io/en/latest/standalone-training.html>`_).
