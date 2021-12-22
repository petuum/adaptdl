.. AdaptDL documentation master file, created by
   sphinx-quickstart on Tue Aug 11 19:20:38 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

AdaptDL Documentation
=====================

.. include:: README.rst
  :start-after: include-start-after
  :end-before: include-end-before

Getting Started
---------------

AdaptDL consists of a *job scheduler* and an *adaptive training library*. They
can be used in multiple ways:

1.  Scheduling multiple training jobs on a shared Kubernetes cluster or the cloud
    (:doc:`Scheduler Installation<installation/index>`).
2.  Adapting the batch size and learning rate for a single training job
    (:doc:`Standalone Training<standalone-training>`).
3.  As a Ray Tune Trial Scheduler
    (:doc:`Tune Trial Scheduler<ray/tune_tutorial>`).
4.  As a single training job running on a Ray AWS cluster
    (:doc:`Ray AWS Launcher<ray/aws_ray_adaptdl>`)

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :hidden:

   Home<self>
   Installation<installation/index.rst>
   Command Line Interface<commandline/index.rst>
   AdaptDL with PyTorch<adaptdl-pytorch.rst>
   Standalone Training<standalone-training.rst>
   Tune Trial Scheduler<ray/tune_tutorial.rst>
   API Reference<api/adaptdl.rst>
