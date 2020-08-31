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

AdaptDL consists of a *Kubernetes job scheduler* and a *distributed training
library*. They can be used in two ways:

1.  Scheduling multiple training jobs on a shared cluster or the cloud
    (:doc:`Scheduler Installation<installation/index>`).
2.  Adapting the batch size and learning rate for a single training job
    (:doc:`Standalone Training<standalone-training>`).

.. toctree::
   :maxdepth: 2
   :caption: Contents:
   :hidden:

   Home<self>
   Installation<installation/index.rst>
   Command Line Interface<commandline/index.rst>
   AdaptDL with PyTorch<adaptdl-pytorch.rst>
   Standalone Training<standalone-training.rst>
   API Reference<api/adaptdl.rst>
