Adaptdl on Ray AWS
==================

<WORKING-NAME-HERE> allows you to run an AdaptDL job on an AWS-Ray cluster.
The intention of this module is to allow you to get AdaptDL jobs working quickly, without the need to deploy kubernetes, and you to use Ray's cluster rescaling with AdaptDL's worker autoscaling.

Usage
-----

Modifications to your training code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In order for your code to run, your training code will need to use AdaptDL. Please follow `this tutorial <adaptdl-pytorch.rst>`_ for more information. 

Your code should follow these properties:

* You do not need to make any calls to Ray
* Your code will also need to be able to run from the command line
* The code can take can take command line arguments via ``sys.argv`` and ``argparse``
* The code is run as ``__main__``
* Local imports from the same directory as ``code.py`` are supported

Deploying a Ray cluster on AWS EC2
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You will need a ray cluster already deployed. Please see these `instructions <https://docs.ray.io/en/latest/cluster/cloud.html>`_ and `tutorial <https://medium.com/distributed-computing-with-ray/a-step-by-step-guide-to-scaling-your-first-python-application-in-the-cloud-8761fe331ef1>`_ for configuring and launching a ray cluster.

When creating the cluster, you will need configure the following:

* A dockerfile with these installed:
  * The pip requirements in `ray/aws/requirements.txt`
  * A working installation of pytorch-gpu
  * Whatever other pip dependencies you may require
* Sufficient disk space for the above docker image, and whatever disk space you may need to run your code
* Some maximum number of worker nodes

See `examples/cluster_config.yaml` for an example of the cluster.

To ensure that the ndoes have enough space for Docker to use, you will need to include something like the following `BlockDeviceMapping` configuration in all of the nodes:

.. code-block:: yaml
    node_config:
      InstanceType: <your instance type>
      BlockDeviceMappings:
        - DeviceName: /dev/sda1
          Ebs:
             VolumeSize: 100 #  Feel free to change this value

Just creating the EBS volume will not make it available for docker. You will also need to format and mount the volume as part of the initialization commands:

.. code-block:: yaml
    initialization_commands:
      - sudo usermod -aG docker $USER
      - sudo mkdir /docker_volume
      - sudo mkfs -t xfs /dev/nvme1n1
      - sudo mount /dev/nmve1n1 /docker_volume -w
      - sudo dockerd --data-root /docker_volume &

If you find that your code does not have enough access to disk space, you can also mount an external volume (as provisioned above) to the runtime containers via:

.. code-block:: yaml
   docker:
     image: <your-image-name>
     run_options:
     - -v '/<your-external-volume>:/<the-path-in-the-container>

Make sure that the permissions for the external volume are set properly.

Running your code
^^^^^^^^^^^^^^^^^

Once the cluster has been deployed, you will need the address and port of the cluster head. Generally, this will be of the form ``<head-node-ip>:10001``. Make sure that you have access to that port via the AWS subnet and inbound rules. 

If you have some AdaptDL training code runnable at ``code.py`` via ``python3 code.py <command-line-args>``, you can run the training code on Ray via 

``python3 adaptdl-ray -u "ray://head-node-ip:10001" -f code.py -m <maximum-number-of-workers> --cpus <cpus-per-worker> --gpus <gpus-per-worker> -- <command-line-args>``

If your local version of Python does not match the cluster's, Ray will not work. In this case, one option is to run the command within a Docker container. Be sure to mount your code directory in the container, e.g. via `-v`.

Retrieving your trained model
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In order to retrieve the result of your training code, you will need to manually save it to some external store. For example, you could write it to S3, or you could mount an EFS store to the cluster, and write it to that. See the Advanced Usage for more details on using EFS.

Example
-------

To run the example code found in ``examples/pytorch-cifar/main.py``, do the following:

1. Install the AWS CLI and authenticate.
2. Install Docker or Python version <exact version here> # TODO: replace version
3. Inside the ``example/ray/aws`` directory, run ``ray up -y cluster.yaml -v``. Note: running this step will create an AWS EC2 cluster, which will cost money
4. Keep track of the ip and port `ray up` returns.
5. Still inside ``example/ray/aws``, run ``docker run <docker version> python3 adaptdl_ray.py -f main.py -m 3 -u ray://<ip>:<port> -- -autoscale-bsz``. If you are using Python <exact version here>. then install the requirements in ``ray/aws/requirements.txt` and run `python3 adaptdl_ray.py -f main.py -m 3 -u ray://<ip>:<port> -- -autoscale-bsz``

 # TODO: replace stuff

Advanced Usage
--------------

Spot instances
^^^^^^^^^^^^^^

AdaptDL on Ray AWS supports spot instances for the ray cluster. Each of the workers will listen to the for the spot instance termination notification. If a node is scheduled to be deleted, a checkpoint will be taken and the job will be rescaled to exclude and find a replacement for that node.

Dealing with Large Datasets
^^^^^^^^^^^^^^^^^^^^^^^^^^^

As workers can be rescheduled to fresh nodes, downloading large datasets to each worker can be expensive. For example, if a worker downloads data for 20 minutes when it is scheduled to a new node, then the other workers will be idle for 20 minutes as well, even if they already have the data. This is exacerbated if the autoscaler gradually increases the number of workers.

There are several options to deal with this:

1. Use Amazon S3 with an `S3Dataset <https://aws.amazon.com/blogs/machine-learning/announcing-the-amazon-s3-plugin-for-pytorch/>`_.
2. Use EFS to share the data between the nodes

Using S3
^^^^^^^^

The difference with using an S3 Dataset in the Ray cluster versus on your local machine is ensuring that all of the nodes have the proper permissions. Please follow `these instructions <https://docs.ray.io/en/latest/cluster/aws-tips.html?highlight=s3#configure-worker-nodes-to-access-amazon-s3>`_

Using EFS
^^^^^^^^^

`EFS <https://aws.amazon.com/efs/>`_ allows you to use a distributed filesystem with your EC2 cluster. To begin, you will need to create an EFS instance. Once that is done, use the ``setup_commands`` listed `here <https://docs.ray.io/en/master/cluster/aws-tips.html?highlight=efs#using-amazon-efs>`_ to attach your EFS instance to the nodes.

Imports
^^^^^^^

If you need Python modules that are local to your machine but not located in the same directory as your main script, set ``--working-dir`` to a directory that contains the main script and all the Python modules. The argument to ``-f/--file`` should then be the path to the main script relative to the argument to ``--working-dir``.

Timeouts
^^^^^^^^

There are two conditions where the job controller will need to wait for some reponse. In order to prevent a lack of response from permamently stopping the job, there are timeouts.

First, when the workers are terminated in order to perform a rescaling, the controller will wait to recieve a checkpoint object of the training state from worker 0. If the controller does not receive a checkpoint by the amount of time specified in ``--checkpoint-timeout`` (default 120 seconds), then the controller will use a previous version of the checkpoint, or restart from 0, if a previous checkpoint does not exist. Note that spot instances have around a 2 minute warning for termination.

Second, when the cluster is rescaling to more workers, it can take some time for the new workers to be ready. In addition, spot instances requests may never be fulfilled if their bid price is too low. The controller therefore waits for some time, up to the amount specified in ``--cluster-rescale-timeout`` (default 60), for the new nodes to be provisioned and ready. If the nodes are not ready by that time, it schedules up to the maximum supported by the current cluster.
