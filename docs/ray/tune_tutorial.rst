=======================================
Using the Adaptive Tune Trial Scheduler
=======================================

This is a tutorial on using the AdaptDL as a Tune Trial Scheduler. We'll go
through an example that uses HyperOpt to tune hyperparameters like the learning
rate, momentum and initial batch size. The batch size and number of replicas
will be automatically adjusted by AdaptDL throughout the lifetimes of the
trials so as to efficiently and fairly share the resources of the Ray cluster.

We'll be relying on the PyTorch `DistributedTrainable` Tune API `documented
here <https://docs.ray.io/en/latest/tune/api_docs/trainable.html#distributed-torch>`_.


Setup
-----

1. Install the Ray nightly package for your platform by following instructions
   `here
   <https://docs.ray.io/en/latest/installation.html#daily-releases-nightlies>`_.

2. Install latest `adaptdl`, `adaptdl-ray` and `adaptdl-sched` pip packages.

3. Install HyperOpt.

4. Start the ray cluster.


Incorporating the AdaptDL API
-----------------------------

In order to make use of the Adaptive functionality, we will need to change the
trainable to include the AdaptDL API.

We don't change the model definition and test and train functions

.. code-block:: python

  class ConvNet(nn.Module):
      def __init__(self):
          super(ConvNet, self).__init__()
          # In this example, we don't change the model architecture
          # due to simplicity.
          self.conv1 = nn.Conv2d(1, 3, kernel_size=3)
          self.fc = nn.Linear(192, 10)

      def forward(self, x):
          x = F.relu(F.max_pool2d(self.conv1(x), 3))
          x = x.view(-1, 192)
          x = self.fc(x)
          return F.log_softmax(x, dim=1)


  # Change these values if you want the training to run quicker or slower.
  EPOCH_SIZE = 512
  TEST_SIZE = 256


  def train(model, optimizer, train_loader):
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      model.train()
      for batch_idx, (data, target) in enumerate(train_loader):
          # We set this just for the example to run quickly.
          if batch_idx * len(data) > EPOCH_SIZE:
              return
          data, target = data.to(device), target.to(device)
          optimizer.zero_grad()
          output = model(data)
          loss = F.nll_loss(output, target)
          loss.backward()
          optimizer.step()


  def test(model, data_loader):
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
      model.eval()
      correct = 0
      total = 0
      with torch.no_grad():
          for batch_idx, (data, target) in enumerate(data_loader):
              # We set this just for the example to run quickly.
              if batch_idx * len(data) > TEST_SIZE:
                  break
              data, target = data.to(device), target.to(device)
              outputs = model(data)
              _, predicted = torch.max(outputs.data, 1)
              total += target.size(0)
              correct += (predicted == target).sum().item()
          else:
              return 0
      return correct / total

The trainable function `train_mnist` needs to change though.

.. code-block:: diff

  +import adaptdl.torch as adl
  +
   def train_mnist(config: Dict, checkpoint_dir: Optional[str] = None):
       # Data Setup
       mnist_transforms = transforms.Compose(
           [transforms.ToTensor(),
            transforms.Normalize((0.1307, ), (0.3081, ))])

  -    train_loader = DataLoader(datasets.MNIST("~/data",
  +    train_loader = adl.AdaptiveDataLoader(datasets.MNIST("~/data",
           train=True, download=True, transform=mnist_transforms),
           batch_size=64,
           shuffle=True)

  -    test_loader = DataLoader(
  +    test_loader = adl.AdaptiveDataLoader(
           datasets.MNIST("~/data", train=False, transform=mnist_transforms),
           batch_size=64,
           shuffle=True)
  @@ -21,8 +23,9 @@

       model = ConvNet()
       model.to(device)
  -    model = DistributedDataParallel(model)
  +    model = adl.AdaptiveDataParallel(model, optimizer)

  -    for i in range(10):
  +    for epoch in adl.remaining_epochs_until(config.get("epochs", 10)):
           train(model, optimizer, train_loader)
           acc = test(model, test_loader)
           # Send the current training result back to Tune

The changes essentially make the dataloaders and model elastic and restart-safe
thus adding AdaptDL functionality. Now we need to use the the AdaptDL trial
scheduler which can actually make decisions based on available cluster
resources and trial characteristics.


.. code-block:: python
   :emphasize-lines: 17

  ray.init(address="auto")

  trainable_cls = DistributedTrainableCreator(train_mnist)

  space = {
      "bs": hp.choice("bs", range(64, 1024, 64)),
      "lr": hp.uniform("lr", 0.01, 0.1),
      "momentum": hp.uniform("momentum", 0.1, 0.9),
  }

  hyperopt_search = HyperOptSearch(space, metric="mean_accuracy", mode="max")

  analysis = tune.run(
      trainable_cls,
      num_samples=4,  # total trials will be num_samples x points on the grid
      scheduler=AdaptDLScheduler(),
      search_alg=hyperopt_search)

We first create a trainable (class) and a search space for HyperOpt. We call
`tune.run` and pass in `AdaptDLScheduler` as the trial scheduler for all the
trials. The `AdaptDLScheduler` will first try to use GPUs on the Ray cluster.
If it finds none, it will use CPUs to run the trials.

Full example can be found at `hyperopt_example.py
<https://github.com/petuum/adaptdl/ray/adaptdl_ray/examples/hyperopt_example.py>`_.

To run the example, simply run it from command line

.. code-block:: shell

   $ python3 hyperopt_example.py

   == Status ==
    Current time: 2021-10-26 12:55:14 (running for 00:04:55.09)
    Memory usage on this node: 2.1/31.2 GiB
    Using AdaptDL scheduling algorithm.
    Resources requested: 0/8 CPUs, 0/0 GPUs, 0.0/18.43 GiB heap, 0.0/9.21 GiB objects
    Result logdir: /tmp
    Number of trials: 4/4 (4 TERMINATED)
    +-------------------------------+------------+---------------------+----------+--------+------------------+
    | Trial name                    | status     | loc                 |      acc |   iter |   total time (s) |
    |-------------------------------+------------+---------------------+----------+--------+------------------|
    | AdaptDLTrainable_7_2_cd64740f | TERMINATED | 192.168.1.196:20687 | 0.957576 |    102 |          92.0071 |
    | AdaptDLTrainable_1_2_cd64740e | TERMINATED | 192.168.1.196:21408 | 0.930804 |    102 |         115.433  |
    | AdaptDLTrainable_1_2_cd647410 | TERMINATED | 192.168.1.196:21407 | 0.953125 |    102 |          75.8803 |
    | AdaptDLTrainable_5_2_ceeea272 | TERMINATED | 192.168.1.196:21612 | 0.872396 |    102 |         102.775  |
    +-------------------------------+------------+---------------------+----------+--------+------------------+

    Best trial config: {'bs': 960, 'epochs': 100, 'lr': 0.010874198064009714, 'momentum': 0.5627724615056127}
    Best trial mean_accuracy: 0.8723958333333334

The trial names in the end can be interpreted as
`AdaptDLTrainable_$num_replicas_$num_restarts_$trial_id`. Trials can expand or
shrink based on the decisions of the AdaptDL optimizer and this gets reflected
through their names.

