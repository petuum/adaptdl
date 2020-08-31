AdaptDL with PyTorch
====================

This page describes the steps needed to modify a simple `MNIST example
<https://github.com/pytorch/examples/blob/49ec0bd72b85be55579ae8ceb278c66145f593e1/mnist/main.py>`__
to use AdaptDL. Please see :download:`mnist_original.py <tutorial/mnist_original.py>`
for the original version and ``tutorial/mnist_step_<#>.py`` for the resulting changes
from each step number of this tutorial.``diff`` may be useful here to compare versions.

Initializing AdaptDL
--------------------

Once the training model ``model`` with optimizer ``optimizer`` and
(optional) learning rate scheduler ``scheduler`` have been created,
register all three with the following commands:

::

   adaptdl.torch.init_process_group("nccl" if torch.cuda.is_available()
                                    else "gloo")
   model = adaptdl.torch.AdaptiveDataParallel(model, optimizer, scheduler)

Please note that ``init_process_group`` must be called before the
``AdaptiveDataParallel`` object is created

In the MNIST tutorial example (:download:`mnist_step_1.py <tutorial/mnist_step_1.py>`), the changes will look like the following:

::

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    adaptdl.torch.init_process_group("nccl" if torch.cuda.is_available()
                                     else "gloo") # Changed
    model = adaptdl.torch.AdaptiveDataParallel(model, optimizer, scheduler) # Changed

Dataloading
-----------

AdaptDL requires you to use ``adaptdl.torch.AdaptiveDataLoader``. This
will require you to first have your dataset as a `torch dataset
object <https://pytorch.org/docs/stable/data.html#dataset-types>`__.
From there, the ``AdaptiveDataLoader`` supports the same arguments as the
standard PyTorch `DataLoader
class <https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader>`__.
Furthermore, the batchsize is not guaranteed to be the same as the
``batch_size`` argument. However, if batchsize autoscaling is not
enabled (see part 3), then the global batchsize will be very close that
provided via ``batch_size``.

In the MNIST example (:download:`mnist_step_2.py <tutorial/mnist_step_2.py>`), this is a matter of changing the dataloaders from

::

    dataset1 = datasets.MNIST('../data', train=True, download=True,
                              transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                              transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, batch_size=64,
                                               num_workers=1, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset2, batch_size=64,
                                              num_workers=1, shuffle=True)

to 

::

    dataset1 = datasets.MNIST('../data', train=True, download=True,
                              transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                              transform=transform)
    train_loader = adaptdl.torch.AdaptiveDataLoader(dataset1, drop_last=True, batch_size=64,
                                                   num_workers=1, shuffle=True)
    test_loader = adaptdl.torch.AdaptiveDataLoader(dataset2, batch_size=64,
                                                  num_workers=1, shuffle=True)

Setting ``drop_last=True`` allows the dataloader to properly deal with remainders when
dividing the dataset by the number of replicas

Adaptive Batch Size
-------------------

Enable AdaptDL to automatically scale the batch size based off of
throughput and gradient statistics via

::

   data_loader.autoscale_batch_size(
       max_global_batchsize,
       local_bsz_bounds=(min_local_batchsize, max_local_batchsize))

Note: this will allow the batchsize to change dynamically in training
via Adascale. Also note that this will generally require your optimizer
to be SGD.

In the context of the MNIST example (:download:`mnist_step_3.py <tutorial/mnist_step_3.py>`), the following change will need to be made:

::

    train_loader = adaptdl.torch.AdaptiveDataLoader(dataset1, drop_last=True, **kwargs)
    test_loader = adaptdl.torch.AdaptiveDataLoader(dataset2, **kwargs)

    train_loader.autoscale_batch_size(1028, local_bsz_bounds=(32, 128))

Please note that this call is optional, but required to allow the global batchsize to change dynamically over time.

Training Loop
-------------

The core training loop requires the following change from:

::

    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

to

::

    for epoch in adaptdl.torch.remaining_epochs_until(args.epochs): # Changed
        train(args, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)
        scheduler.step()

The call ``adaptdl.torch.remaning_epochs_until(args.epochs)`` will resume the epochs and batches
progressed when resuming from checkpoint after a job has been rescaled. See (:download:`mnist_step_4.py <tutorial/mnist_step_4.py>`).

Statistics Accumulation
-----------------------

To calculate useful metrics like loss or accuracy across replicas, use the
``adaptdl.torch.Accumulator`` class, which is a ``dict``-like object that sums across replicas
when ``synchronized`` is called.
However, outside of the ``stats.synchronized()`` context, get operations
are not supported. Furthermore, calling ``stats.synchronized()`` forces
blocking for synchronization across all replicas.

Whereas before collecting test data would look like:

::

    def test(model, device, test_loader):
        model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
    
        test_loss /= len(test_loader.dataset)
    
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

With AdaptDL statistics accumulation, it would look like:

::

    def test(model, device, test_loader):
        model.eval()
        stats = adaptdl.torch.Accumulator() # Changed in step 5
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                # CHANGED:
                stats["test_loss"] += F.nll_loss(output, target, reduction='sum').item()
                pred = output.argmax(dim=1, keepdim=True)
                stats["correct"] += pred.eq(target.view_as(pred)).sum().item()
   
        with stats.synchronized(): # Changed in step 5
            test_loss = stats["test_loss"] / len(test_loader.dataset) # Changed
            correct = stats["correct"] # Changed in step 5
   
            print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
                test_loss, correct, len(test_loader.dataset),
                100. * correct / len(test_loader.dataset)))

See (:download:`mnist_step_5.py <tutorial/mnist_step_5.py>`) for the full changes.
