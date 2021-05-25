import argparse
import os
import random
import shutil
import time
import warnings

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import adaptdl
import adaptdl.torch
from apex import amp
from apex.amp._amp_state import _amp_state
from torch.utils.tensorboard import SummaryWriter


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--autoscale-bsz', dest='autoscale_bsz', default=False, action='store_true', help='autoscale batchsize')


def main_worker(args):

    # create model
    print("=> creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch]().cuda()

    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    optimizer = torch.optim.SGD(model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    adaptdl.torch.init_process_group("nccl")
    model, optimizer = amp.initialize(model, optimizer)
    model = adaptdl.torch.AdaptiveDataParallel(model, optimizer, patch_optimizer=False)

    cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = adaptdl.torch.AdaptiveDataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.workers, pin_memory=True)
    if args.autoscale_bsz:
        train_loader.autoscale_batch_size(12800, local_bsz_bounds=(20, 200), gradient_accumulation=True)

    val_loader = adaptdl.torch.AdaptiveDataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    with SummaryWriter(adaptdl.env.checkpoint_path() + "/tensorboard") as writer:
        for epoch in adaptdl.torch.remaining_epochs_until(args.epochs):
            adjust_learning_rate(optimizer, epoch, args)

            # train for one epoch
            train(train_loader, model, criterion, optimizer, epoch, args, writer)

            # evaluate on validation set
            acc1 = validate(val_loader, model, criterion, epoch, args, writer)


def train(train_loader, model, criterion, optimizer, epoch, args, writer):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader.dataset),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))
    stats = adaptdl.torch.Accumulator()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        output = model(images)
        loss = criterion(output, target)

        # measure accuracy and record loss
        (acc1, correct1, total1), (acc5, correct5, total5) = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))
        stats["correct1"] += correct1
        stats["total1"] += total1
        stats["correct5"] += correct5
        stats["total5"] += total5

        # compute gradient and do SGD step
        delay_unscale = not train_loader._elastic.is_sync_step()
        with amp.scale_loss(loss, optimizer, delay_unscale=delay_unscale) as scaled_loss:
            model.adascale.loss_scale = _amp_state.loss_scalers[0].loss_scale()
            scaled_loss.backward()
        model.adascale.step()

        stats["train_loss_sum"] += loss.item() * images.size(0)
        stats["train_total"] += images.size(0)
        train_loader.to_tensorboard(writer, epoch, "AdaptDL/Data")
        model.to_tensorboard(writer, epoch, "AdaptDL/Model")

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(train_loader._elastic.current_index)

    with stats.synchronized():
        stats["train_loss_avg"] = stats["train_loss_sum"] / stats["train_total"]
        stats["acc1"] = stats["correct1"] / stats["total1"]
        stats["acc5"] = stats["correct5"] / stats["total5"]
        writer.add_scalar("Loss/Train", stats["train_loss_avg"], epoch)


def validate(val_loader, model, criterion, epoch, args, writer):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader.dataset),
        [batch_time, losses, top1, top5],
        prefix='Test: ')
    stats = adaptdl.torch.Accumulator()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # compute output
            output = model(images)
            loss = criterion(output, target)

            # measure accuracy and record loss
            (acc1, correct1, total1), (acc5, correct5, total5) = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            stats["loss_sum"] += loss.item() * images.size(0)
            stats["loss_cnt"] += images.size(0)
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))
            stats["correct1"] += correct1
            stats["total1"] += total1
            stats["correct5"] += correct5
            stats["total5"] += total5

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            #if i % args.print_freq == 0:
            #    progress.display(val_loader.current_index)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))
        with stats.synchronized():
            stats["acc1"] = stats["correct1"] / stats["total1"]
            stats["acc5"] = stats["correct5"] / stats["total5"]
            stats["loss"] = stats["loss_sum"] / stats["loss_cnt"]
            writer.add_scalar("top1/Valid", stats["acc1"], epoch)
            writer.add_scalar("top5/Valid", stats["acc5"], epoch)
    
    return top1.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append((correct_k * 100.0 / batch_size, correct_k.item(), batch_size))
        return res


if __name__ == '__main__':
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    main_worker(args)
