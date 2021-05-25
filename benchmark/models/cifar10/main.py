'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *

import adaptdl
import adaptdl.torch

from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.tensorboard import SummaryWriter

from adaptdl.torch._metrics import report_train_metrics, report_valid_metrics


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--bs', default=128, type=int, help='batch size')
parser.add_argument('--lr', default=0.08, type=float, help='learning rate')
parser.add_argument('--epochs', default=100, type=int, help='number of epochs')
parser.add_argument('--model', default='ResNet18', type=str, help='model')
args = parser.parse_args()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root="/mnt", train=True, download=True, transform=transform_train)
trainloader = adaptdl.torch.AdaptiveDataLoader(trainset, batch_size=args.bs, shuffle=True, num_workers=2, drop_last=True)
trainloader.autoscale_batch_size(4096, local_bsz_bounds=(32, 1024),
                                 gradient_accumulation=True)

validset = torchvision.datasets.CIFAR10(root="/mnt", train=False, download=False, transform=transform_test)
validloader = adaptdl.torch.AdaptiveDataLoader(validset, batch_size=100, shuffle=False, num_workers=2)

# Model
print('==> Building model..')
net = eval(args.model)()
# net = VGG('VGG19')
# net = ResNet18()
# net = PreActResNet18()
# net = GoogLeNet()
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
net = net.to(device)
if device == 'cuda':
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD([{"params": [param]} for param in net.parameters()],
optimizer = optim.SGD(net.parameters(),
                      lr=args.lr, momentum=0.9, weight_decay=5e-4)
lr_scheduler = ExponentialLR(optimizer, 0.0133 ** (1.0 / args.epochs))

adaptdl.torch.init_process_group("nccl")
net = adaptdl.torch.AdaptiveDataParallel(net, optimizer, lr_scheduler)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    stats = adaptdl.torch.Accumulator()
    for inputs, targets in trainloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        stats["loss_sum"] += loss.item() * targets.size(0)
        _, predicted = outputs.max(1)
        stats["total"] += targets.size(0)
        stats["correct"] += predicted.eq(targets).sum().item()

        trainloader.to_tensorboard(writer, epoch, tag_prefix="AdaptDL/Data")
        net.to_tensorboard(writer, epoch, tag_prefix="AdaptDL/Model")

    with stats.synchronized():
        stats["loss_avg"] = stats["loss_sum"] / stats["total"]
        stats["accuracy"] = stats["correct"] / stats["total"]
        writer.add_scalar("Loss/Train", stats["loss_avg"], epoch)
        writer.add_scalar("Accuracy/Train", stats["accuracy"], epoch)
        report_train_metrics(epoch, stats["loss_avg"], accuracy=stats["accuracy"])
        print("Train:", stats)

def valid(epoch):
    net.eval()
    stats = adaptdl.torch.Accumulator()
    with torch.no_grad():
        for inputs, targets in validloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            stats["loss_sum"] += loss.item() * targets.size(0)
            _, predicted = outputs.max(1)
            stats["total"] += targets.size(0)
            stats["correct"] += predicted.eq(targets).sum().item()

    with stats.synchronized():
        stats["loss_avg"] = stats["loss_sum"] / stats["total"]
        stats["accuracy"] = stats["correct"] / stats["total"]
        writer.add_scalar("Loss/Valid", stats["loss_avg"], epoch)
        writer.add_scalar("Accuracy/Valid", stats["accuracy"], epoch)
        report_valid_metrics(epoch, stats["loss_avg"], accuracy=stats["accuracy"])
        print("Valid:", stats)


with SummaryWriter(os.getenv("ADAPTDL_TENSORBOARD_LOGDIR", "/tmp")) as writer:
    for epoch in adaptdl.torch.remaining_epochs_until(args.epochs):
        train(epoch)
        valid(epoch)
        lr_scheduler.step()
