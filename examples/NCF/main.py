# Code adapted from https://github.com/guoyang9/NCF

import os
import time
import argparse
import numpy as np

import torch
import torch.nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import model
import evaluate
import data_utils

import adaptdl
import adaptdl.torch as adl
import adaptdl.env
import os.path

parser = argparse.ArgumentParser()
parser.add_argument("--lr",
                    type=float,
                    default=0.001,
                    help="learning rate")
parser.add_argument("--dropout",
                    type=float,
                    default=0.1,
                    help="dropout rate")
parser.add_argument("--batch_size",
                    type=int,
                    default=256,
                    help="batch size for training")
parser.add_argument("--epochs",
                    type=int,
                    default=40,
                    help="training epoches")
parser.add_argument("--top_k",
                    type=int,
                    default=10,
                    help="compute metrics@top_k")
parser.add_argument("--factor_num",
                    type=int,
                    default=32,
                    help="predictive factors numbers in the model")
parser.add_argument("--num_layers",
                    type=int,
                    default=3,
                    help="number of layers in MLP model")
parser.add_argument("--num_ng",
                    type=int,
                    default=4,
                    help="sample negative items for training")
parser.add_argument("--test_num_ng",
                    type=int,
                    default=99,
                    help="sample part of negative items for testing")
parser.add_argument("--out",
                    default=True,
                    help="save model or not")
parser.add_argument("--gpu",
                    type=str,
                    default="0",
                    help="gpu card ID")
parser.add_argument("--autoscale-bsz",
                    dest='autoscale_bsz',
                    default=False,
                    action='store_true',
                    help="Use AdaptDL batchsize autoscaling")
parser.add_argument("--gradient-accumulation",
                    dest='gradient_accumulation',
                    default=False,
                    action='store_true',
                    help="Use AdaptDL batchsize autoscaling")
parser.add_argument("--dataset",
                    type=str,
                    choices=['ml-1m', 'pinterest-20'],
                    default="ml-1m")
parser.add_argument("--model-type",
                    dest="model_type",
                    type=str,
                    choices=['MLP', 'GMF', 'NeuMF-end', 'NeuMF-pre'],
                    default="NeuMF-end")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
cudnn.benchmark = True

dataset = args.dataset
model_type = args.model_type

# paths
main_path = adaptdl.env.share_path()

train_rating = os.path.join(main_path, '{}.train.rating'.format(dataset))
test_rating = os.path.join(main_path, '{}.test.rating'.format(dataset))
test_negative = os.path.join(main_path, '{}.test.negative'.format(dataset))

model_path = os.path.join(main_path, 'models')
GMF_model_path = os.path.join(model_path, 'GMF.pth')
MLP_model_path = os.path.join(model_path, 'MLP.pth')
NeuMF_model_path = os.path.join(model_path, 'NeuMF.pth')

############################## PREPARE DATASET ##########################
train_data, test_data, user_num, item_num, train_mat = \
    data_utils.load_all(main_path, train_rating, test_negative)

# construct the train and test datasets
train_dataset = data_utils.NCFData(
        train_data, item_num, train_mat, args.num_ng, True)
test_dataset = data_utils.NCFData(
        test_data, item_num, train_mat, 0, False)
train_loader = adl.AdaptiveDataLoader(
    train_dataset,
    batch_size=args.batch_size, shuffle=True, num_workers=4, drop_last=True)
test_loader = adl.AdaptiveDataLoader(
    test_dataset,
    batch_size=args.test_num_ng+1, shuffle=False, num_workers=0)

if args.autoscale_bsz:
    train_loader.autoscale_batch_size(
        8192, local_bsz_bounds=(32, 512),
        gradient_accumulation=args.gradient_accumulation)

########################### CREATE MODEL #################################
if model_type == 'NeuMF-pre':
    assert os.path.exists(GMF_model_path), 'lack of GMF model'
    assert os.path.exists(MLP_model_path), 'lack of MLP model'
    GMF_model = torch.load(GMF_model_path)
    MLP_model = torch.load(MLP_model_path)
else:
    GMF_model = None
    MLP_model = None

network = model.NCF(user_num, item_num, args.factor_num, args.num_layers,
                    args.dropout, model_type, GMF_model, MLP_model)
adaptdl.torch.init_process_group("nccl" if torch.cuda.is_available()
                                 else "gloo")
network.cuda()
loss_function = torch.nn.BCEWithLogitsLoss()

if model_type == 'NeuMF-pre':
    optimizer = optim.SGD(network.parameters(), lr=args.lr)
else:
    optimizer = optim.Adam(network.parameters(), lr=args.lr)
network = adl.AdaptiveDataParallel(network, optimizer,
                                   find_unused_parameters=True)
########################### TRAINING #####################################
count, best_hr = 0, 0
tensorboard_dir = os.path.join(os.getenv("ADAPTDL_TENSORBOARD_LOGDIR", "/tmp"),
                               adaptdl.env.job_id())
with SummaryWriter(tensorboard_dir) as writer:
    for epoch in adl.remaining_epochs_until(args.epochs):
        network.train()  # Enable dropout (if have).
        start_time = time.time()
        train_loader.dataset.ng_sample()
        gain = 0
        for user, item, label in train_loader:
            user = user.cuda()
            item = item.cuda()
            label = label.float().cuda()

            network.zero_grad()
            prediction = network(user, item)
            loss = loss_function(prediction, label)
            loss.backward()
            optimizer.step()
            count += 1
            gain = network.gain
            batchsize = train_loader.current_batch_size
            accumulation_steps = train_loader.accumulation_steps

        writer.add_scalar("Throughput/Gain", gain, epoch)
        writer.add_scalar("Throughput/Global_Batchsize",
                          batchsize, epoch)
        writer.add_scalar("Throughput/Accumulation_Steps",
                          accumulation_steps, epoch)

        network.eval()
        stats = adl.Accumulator()
        HR, NDCG = evaluate.metrics(network, test_loader, args.top_k)
        stats['HR'] += HR
        stats['replicas'] += 1.0
        with stats.synchronized():
            writer.add_scalar('Loss/HR', stats['HR'] / stats['replicas'],
                              epoch)

        elapsed_time = time.time() - start_time
        print("The time elapse of epoch {:03d}".format(epoch) + " is: " +
              time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
        print("HR: {:.3f}\tNDCG: {:.3f}".format(np.mean(HR), np.mean(NDCG)))

        if HR > best_hr:
            best_hr, best_ndcg, best_epoch = HR, NDCG, epoch
            if args.out and adaptdl.env.replica_rank() == 0:
                if not os.path.exists(model_path):
                    os.mkdir(model_path)
                torch.save(
                    network._state.model,
                    '{}/{}.pth'.format(model_path, model_type))

print("End. Best epoch {:03d}: HR = {:.3f}, NDCG = {:.3f}".format(
      best_epoch, best_hr, best_ndcg))
