import os
import math
import time
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

import adaptdl
import adaptdl.torch
from adaptdl.torch._metrics import report_train_metrics, report_valid_metrics

import model
import config
import evaluate
import data_utils


parser = argparse.ArgumentParser()
parser.add_argument("--lr", 
  type=float, 
  default=0.001, 
  help="learning rate")
parser.add_argument("--dropout", 
  type=float,
  default=0.0,  
  help="dropout rate")
parser.add_argument("--batch_size", 
  type=int, 
  default=256, 
  help="batch size for training")
parser.add_argument("--epochs", 
  type=int,
  default=20,
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
parser.add_argument("--adaptdl",
  action="store_true",
  help="enable adaptdl")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
cudnn.benchmark = True


############################## PREPARE DATASET ##########################
train_data, test_data, user_num, item_num, train_mat = data_utils.load_all()

# construct the train and test datasets
train_dataset = data_utils.NCFData(
    train_data, item_num, train_mat, args.num_ng, True)
test_dataset = data_utils.NCFData(
    test_data, item_num, train_mat, 0, False)
train_loader = data.DataLoader(train_dataset,
    batch_size=args.batch_size, shuffle=True, num_workers=4)
test_loader = data.DataLoader(test_dataset,
    batch_size=args.test_num_ng+1, shuffle=False, num_workers=0)

########################### CREATE MODEL #################################
if config.model == 'NeuMF-pre':
    assert os.path.exists(config.GMF_model_path), 'lack of GMF model'
    assert os.path.exists(config.MLP_model_path), 'lack of MLP model'
    GMF_model = torch.load(config.GMF_model_path)
    MLP_model = torch.load(config.MLP_model_path)
else:
    GMF_model = None
    MLP_model = None

model = model.NCF(user_num, item_num, args.factor_num, args.num_layers, 
                  args.dropout, config.model, GMF_model, MLP_model)
model.cuda()
loss_function = nn.BCEWithLogitsLoss()

if config.model == 'NeuMF-pre':
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
else:
    optimizer = optim.Adam(model.parameters(), betas=(0.9, 0.999), lr=args.lr)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [4, 7, 10], gamma=0.2)

train_loader = adaptdl.torch.AdaptiveDataLoader(train_dataset, drop_last=True,
                                                batch_size=args.batch_size,
                                                shuffle=True, num_workers=4)
train_loader.autoscale_batch_size(32768, local_bsz_bounds=(32, 32768),
                                  gradient_accumulation=True)

adaptdl.torch.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
model = adaptdl.torch.AdaptiveDataParallel(model, optimizer, scheduler)

########################### TRAINING #####################################
with SummaryWriter(os.getenv("ADAPTDL_TENSORBOARD_LOGDIR", "/tmp")) as writer:
    for epoch in adaptdl.torch.remaining_epochs_until(args.epochs):
        model.train() # Enable dropout (if have).
        start_time = time.time()
        train_loader.dataset.ng_sample()
        stats = adaptdl.torch.Accumulator()
        for idx, (user, item, label) in enumerate(train_loader):
            user = user.cuda()
            item = item.cuda()
            label = label.float().cuda()

            model.zero_grad()
            prediction = model(user, item)
            loss = loss_function(prediction, label)
            loss.backward()
            optimizer.step()

            stats['loss_sum'] += loss.item() * label.size(0)
            stats['total'] += label.size(0)

            train_loader.to_tensorboard(writer, epoch, "AdaptDL/Data")
            model.to_tensorboard(writer, epoch, "AdaptDL/Model")

        with stats.synchronized():
            stats['loss_avg'] = stats['loss_sum'] / stats['total']
            writer.add_scalar("Loss/Train", stats["loss_avg"], epoch)
            report_train_metrics(epoch, stats["loss_avg"])
            print("Train:", stats)

        model.eval()
        HR, NDCG = evaluate.metrics(model, test_loader, args.top_k)

        writer.add_scalar("HR/Valid", HR, epoch)
        writer.add_scalar("NDCG/Valid", NDCG, epoch)
        report_valid_metrics(epoch, 0., hr=HR, ndcg=NDCG)

        elapsed_time = time.time() - start_time
        print("\nThe time elapse of epoch {:03d}".format(epoch) + " is: " + 
              time.strftime("%H: %M: %S", time.gmtime(elapsed_time)))
        #print("HR: {:.3f}\tNDCG: {:.3f}".format(np.mean(HR), np.mean(NDCG)))

        scheduler.step()
