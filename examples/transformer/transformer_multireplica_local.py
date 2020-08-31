# Copyright 2020 Petuum, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


""" This is a distributed training app which runs locally on your machine from
the command line. It will spawn 2 workers and train Transformer in
data-parallel fashion locally. The model size is cut down to make the model fit
on local memory. This should mainly be used for debugging things like AdaptDL
scheduler locally. Also can be used for testing the Reducer"""

import os
import sys
import time
from multiprocessing import Process
import argparse
import signal
import faulthandler

import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
import torchtext
from torchtext.data.utils import get_tokenizer

import adaptdl
import adaptdl.torch as adl
from adaptdl.torch.iterator import AdaptiveBPTTIterator

parser = argparse.ArgumentParser(description='PyTorch Transformer')
parser.add_argument('--bs', default=20, type=int, help='batch size')
parser.add_argument('--lr', default=5.0, type=float, help='learning rate')
parser.add_argument('--epochs', default=2, type=int, help='number of epochs')
parser.add_argument('--autoscale-bsz', dest='autoscale_bsz', default=False,
                    action='store_true', help='autoscale batchsize')
parser.add_argument('--bptt', default=35, type=int, help='bptt')

args = parser.parse_args()

signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))


class TransformerModel(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerModel, self).__init__()
        from torch.nn import TransformerEncoder, TransformerEncoderLayer
        self.model_type = 'Transformer'
        self.src_mask = None
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)

        self.init_weights()

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(
                mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            mask = self._generate_square_subsequent_mask(len(src)).to(device)
            self.src_mask = mask

        src = self.encoder(src) * np.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, self.src_mask)
        output = self.decoder(output)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


TEXT = torchtext.data.Field(tokenize=get_tokenizer("basic_english"),
                            init_token='<sos>',
                            eos_token='<eos>',
                            lower=True)
train_txt, val_txt, test_txt = torchtext.datasets.WikiText2.splits(TEXT)

TEXT.build_vocab(train_txt)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = args.bs
eval_batch_size = args.bs

ntokens = len(TEXT.vocab.stoi)  # the size of vocabulary
emsize = 8  # embedding dimension
nhid = 8   # dimension of the feedforward network in nn.TransformerEncoder
nlayers = 1  # number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 1  # the number of heads in the multiheadattention models
dropout = 0.2  # the dropout value


class Replica(object):
    def __init__(self, rank=0, world=1):
        faulthandler.enable(all_threads=True)
        faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)
        signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

        self.rank = rank
        self.world = world

        if not os.getenv("ADAPTDL_REPLICA_RANK"):
            os.environ["ADAPTDL_REPLICA_RANK"] = str(self.rank)
        if not os.getenv("ADAPTDL_MASTER_ADDR"):
            os.environ["ADAPTDL_MASTER_ADDR"] = "127.0.0.1"
        if not os.getenv("ADAPTDL_JOB_ID"):
            os.environ["ADAPTDL_JOB_ID"] = "transformer"

        if not os.getenv("ADAPTDL_PLACEMENT"):
            # localhost,localhost
            os.environ["ADAPTDL_PLACEMENT"] = (self.world * "localhost,")[:-1]

        self.model = TransformerModel(ntokens, emsize, nhead, nhid,
                                      nlayers, dropout).to(device)

        self.criterion = nn.CrossEntropyLoss()
        self.lr = args.lr
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1.0,
                                                         gamma=0.95)

        self.best_val_loss = float("inf")
        self.best_model = None

        adaptdl.torch.init_process_group("gloo")
        self.model = adl.AdaptiveDataParallel(self.model, self.optimizer,
                                            self.scheduler)
        if args.autoscale_bsz:
            self.max_batch_size = 1024 * batch_size
            self.local_bsz_bounds = (batch_size, 128)
        else:
            self.max_batch_size = None
            self.local_bsz_bounds = None

    def train(self, train_data, epoch, writer):
        stats = adl.Accumulator()
        self.model.train()  # Turn on the train mode
        total_loss = 0.
        start_time = time.time()
        ntokens = len(TEXT.vocab.stoi)
        for i, batch in enumerate(AdaptiveBPTTIterator(train_data,
                                                      batch_size=args.bs,
                                                      bptt_len=args.bptt,
                                                      max_batch_size=self.max_batch_size,  # noqa: E501
                                                      local_bsz_bounds=self.local_bsz_bounds)):  # noqa: E501
            self.optimizer.zero_grad()
            output = self.model(batch.text.to(device))
            loss = self.criterion(output.view(-1, ntokens),
                                  batch.target.view(-1).to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
            total_loss += loss.item()

            stats["loss_sum"] += loss.item() * batch.target.size(1)
            stats["total"] += batch.target.size(1)

            writer.add_scalar("Throughput/Gain", self.model.gain, epoch)

            log_interval = 10
            if i % log_interval == 0 and i > 0:
                cur_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                print(f'| epoch {epoch:3d} | batch {i:5d} | '
                      f'lr {self.scheduler.get_lr()[0]:02.2f} | '
                      f'ms/batch {elapsed * 1000 / log_interval:7.2f} | '
                      f'loss {cur_loss:5.2f} | ppl {np.exp(cur_loss):8.2f}')
                total_loss = 0
                start_time = time.time()

        with stats.synchronized():
            stats["loss_avg"] = stats["loss_sum"] / stats["total"]
            writer.add_scalar("Loss/Train", stats["loss_avg"], epoch)
            print("Train:", stats)

    def evaluate(self, eval_model, data_source, epoch=0, writer=None):
        eval_model.eval()  # Turn on the evaluation mode
        stats = adl.Accumulator()
        ntokens = len(TEXT.vocab.stoi)
        with torch.no_grad():
            for batch in AdaptiveBPTTIterator(data_source,
                                             batch_size=args.bs,
                                             bptt_len=args.bptt):
                output = eval_model(batch.text.to(device))
                output_flat = output.view(-1, ntokens)
                stats["loss_sum"] += batch.text.size(1) * \
                    self.criterion(output_flat,
                                   batch.target.view(-1).to(device)).item()
                stats["total"] += batch.target.size(1)

        with stats.synchronized():
            loss_avg = stats["loss_avg"] = stats["loss_sum"] / stats["total"]
            if writer:
                writer.add_scalar("Loss/Valid", stats["loss_avg"], epoch)
            print("Valid:", stats)

        return loss_avg

    def run(self):
        tensorboard_dir = os.path.join(os.getenv("ADAPTDL_TENSORBOARD_LOGDIR", "/tmp"),
                                       adaptdl.env.job_id())
        with SummaryWriter(tensorboard_dir) as writer:
            for epoch in adl.remaining_epochs_until(args.epochs):
                epoch_start_time = time.time()
                self.train(train_txt, epoch, writer)
                val_loss = self.evaluate(self.model, val_txt, epoch, writer)

                print('-' * 89)
                print(f'| end of epoch {epoch:3d} '
                      f'| time: {(time.time() - epoch_start_time):5.2f}s '
                      f'| valid loss {val_loss:5.2f} '
                      f'| valid ppl {np.exp(val_loss):8.2f}')
                print('-' * 89)

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.best_model = self.model

                self.scheduler.step()

        test_loss = self.evaluate(self.best_model, test_txt)
        print('=' * 89)
        print(f'| End of training | test loss {test_loss:5.2f} | '
              f'test ppl {np.exp(test_loss):8.2f}')
        print('=' * 89)


def main(rank, size):
    Replica(rank, size).run()


if __name__ == "__main__":
    size = 2  # number of replicas
    processes = []
    for rank in range(size):
        p = Process(target=main, args=(rank, size), daemon=True)
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
