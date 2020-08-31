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


import signal
import faulthandler
import argparse
import time
import os

import numpy as np
import torch
import torch.nn as nn
import torchtext
from torchtext.data.utils import get_tokenizer
from torch.utils.tensorboard import SummaryWriter

import adaptdl
import adaptdl.torch as adl
from adaptdl.torch.iterator import AdaptiveBPTTIterator


faulthandler.enable(all_threads=True)
faulthandler.register(signal.SIGUSR1, all_threads=True, chain=False)


parser = argparse.ArgumentParser(description='PyTorch Transformer')
parser.add_argument('--bs', default=32, type=int, help='batch size')
parser.add_argument('--lr', default=5.0, type=float, help='learning rate')
parser.add_argument('--epochs', default=16, type=int, help='number of epochs')
parser.add_argument('--autoscale-bsz', dest='autoscale_bsz', default=False,
                    action='store_true', help='autoscale batchsize?')
parser.add_argument('--bptt', default=30, type=int, help='sequence length')
parser.add_argument('--emsize', default=200, type=int,
                    help='embedding dimension')
parser.add_argument('--nhid', default=200, type=int, help='the dimension of \
                    the feedforward network model in nn.TransformerEncoder')
parser.add_argument('--nlayers', default=2, type=int, help='the number of \
                    nn.TransformerEncoderLayer in nn.TransformerEncoder')
parser.add_argument('--nhead', default=2, type=int, help='the number of heads \
                    in the multiheadattention models')
parser.add_argument('--dropout', default=0.2, type=float,
                    help='the dropout value')

args = parser.parse_args()


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
        mask = mask.float().masked_fill(mask == 0, float('-inf')).\
            masked_fill(mask == 1, float(0.0))
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
emsize = args.emsize  # embedding dimension
nhid = args.nhid  # dimension of feedforward network in nn.TransformerEncoder
nlayers = args.nlayers  # nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = args.nhead  # the number of heads in the multiheadattention models
dropout = args.dropout  # the dropout value

model = TransformerModel(ntokens, emsize, nhead,
                         nhid, nlayers, dropout).to(device)

criterion = nn.CrossEntropyLoss()
lr = args.lr
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

best_val_loss = float("inf")
epochs = args.epochs
best_model = None

adaptdl.torch.init_process_group("nccl" if torch.cuda.is_available()
                                     else "gloo")

model = adl.AdaptiveDataParallel(model, optimizer, scheduler)

if args.autoscale_bsz:
    max_batch_size = 1024 * batch_size
    local_bsz_bounds = (16, 256)
else:
    max_batch_size = None
    local_bsz_bounds = None

train_iter = AdaptiveBPTTIterator(train_txt,
                                 batch_size=args.bs,
                                 bptt_len=args.bptt,
                                 max_batch_size=max_batch_size,
                                 local_bsz_bounds=local_bsz_bounds)

val_iter = AdaptiveBPTTIterator(val_txt,
                               batch_size=args.bs,
                               bptt_len=args.bptt)

test_iter = AdaptiveBPTTIterator(test_txt,
                                batch_size=args.bs,
                                bptt_len=args.bptt)

def train(train_iter, epoch):
    stats = adl.Accumulator()
    model.train()  # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    ntokens = len(TEXT.vocab.stoi)
    for i, batch in enumerate(train_iter):
        optimizer.zero_grad()
        output = model(batch.text.to(device))
        loss = criterion(output.view(-1, ntokens),
                         batch.target.view(-1).to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        total_loss += loss.item()

        stats["loss_sum"] += loss.item() * batch.target.size(1)  # noqa: E501 [bptt, batch]
        stats["total"] += batch.target.size(1)

        writer.add_scalar("Throughput/Gain", model.gain, epoch)

        log_interval = 100
        if i % log_interval == 0 and i > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print(f'| epoch {epoch:3d} | batch {i:5d} | '
                  f'lr {scheduler.get_lr()[0]:02.2f} | '
                  f'ms/batch {elapsed * 1000 / log_interval:7.2f} | '
                  f'loss {cur_loss:5.2f} | ppl {np.exp(cur_loss):8.2f}')
            total_loss = 0
            start_time = time.time()

    with stats.synchronized():
        stats["loss_avg"] = stats["loss_sum"] / stats["total"]
        writer.add_scalar("Loss/Train", stats["loss_avg"], epoch)
        print("Train:", stats)


def evaluate(eval_model, val_iter):
    eval_model.eval()  # Turn on the evaluation mode
    stats = adl.Accumulator()
    ntokens = len(TEXT.vocab.stoi)
    with torch.no_grad():
        for batch in val_iter:
            output = eval_model(batch.text.to(device))
            output_flat = output.view(-1, ntokens)
            stats["loss_sum"] += batch.text.size(1) * \
                criterion(output_flat, batch.target.view(-1).to(device)).item()
            stats["total"] += batch.target.size(1)

    with stats.synchronized():
        loss_avg = stats["loss_avg"] = stats["loss_sum"] / stats["total"]
        writer.add_scalar("Loss/Valid", stats["loss_avg"], epoch)
        print("Valid:", stats)

    return loss_avg


tensorboard_dir = os.path.join(os.getenv("ADAPTDL_TENSORBOARD_LOGDIR", "/tmp"),
                               adaptdl.env.job_id())
with SummaryWriter(tensorboard_dir) as writer:
    for epoch in adl.remaining_epochs_until(args.epochs):
        epoch_start_time = time.time()
        train(train_iter, epoch)

        val_loss = evaluate(model, val_iter)
        print('-' * 89)
        print(f'| end of epoch {epoch:3d} '
              f'| time: {(time.time() - epoch_start_time):5.2f}s '
              f'| valid loss {val_loss:5.2f} '
              f'| valid ppl {np.exp(val_loss):8.2f}')
        print('-' * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model

        scheduler.step()

test_loss = evaluate(best_model, test_iter)
print('=' * 89)
print(f'| End of training | test loss {test_loss:5.2f} | '
      f'test ppl {np.exp(test_loss):8.2f}')
print('=' * 89)
