import argparse
import time
import math
import torch
import torch.nn as nn
from model import MLMTask
from utils import run_demo, run_ddp, wrap_up
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter # Added for tensorboard

import adaptdl # Changed in step 1
import adaptdl.torch # Changed in step 1
import os # Added for tensorboard

def collate_batch(batch_data, args, mask_id, cls_id):
    if len(batch_data) %  args.bptt != 0:
        # print(len(batch_data))
        batch_data = batch_data[:len(batch_data)//args.bptt*args.bptt]
    # print(len(batch_data))
    batch_data = torch.tensor(batch_data).long().view(args.bptt, -1).t().contiguous()
    # print(batch_data.shape)
    # Generate masks with args.mask_frac
    data_len = batch_data.size(0)
    ones_num = int(data_len * args.mask_frac)
    zeros_num = data_len - ones_num
    lm_mask = torch.cat([torch.zeros(zeros_num), torch.ones(ones_num)])
    lm_mask = lm_mask[torch.randperm(data_len)]
    batch_data = torch.cat((torch.tensor([[cls_id] * batch_data.size(1)]).long(), batch_data))
    lm_mask = torch.cat((torch.tensor([0.0]), lm_mask))

    targets = torch.stack([batch_data[i] for i in range(lm_mask.size(0)) if lm_mask[i]]).view(-1)
    batch_data = batch_data.masked_fill(lm_mask.bool().unsqueeze(1), mask_id)
    return batch_data, lm_mask, targets


def process_raw_data(raw_data, args):
    _num = raw_data.size(0) // (args.batch_size * args.bptt)
    raw_data = raw_data[:(_num * args.batch_size * args.bptt)]
    return raw_data


def evaluate(data_source, model, vocab, ntokens, criterion, args, device, test = False, epoch = None, writer = None):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    if test:
        total_loss = 0. # original
    else: 
        stats = adaptdl.torch.Accumulator() # Changed in step 5

    mask_id = vocab.stoi['<MASK>']
    cls_id = vocab.stoi['<cls>']

    if test:
        dataloader = DataLoader(data_source, batch_size=args.batch_size * args.bptt, # original
                                shuffle=False, collate_fn=lambda b: collate_batch(b, args, mask_id, cls_id)) # original
    else:
        dataloader = adaptdl.torch.AdaptiveDataLoader(data_source, drop_last=True, batch_size=args.batch_size * args.bptt,
                                shuffle=False, collate_fn=lambda b: collate_batch(b, args, mask_id, cls_id)) # Changed in step 2

    with torch.no_grad():
        for batch, (data, lm_mask, targets) in enumerate(dataloader):
            if args.parallel == 'DDP':
                data = data.to(device[0])
                targets = targets.to(device[0])
            else:
                data = data.to(device)
                targets = targets.to(device)
            data = data.transpose(0, 1)  # Wrap up by DDP or DataParallel
            output = model(data)
            output = torch.stack([output[i] for i in range(lm_mask.size(0)) if lm_mask[i]])
            output_flat = output.view(-1, ntokens)
            if test:
                total_loss += criterion(output_flat, targets).item()
            else:
                stats['test_loss'] += criterion(output_flat, targets).item() # Changed in step 5
                stats['total'] += targets.size(0)

    if test:
        return total_loss / ((len(data_source) - 1) / args.bptt / args.batch_size)


    with stats.synchronized(): # Changed in step 5
        test_loss = stats['test_loss'] / (stats['total'] / args.bptt) # Changed in step 5
        writer.add_scalar("Loss/valid", test_loss, epoch) # Added for tensorboard

    return test_loss # Changed in step 5


def train(model, vocab, train_loss_log, train_data,
          optimizer, criterion, ntokens, epoch, scheduler, args, device, rank=None, batch_size_log = None, writer = None):
    model.train()
    total_loss = 0
    start_time = time.time()
    mask_id = vocab.stoi['<MASK>']
    cls_id = vocab.stoi['<cls>']
    train_loss_log.append(0.0)
    # dataloader = DataLoader(train_data, batch_size=args.batch_size * args.bptt, # original
    #                         shuffle=False, collate_fn=lambda b: collate_batch(b, args, mask_id, cls_id)) # original

    dataloader = adaptdl.torch.AdaptiveDataLoader(train_data, drop_last=True, batch_size=args.batch_size * args.bptt, 
                            shuffle=False, collate_fn=lambda b: collate_batch(b, args, mask_id, cls_id)) # Changed in step 2

    dataloader.autoscale_batch_size(16 * args.batch_size * args.bptt, local_bsz_bounds=(args.batch_size * args.bptt / 4, 4 * args.batch_size * args.bptt)) # Changed in step 3
    # dataloader.autoscale_batch_size(16 * args.batch_size * args.bptt, local_bsz_bounds=(args.batch_size * args.bptt, args.batch_size * args.bptt))

    for batch, (data, lm_mask, targets) in enumerate(dataloader):
        optimizer.zero_grad()
        if args.parallel == 'DDP':
            print("DDP")
            data = data.to(device[0])
            targets = targets.to(device[0])
        else:
            data = data.to(device)
            targets = targets.to(device)
        data = data.transpose(0, 1)  # Wrap up by DDP or DataParallel
        output = model(data)
        output = torch.stack([output[i] for i in range(lm_mask.size(0)) if lm_mask[i]])
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()
        total_loss += loss.item()
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss / args.log_interval
            cur_bs = dataloader.current_batch_size
            elapsed = time.time() - start_time
            if (rank is None) or rank == 0:
                train_loss_log[-1] = cur_loss
                print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:05.5f} | ms/batch {:5.2f} | '
                      'loss {:5.2f} | ppl {:8.2f} | batch_size {:5d}'.format(epoch, batch,
                                                          # len(train_data) // (args.bptt * args.batch_size),
                                                          len(train_data) // dataloader.current_batch_size,
                                                          scheduler.get_last_lr()[0],
                                                          elapsed * 1000 / args.log_interval,
                                                          cur_loss, math.exp(cur_loss),
                                                          dataloader.current_batch_size))
            total_loss = 0
            start_time = time.time()
    writer.add_scalar("Loss/train", cur_loss, epoch) # Added for tensorboard
    writer.add_scalar("Throughput/Gain", model.gain, epoch) # Added for tensorboard
    writer.add_scalar("Throughput/Global_Batchsize", cur_bs, epoch) # Added for tensorboard
    batch_size_log.append(dataloader.current_batch_size)


def run_main(args, rank=None):
    torch.manual_seed(args.seed)
    if args.parallel == 'DDP':
        n = torch.cuda.device_count() // args.world_size
        device = list(range(rank * n, (rank + 1) * n))
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    import torchtext
    if args.dataset == 'WikiText103':
        from torchtext.experimental.datasets import WikiText103 as WLMDataset
    elif args.dataset == 'WikiText2':
        from torchtext.experimental.datasets import WikiText2 as WLMDataset
    elif args.dataset == 'WMTNewsCrawl':
        from data import WMTNewsCrawl as WLMDataset
    elif args.dataset == 'EnWik9':
        from torchtext.datasets import EnWik9
    elif args.dataset == 'BookCorpus':
        from data import BookCorpus
    else:
        print("dataset for MLM task is not supported")

    try:
        vocab = torch.load(args.save_vocab)
    except:
        train_dataset, test_dataset, valid_dataset = WLMDataset()
        old_vocab = train_dataset.vocab
        vocab = torchtext.vocab.Vocab(counter=old_vocab.freqs,
                                      specials=['<unk>', '<pad>', '<MASK>'])
        with open(args.save_vocab, 'wb') as f:
            torch.save(vocab, f)

    if args.dataset == 'WikiText103' or args.dataset == 'WikiText2':
        train_dataset, test_dataset, valid_dataset = WLMDataset(vocab=vocab)
    elif args.dataset == 'WMTNewsCrawl':
        from torchtext.experimental.datasets import WikiText2
        test_dataset, valid_dataset = WikiText2(vocab=vocab, data_select=('test', 'valid'))
        train_dataset, = WLMDataset(vocab=vocab, data_select='train')
    elif args.dataset == 'EnWik9':
        enwik9 = EnWik9()
        idx1, idx2 = int(len(enwik9) * 0.8), int(len(enwik9) * 0.9)
        train_data = torch.tensor([vocab.stoi[_id]
                                  for _id in enwik9[0:idx1]]).long()
        val_data = torch.tensor([vocab.stoi[_id]
                                 for _id in enwik9[idx1:idx2]]).long()
        test_data = torch.tensor([vocab.stoi[_id]
                                 for _id in enwik9[idx2:]]).long()
        from torchtext.experimental.datasets import LanguageModelingDataset
        train_dataset = LanguageModelingDataset(train_data, vocab)
        valid_dataset = LanguageModelingDataset(val_data, vocab)
        test_dataset = LanguageModelingDataset(test_data, vocab)
    elif args.dataset == 'BookCorpus':
        train_dataset, test_dataset, valid_dataset = BookCorpus(vocab)

    train_data = process_raw_data(train_dataset.data, args)
    # if rank is not None:
    #     # Chunk training data by rank for different gpus
    #     chunk_len = len(train_data) // args.world_size
    #     train_data = train_data[(rank * chunk_len):((rank + 1) * chunk_len)]
    val_data = process_raw_data(valid_dataset.data, args)
    test_data = process_raw_data(test_dataset.data, args)

    ntokens = len(train_dataset.get_vocab())
    if args.checkpoint != 'None':
        model = torch.load(args.checkpoint)
    else:
        model = MLMTask(ntokens, args.emsize, args.nhead, args.nhid, args.nlayers, args.dropout)
    if args.parallel == 'DDP':
        model = model.to(device[0])
        model = DDP(model, device_ids=device)
    else:
        model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.1)

    adaptdl.torch.init_process_group("nccl" if torch.cuda.is_available()else "gloo") # Changed
    model = adaptdl.torch.AdaptiveDataParallel(model, optimizer, scheduler) # Changed

    best_val_loss = None
    train_loss_log, val_loss_log, batch_size_log = [], [], []

    tensorboard_dir = os.path.join(os.getenv("ADAPTDL_TENSORBOARD_LOGDIR", "/tmp") if adaptdl.env.replica_rank() == 0 else "/tmp", adaptdl.env.job_id())

    writer =  SummaryWriter(tensorboard_dir)# Added for tensorboard

    # for epoch in range(1, args.epochs + 1): # original
    for epoch in adaptdl.torch.remaining_epochs_until(args.epochs): # Changed
        epoch_start_time = time.time()
        train(model, train_dataset.vocab, train_loss_log, train_data,
              optimizer, criterion, ntokens, epoch, scheduler, args, device, rank, batch_size_log, writer)
        val_loss = evaluate(val_data, model, train_dataset.vocab, ntokens, criterion, args, device , test = False, epoch = epoch, writer = writer) # Changed for tensorboard
        if (rank is None) or (rank == 0):
            val_loss_log.append(val_loss)
            print('-' * 89)
            print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                  'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                             val_loss, math.exp(val_loss)))
            print('-' * 89)
        if not best_val_loss or val_loss < best_val_loss:
            if rank is None:
                with open(args.save, 'wb') as f:
                    torch.save(model.module, f)
            elif rank == 0:
                with open(args.save, 'wb') as f:
                    torch.save(model.module.state_dict(), f)
            best_val_loss = val_loss
        else:
            scheduler.step()
    if args.parallel == 'DDP':
        dist.barrier()
        rank0_devices = [x - rank * len(device) for x in device]
        device_pairs = zip(rank0_devices, device)
        map_location = {'cuda:%d' % x: 'cuda:%d' % y for x, y in device_pairs}
        model.load_state_dict(
            torch.load(args.save, map_location=map_location))
        model = adaptdl.torch.AdaptiveDataParallel(model, optimizer, scheduler) # Changed
        test_loss = evaluate(test_data, model, train_dataset.vocab, ntokens, criterion, args, device)
        if rank == 0:
            wrap_up(train_loss_log, val_loss_log, test_loss, args, model.module, 'mlm_loss.txt', 'full_mlm_model.pt', batch_size_log)
    else:
        with open(args.save, 'rb') as f:
            model.module.load_state_dict(torch.load(f)) # Changed
            # model = torch.load(f)
        # model = adaptdl.torch.AdaptiveDataParallel(model, optimizer, scheduler) # Changed
        test_loss = evaluate(test_data, model, train_dataset.vocab, ntokens, criterion, args, device, True)
        wrap_up(train_loss_log, val_loss_log, test_loss, args, model.module, 'mlm_loss.txt', 'full_mlm_model.pt')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Wikitext-2 Transformer Language Model')
    parser.add_argument('--emsize', type=int, default=768,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=3072,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=12,
                        help='number of layers')
    parser.add_argument('--nhead', type=int, default=12,
                        help='the number of heads in the encoder/decoder of the transformer model')
    parser.add_argument('--lr', type=float, default=6,
                        help='initial learning rate')
    parser.add_argument('--clip', type=float, default=0.1,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=8,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                        help='batch size')
    parser.add_argument('--bptt', type=int, default=128,
                        help='sequence length')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--seed', type=int, default=5431916812,
                        help='random seed')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='report interval')
    parser.add_argument('--checkpoint', type=str, default='None',
                        help='path to load the checkpoint')
    parser.add_argument('--save', type=str, default='mlm_bert.pt',
                        help='path to save the final model')
    parser.add_argument('--save-vocab', type=str, default='torchtext_bert_vocab.pt',
                        help='path to save the vocab')
    parser.add_argument('--mask_frac', type=float, default=0.15,
                        help='the fraction of masked tokens')
    parser.add_argument('--dataset', type=str, default='WikiText2',
                        help='dataset used for MLM task')
    parser.add_argument('--parallel', type=str, default='None',
                        help='Use DataParallel to train model')
    parser.add_argument('--world_size', type=int, default=8,
                        help='the world size to initiate DPP')
    args = parser.parse_args()

    if args.parallel == 'DDP':
        run_demo(run_ddp, run_main, args)
    else:
        run_main(args, adaptdl.env.replica_rank())
        time.sleep(100)
