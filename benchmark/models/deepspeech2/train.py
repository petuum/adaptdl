import argparse
import json
import os
import random
import time
import traceback

import numpy as np
import torch
import torch.nn
from torch.utils.tensorboard import SummaryWriter

from data.data_loader import AudioDataLoader, SpectrogramDataset, BucketingSampler, DistributedBucketingSampler
from decoder import GreedyDecoder
from logger import VisdomLogger, TensorBoardLogger
from model import DeepSpeech, supported_rnns

import adaptdl
import adaptdl.torch
from adaptdl.torch._metrics import report_train_metrics, report_valid_metrics

parser = argparse.ArgumentParser(description='DeepSpeech training')
parser.add_argument('--train-manifest', metavar='DIR',
                    help='path to train manifest csv', default='data/train_manifest.csv')
parser.add_argument('--val-manifest', metavar='DIR',
                    help='path to validation manifest csv', default='data/val_manifest.csv')
parser.add_argument('--sample-rate', default=16000, type=int, help='Sample rate')
parser.add_argument('--batch-size', default=20, type=int, help='Batch size for training')
parser.add_argument('--num-workers', default=4, type=int, help='Number of workers used in data-loading')
parser.add_argument('--labels-path', default='labels.json', help='Contains all characters for transcription')
parser.add_argument('--window-size', default=.02, type=float, help='Window size for spectrogram in seconds')
parser.add_argument('--window-stride', default=.01, type=float, help='Window stride for spectrogram in seconds')
parser.add_argument('--window', default='hamming', help='Window type for spectrogram generation')
parser.add_argument('--hidden-size', default=800, type=int, help='Hidden size of RNNs')
parser.add_argument('--hidden-layers', default=5, type=int, help='Number of RNN layers')
parser.add_argument('--rnn-type', default='gru', help='Type of the RNN. rnn|gru|lstm are supported')
parser.add_argument('--epochs', default=80, type=int, help='Number of training epochs')
parser.add_argument('--cuda', dest='cuda', action='store_true', help='Use cuda to train model')
parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float, help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--max-norm', default=400, type=int, help='Norm cutoff to prevent explosion of gradients')
parser.add_argument('--learning-anneal', default=1.1, type=float, help='Annealing applied to learning rate every epoch')
parser.add_argument('--speed-volume-perturb', dest='speed_volume_perturb', action='store_true', help='Use random tempo and gain perturbations.')
parser.add_argument('--spec-augment', dest='spec_augment', action='store_true', help='Use simple spectral augmentation on mel spectograms.')
parser.add_argument('--noise-dir', default=None,
                    help='Directory to inject noise into audio. If default, noise Inject not added')
parser.add_argument('--noise-prob', default=0.4, help='Probability of noise being added per sample')
parser.add_argument('--noise-min', default=0.0,
                    help='Minimum noise level to sample from. (1.0 means all noise, not original signal)', type=float)
parser.add_argument('--noise-max', default=0.5,
                    help='Maximum noise levels to sample from. Maximum 1.0', type=float)
parser.add_argument('--no-shuffle', dest='no_shuffle', action='store_true',
                    help='Turn off shuffling and sample from dataset based on sequence length (smallest to largest)')
parser.add_argument('--no-sortaGrad', dest='no_sorta_grad', action='store_true',
                    help='Turn off ordering of dataset on sequence length for the first epoch.')
parser.add_argument('--no-bidirectional', dest='bidirectional', action='store_false', default=True,
                    help='Turn off bi-directional RNNs, introduces lookahead convolution')
parser.add_argument('--seed', default=None, type=int, help='Seed to generators')


def evaluate(test_loader, device, model, decoder, target_decoder):
    model.eval()
    stats_valid = adaptdl.torch.Accumulator()
    for data in test_loader:
        inputs, targets, input_sizes, target_sizes = data
        inputs = inputs.to(device)
        input_sizes = input_sizes.to(device)
        # unflatten targets
        split_targets = []
        offset = 0
        for size in target_sizes:
            split_targets.append(targets[offset:offset + size])
            offset += size

        out, output_sizes = model(inputs, input_sizes)

        decoded_output, _ = decoder.decode(out, output_sizes)
        target_strings = target_decoder.convert_to_strings(split_targets)

        for x in range(len(target_strings)):
            transcript, reference = decoded_output[x][0], target_strings[x][0]
            stats_valid["wer_total"] += decoder.wer(transcript, reference)
            stats_valid["cer_total"] += decoder.cer(transcript, reference)
            stats_valid["num_tokens"] += len(reference.split())
            stats_valid["num_chars"] += len(reference.replace(' ', ''))

    with stats_valid.synchronized():
        print(stats_valid)
        # wer is defined in evaluate() as 100*(total word errors in test batch)/(total words in test batch)
        stats_valid["wer"] = stats_valid["wer_total"] / stats_valid["num_tokens"] * 100
        # cer is defined in evaluate() as 100*(total character errors in test batch)/(total characters in test batch)
        stats_valid["cer"] = stats_valid["cer_total"] / stats_valid["num_chars"] * 100
        writer.add_scalar("WER/Valid", stats_valid["wer"], epoch)
        writer.add_scalar("CER/Valid", stats_valid["cer"], epoch)
        report_valid_metrics(epoch, 0, wer=stats_valid["wer"], cer=stats_valid["cer"])
        return stats_valid["wer"], stats_valid["cer"]


if __name__ == '__main__':
    args = parser.parse_args()

    if args.seed is not None:
        # Set seeds for determinism
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)

    device = torch.device("cuda" if args.cuda else "cpu")

    with open(args.labels_path) as label_file:
        labels = str(''.join(json.load(label_file)))

    audio_conf = dict(sample_rate=args.sample_rate,
                      window_size=args.window_size,
                      window_stride=args.window_stride,
                      window=args.window,
                      noise_dir=args.noise_dir,
                      noise_prob=args.noise_prob,
                      noise_levels=(args.noise_min, args.noise_max))

    rnn_type = args.rnn_type.lower()
    assert rnn_type in supported_rnns, "rnn_type should be either lstm, rnn or gru"
    model = DeepSpeech(rnn_hidden_size=args.hidden_size,
                       nb_layers=args.hidden_layers,
                       labels=labels,
                       rnn_type=supported_rnns[rnn_type],
                       audio_conf=audio_conf,
                       bidirectional=args.bidirectional)

    decoder = GreedyDecoder(labels)
    train_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.train_manifest, labels=labels,
                                       normalize=True, speed_volume_perturb=args.speed_volume_perturb, spec_augment=args.spec_augment)
    test_dataset = SpectrogramDataset(audio_conf=audio_conf, manifest_filepath=args.val_manifest, labels=labels,
                                      normalize=True, speed_volume_perturb=False, spec_augment=False)

    train_loader = AudioDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
                                   num_workers=args.num_workers)
    train_loader.autoscale_batch_size(640, local_bsz_bounds=(10, 80), gradient_accumulation=True)
    test_loader = AudioDataLoader(test_dataset, batch_size=args.batch_size,
                                  num_workers=args.num_workers)

    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum, nesterov=False)

    adaptdl.torch.init_process_group("nccl")
    model = adaptdl.torch.AdaptiveDataParallel(model, optimizer)
    model.adascale._smoothing = 0.997
    
    print(model)
    print("Number of parameters: %d" % DeepSpeech.get_param_size(model))

    criterion = torch.nn.CTCLoss()

    logdir = "/tmp" if adaptdl.env.replica_rank() else os.getenv("ADAPTDL_TENSORBOARD_LOGDIR", "/tmp")
    with SummaryWriter(logdir) as writer:
        for epoch in adaptdl.torch.remaining_epochs_until(args.epochs):
            model.train()
            stats_train = adaptdl.torch.Accumulator()
            for i, data in enumerate(train_loader):
                inputs, targets, input_sizes, target_sizes = data
                inputs = inputs.to(device)
                input_sizes = input_sizes.to(device)
                out, output_sizes = model(inputs, input_sizes)
                out = out.transpose(0, 1)  # TxNxH

                float_out = out.float()  # ensure float32 for loss
                loss = criterion(float_out, targets, output_sizes, target_sizes).to(device)

                loss_value = loss.item()

                loss.backward()
                if train_loader._elastic.is_sync_step():
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
                optimizer.step()

                # measure elapsed time
                print('Epoch: [{0}][{1}/{2}]\tLoss {loss:.4f}\t'
                      .format((epoch + 1), (i + 1), len(train_loader), loss=loss.item()))

                stats_train["train_loss_sum"] += loss.item() * inputs.size(0)
                stats_train["train_total"] += inputs.size(0)

                global_step = int(model.adascale._state["progress"])
                train_loader.to_tensorboard(writer, global_step, tag_prefix="AdaptDL/Data")
                model.to_tensorboard(writer, global_step, tag_prefix="AdaptDL/Model")

                del loss, out, float_out

            with stats_train.synchronized():
                stats_train["train_loss_avg"] = stats_train["train_loss_sum"] / stats_train["train_total"]
                writer.add_scalar("Loss/Train", stats_train["train_loss_avg"], epoch)
                report_train_metrics(epoch, stats_train["train_loss_avg"])
                print('Training Summary Epoch: [{0}]\t'
                      'Average Loss {loss:.3f}\t'.format(epoch + 1, loss=stats_train["train_loss_avg"]))

            with torch.no_grad():
                wer, cer, = evaluate(test_loader=test_loader,
                                     device=device,
                                     model=model,
                                     decoder=decoder,
                                     target_decoder=decoder)
                print('Validation Summary Epoch: [{0}]\t'
                      'Average WER {wer:.3f}\t'
                      'Average CER {cer:.3f}\t'.format(epoch + 1, wer=wer, cer=cer))

            # anneal lr
            for g in optimizer.param_groups:
                g['lr'] = g['lr'] / args.learning_anneal
            print('Learning rate annealed to: {lr:.6f}'.format(lr=g['lr']))
