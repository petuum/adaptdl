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


import collections
import math

import pytest
import torch
import torchtext
from torch.utils.data import TensorDataset
from torchtext.data.utils import get_tokenizer

from adaptdl.conftest import elastic_multiprocessing
from adaptdl.torch.data import (ElasticSampler, AdaptiveDataLoader,
                                current_dataloader)
from adaptdl.torch.iterator import AdaptiveBPTTIterator


@pytest.mark.parametrize("num_replicas", [1, 3, 5])
@pytest.mark.parametrize("dataset_size", [9, 15, 25])
def test_sampler_epoch(num_replicas, dataset_size,
                       epoch=0, index=0, shuffle=True):
    dataset = TensorDataset(torch.rand(dataset_size))
    sampler = ElasticSampler(dataset, shuffle=shuffle)
    sampler.num_replicas = num_replicas
    sampler.set_epoch(epoch, index)
    epoch_samples = []
    sample_counts = collections.Counter()
    for rank in range(num_replicas):
        sampler.rank = rank
        epoch_samples.append(list(sampler))
        # Check indices are split evenly between replicas.
        assert len(sampler) == \
            math.ceil((dataset_size - index % dataset_size) / num_replicas)
        # Check the actual samples obey the length.
        assert len(sampler) == len(epoch_samples[rank])
        # Check ordering is the same within the same epoch.
        assert list(sampler) == epoch_samples[rank]
        sample_counts.update(epoch_samples[rank])
    # Check all indices are present.
    assert len(sample_counts) >= dataset_size - index % dataset_size
    assert all(0 <= key < dataset_size for key in sample_counts)
    # Check each index is counted roughly the same number of times.
    assert max(sample_counts.values()) - min(sample_counts.values()) <= 1
    return epoch_samples


@pytest.mark.parametrize("num_replicas", [1, 3, 5])
@pytest.mark.parametrize("dataset_size", [9, 15, 25])
def test_sampler_shuffle(num_replicas, dataset_size):
    epoch0_samples = test_sampler_epoch(num_replicas, dataset_size, epoch=0)
    epoch1_samples = test_sampler_epoch(num_replicas, dataset_size, epoch=1)
    assert epoch0_samples != epoch1_samples  # Shuffle is on.
    epoch0_samples = test_sampler_epoch(num_replicas, dataset_size,
                                        epoch=0, shuffle=False)
    epoch1_samples = test_sampler_epoch(num_replicas, dataset_size,
                                        epoch=1, shuffle=False)
    assert epoch0_samples == epoch1_samples  # Shuffle is off.


@pytest.mark.parametrize("num_replicas", [1, 3, 5])
@pytest.mark.parametrize("dataset_size", [9, 15, 25])
def test_sampler_index(num_replicas, dataset_size):
    index = dataset_size // 2  # Set index to halfway through the dataset.
    epoch_samples = test_sampler_epoch(num_replicas, dataset_size,
                                       index=index, shuffle=False)
    samples = sum(epoch_samples, [])
    # Check contains second half of dataset.
    for idx in range(index, dataset_size):
        assert idx in samples

    index = 2 * dataset_size  # Test sampler wrap-around.
    epoch_samples = test_sampler_epoch(num_replicas, dataset_size,
                                       index=index, shuffle=False)
    assert set(sum(epoch_samples, [])) == set(range(dataset_size))


@elastic_multiprocessing
def test_dataloader_restarts():
    import adaptdl.checkpoint
    import adaptdl.collective
    from adaptdl.env import num_restarts, num_replicas
    adaptdl.collective.initialize("0.0.0.0")
    dataset_size = 100
    init_batch_size = 10
    dataset = TensorDataset(torch.rand(dataset_size))
    dataloader = AdaptiveDataLoader(dataset, batch_size=init_batch_size)
    # Load data samples in the following order:
    # 2 batches (20 samples) using 1 replica (local_bsz = 10, batch_size = 10)
    # 5 batches (60 samples) using 4 replica (local_bsz = 3, batch_size = 12)
    # 2 batches (20 samples) using 2 replica (local_bsz = 5, batch_size = 10)
    assert current_dataloader() is None
    for idx, batch in enumerate(dataloader):
        if num_restarts() == 0 and idx == 2:
            adaptdl.checkpoint.save_all_states()
            return 4  # Restart with 4 replicas.
        if num_restarts() == 1 and idx == 5:
            adaptdl.checkpoint.save_all_states()
            return 2  # Restart with 2 replicas.
        assert current_dataloader() is dataloader._elastic
        local_bsz = batch[0].size(0)
        assert dataloader.current_local_bsz == local_bsz
        assert local_bsz == math.ceil(init_batch_size / num_replicas())
        assert dataloader.current_batch_size == num_replicas() * local_bsz
    # After the last 2 batches.
    assert idx == 1


@elastic_multiprocessing
def test_dataloader_break():
    import adaptdl.checkpoint
    import adaptdl.collective
    from adaptdl.env import num_restarts
    if num_restarts() == 0:
        return 2
    adaptdl.collective.initialize("0.0.0.0")
    dataset = TensorDataset(torch.rand(100))
    dataloader = AdaptiveDataLoader(dataset, batch_size=10)
    # Break in the middle of the first for-loop, and start another for-loop.
    assert current_dataloader() is None
    for idx, batch in enumerate(dataloader):
        assert current_dataloader() is dataloader._elastic
        if idx == 5:
            break
    assert current_dataloader() is None
    for idx, batch in enumerate(dataloader):
        assert current_dataloader() is dataloader._elastic
    assert idx == 9  # Run 10 batches total.


@elastic_multiprocessing
def test_bptt_iterator():
    import adaptdl.checkpoint
    import adaptdl.collective
    from adaptdl.env import num_restarts
    adaptdl.collective.initialize("0.0.0.0")
    # Load the iterator with 500 words
    # 1 batch (5x10) using 1 replica. Restart after one iteration.
    # 9 batches (5x5) using 2 replicas to consume remaining batches.
    TEXT = torchtext.data.Field(tokenize=get_tokenizer("basic_english"),
                                init_token='<sos>',
                                eos_token='<eos>')
    fields = [('text', TEXT)]
    examples = [torchtext.data.Example.fromlist([['The'] * 500], fields)]
    dataset = torchtext.data.Dataset(examples, fields)
    TEXT.build_vocab(dataset)
    bptt_iter = AdaptiveBPTTIterator(dataset, batch_size=10, bptt_len=5)
    for idx, batch in enumerate(bptt_iter):
        if num_restarts() == 0 and idx == 1:
            assert batch.text.shape == (5, 10)
            adaptdl.checkpoint.save_all_states()
            return 2
        if adaptdl.env.num_replicas() == 2:
            assert batch.text.shape == (5, 5) or batch.text.shape == (4, 5)
    if adaptdl.env.num_replicas() == 2:
        assert idx == 8
