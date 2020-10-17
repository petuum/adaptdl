# Code adapted from https://github.com/guoyang9/NCF

import numpy as np 
import pandas as pd 
import scipy.sparse as sp
import os
import torch.utils.data as data
import hashlib

import adaptdl.env
from urllib.request import urlretrieve


base_url = "https://raw.githubusercontent.com/hexiangnan/neural_collaborative_filtering/312aff0ee6f4df0ba60041329a74ed29bc7ac4b4/Data/"

def download_all(root_path, dataset="ml-1m"):
    datafiles = ['{}.{}'.format(dataset, suffix) for suffix in
                 ['train.rating', 'test.rating', 'test.negative']]
    for filename in datafiles:
        local_filename = os.path.join(root_path, filename)
        # Todo: verify file
        if not os.path.exists(local_filename):
            print(base_url + filename, local_filename)
            urlretrieve(base_url + filename, local_filename)


def load_all(root_path, train_rating, test_negative, test_num=100):
    """ We load all the three file here to save time in each epoch. """
    download_all(root_path)
    train_data = pd.read_csv(
        train_rating, 
        sep='\t', header=None, names=['user', 'item'], 
        usecols=[0, 1], dtype={0: np.int32, 1: np.int32})

    user_num = train_data['user'].max() + 1
    item_num = train_data['item'].max() + 1

    train_data = train_data.values.tolist()

    # load ratings as a dok matrix
    train_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)
    for x in train_data:
        train_mat[x[0], x[1]] = 1.0

    test_data = []
    with open(test_negative, 'r') as fd:
        line = fd.readline()
        while line != None and line != '':
            arr = line.split('\t')
            u = eval(arr[0])[0]
            test_data.append([u, eval(arr[0])[1]])
            for i in arr[1:]:
                test_data.append([u, int(i)])
            line = fd.readline()
    return train_data, test_data, user_num, item_num, train_mat


class NCFData(data.Dataset):
    def __init__(self, features, 
                num_item, train_mat=None, num_ng=0, is_training=None):
        super(NCFData, self).__init__()
        """ Note that the labels are only useful when training, we thus 
            add them in the ng_sample() function.
        """
        self.features_ps = features
        self.num_item = num_item
        self.train_mat = train_mat
        self.num_ng = num_ng
        self.is_training = is_training
        self.labels = [0 for _ in range(len(features))]

    def ng_sample(self):
        assert self.is_training, 'no need to sampling when testing'

        self.features_ng = []
        for x in self.features_ps:
            u = x[0]
            for t in range(self.num_ng):
                j = np.random.randint(self.num_item)
                while (u, j) in self.train_mat:
                    j = np.random.randint(self.num_item)
                self.features_ng.append([u, j])

        labels_ps = [1 for _ in range(len(self.features_ps))]
        labels_ng = [0 for _ in range(len(self.features_ng))]

        self.features_fill = self.features_ps + self.features_ng
        self.labels_fill = labels_ps + labels_ng

    def __len__(self):
        return (self.num_ng + 1) * len(self.labels)

    def __getitem__(self, idx):
        features = self.features_fill if self.is_training \
                    else self.features_ps
        labels = self.labels_fill if self.is_training \
                    else self.labels

        user = features[idx][0]
        item = features[idx][1]
        label = labels[idx]
        return user, item ,label
        
