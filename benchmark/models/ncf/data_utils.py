import numpy as np 
import pandas as pd 
import scipy.sparse as sp

import torch.utils.data as data

import config


def load_all(test_num=100):
	""" We load all the three file here to save time in each epoch. """
	train_data = pd.read_csv(
		config.train_rating, 
		sep='\t', header=None, names=['user', 'item'], 
		usecols=[0, 1], dtype={0: np.int32, 1: np.int32})

	user_num = train_data['user'].max() + 1
	item_num = train_data['item'].max() + 1

	# load ratings as a dense matrix
	train_mat = np.zeros((user_num, item_num))
	train_mat[train_data['user'], train_data['item']] = 1.0

	train_data = train_data.values.tolist()

	# load ratings as a dok matrix
	#train_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)
	#for x in train_data:
	#	train_mat[x[0], x[1]] = 1.0

	test_data = []
	with open(config.test_negative, 'r') as fd:
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
		if train_mat is not None:
			self.features_ng = [np.where(row == 0.0)[0] for row in train_mat]

	def ng_sample(self):
		assert self.is_training, 'no need to sampling when testing'
		self.features_fill = list(self.features_ps)
		self.labels_fill = [1] * len(self.features_ps)
		for u, items_ng in enumerate(self.features_ng):
			n = self.num_ng * (self.num_item - items_ng.size)
			self.features_fill.extend(
				[(u, i) for i in np.random.choice(items_ng, n)])
			self.labels_fill.extend([0] * n)
		assert len(self.features_fill) == len(self.labels_fill) == len(self)
		#for x in self.features_ps:
		#	u = x[0]
		#	self.np.random.choice(self.features_ng[u], self.num_ng)
		#	for t in range(self.num_ng):
		#		j = np.random.randint(self.num_item)
		#		while (u, j) in self.train_mat:
		#			j = np.random.randint(self.num_item)
		#		self.features_ng.append([u, j])
		#labels_ps = [1 for _ in range(len(self.features_ps))]
		#labels_ng = [0 for _ in range(len(self.features_ng))]
		#self.features_fill = self.features_ps + self.features_ng
		#self.labels_fill = labels_ps + labels_ng

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
		
