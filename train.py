#!/usr/bin/env python
#
# CS224W Fall 2019-2020
# @Jason Zheng
#

import argparse
import models
import networkx as nx
import numpy as np
import pandas as pd
import pickle as pkl
from sklearn.model_selection import StratifiedShuffleSplit
import time
import torch
import torch_geometric.data as pyg_d
import torch_geometric.nn as pyg_nn
import torch.optim as optim

from collections import defaultdict
from torch_geometric.datasets import TUDataset
from torch_geometric.datasets import Planetoid
from torch_geometric.data import DataLoader

NUM_NODES = 100386

# ==============================================================================
# Utils
# ==============================================================================
def local_arg_parse():
	opt_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	opt_parser.add_argument('--opt-restart', dest='opt_restart', type=int, default=0,
			help='Number of epochs before restart (by default set to 0 which means no restart)')
	opt_parser.add_argument('--opt-decay-step', dest='opt_decay_step', type=int,
			help='Number of epochs before decay')
	opt_parser.add_argument('--opt-decay-rate', dest='opt_decay_rate', type=float,
			help='Learning rate decay ratio')
	opt_parser.add_argument('--clip', dest='clip', type=float,
			help='Gradient clipping.')
	opt_parser.add_argument('--batch_size', type=int, default=32)
	opt_parser.add_argument('--hidden_dim', type=int, default=32)
	opt_parser.add_argument('--model_type', type=str, default='GraphSage')
	opt_parser.add_argument('--num_layers', type=int, default=2)
	opt_parser.add_argument('--dropout', type=float, default=0.5)
	opt_parser.add_argument('--epochs', type=int, default=500)
	opt_parser.add_argument('--weight_decay', type=float, default=5e-3)
	opt_parser.add_argument('--lr', type=float, default=0.01)
	opt_parser.add_argument('--opt', type=str, default='adam')
	opt_parser.add_argument('--opt_scheduler', type=str, default='none')
	opt_parser.add_argument('--flag_index', type=str, default='hate')
	args = opt_parser.parse_args()
	return args

def load_hate(features='hate/users_hate_all.content', edges='hate/users.edges', num_features=320):
	"""
	Returns:
		NUM_NODES x num_features matrix of features
		NUM_NODES x 1 matrix of labels (which are printed out)
		adjacency list of the (directed) edges file (each line being n1 n2 representing n1 -> n2)
			as a dictionary of n1 to n2.
	"""
	num_feats = num_features
	feat_data = torch.zeros((NUM_NODES, num_feats))
	labels = torch.empty((NUM_NODES, 1), dtype=torch.long)
	node_map = {}
	label_map = {}

	with open(features) as fp:
		for i, line in enumerate(fp):
			info = line.strip().split()
			feat_data[i, :] = torch.tensor(list(map(float, info[1:-1])))
			node_map[info[0]] = i
			if not info[-1] in label_map:
				label_map[info[-1]] = len(label_map)
			labels[i] = label_map[info[-1]]

	adj_lists = defaultdict(set)
	edge_list = list()
	with open(edges) as fp:
		for i, line in enumerate(fp):
			info = line.strip().split()
			paper1 = node_map[info[0]]
			paper2 = node_map[info[1]]
			edge_list.append([paper1, paper2])
			adj_lists[paper1].add(paper2)
			adj_lists[paper2].add(paper1)

	edge_tensor = torch.tensor(edge_list, dtype=torch.long)
	labels = labels.squeeze()
	print('Label meanings: ', label_map)
	return feat_data, labels, adj_lists, edge_tensor

def get_stratified_batches():
	if args.flag_index == "hate":
		df = pd.read_csv("hate/users_anon.csv")
		df = df[df.hate != "other"]
		y = torch.tensor([1 if v == "hateful" else 0 for v in df["hate"].values])
		x = torch.tensor(df["user_id"].values)
		del df

	else:
		df = pd.read_csv("suspended/users_anon.csv")
		np.random.seed(321)
		df2 = df[df["is_63_2"] == True].sample(668, axis=0)
		df3 = df[df["is_63_2"] == False].sample(5405, axis=0)
		df = pd.concat([df2, df3])
		y = torch.tensor([1 if v else 0 for v in df["is_63_2"].values])
		x = torch.tensor(df["user_id"].values)
		del df, df2, df3

	skf = StratifiedShuffleSplit(n_splits=10, test_size=0.2, train_size=0.8, random_state=123)  # Assuming train-test ratio of 0.8
	return skf, x, y

def get_datasets():
	"""
	Gets the dataset for hateful Twitter users
	"""
	# mask not multiple datasets
	feat_data, labels, adj_lists, edges = load_hate()

	dataset = pyg_d.Data(x=feat_data, edge_index=edges, y=labels, batch=feat_data[:,0])
	return dataset

def build_optimizer(args, params):
	weight_decay = args.weight_decay
	filter_fn = filter(lambda p : p.requires_grad, params)
	if args.opt == 'adam':
		optimizer = optim.Adam(filter_fn, lr=args.lr, weight_decay=weight_decay)
	elif args.opt == 'sgd':
		optimizer = optim.SGD(filter_fn, lr=args.lr, momentum=0.95, weight_decay=weight_decay)
	elif args.opt == 'rmsprop':
		optimizer = optim.RMSprop(filter_fn, lr=args.lr, weight_decay=weight_decay)
	elif args.opt == 'adagrad':
		optimizer = optim.Adagrad(filter_fn, lr=args.lr, weight_decay=weight_decay)
	if args.opt_scheduler == 'none':
		return None, optimizer
	elif args.opt_scheduler == 'step':
		scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.opt_decay_step, gamma=args.opt_decay_rate)
	elif args.opt_scheduler == 'cos':
		scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.opt_restart)
	return scheduler, optimizer


# ==============================================================================
# Train
# ==============================================================================
def train(dataset, args):
	# build model
	model = models.GNNStack(
		320, # dataset.num_node_features
		args.hidden_dim,  # args.hidden_dim
		3,  # dataset.num_classes
		args
	)
	scheduler, opt = build_optimizer(args, model.parameters())
	skf, x, y = get_stratified_batches()

	# train
	validation_accuracy = list()
	for epoch in range(args.epochs):
		total_loss = 0
		avg_test_acc = 0
		model.train()
		# No need to loop over batches since we only have one batch
		num_splits = 0
		for train_indices, test_indices in skf.split(x,y):
			batch = dataset
			opt.zero_grad()
			pred = model(batch)
			label = batch.y

			pred = pred[train_indices]
			label = label[train_indices]

			loss = model.loss(pred, label)
			loss.backward()
			opt.step()
			total_loss += loss.item()
			num_splits += 1

			if epoch % 2 == 0:
				test_acc = test(dataset, model, test_indices)
				avg_test_acc += test_acc
		total_loss /= num_splits
		if epoch % 2 == 0:
			avg_test_acc /= num_splits
			print(test_acc,   '  test')
		print(total_loss)


def test(dataset, model, test_indices):
	model.eval()

	correct = 0
	with torch.no_grad():
		pred = model(dataset).max(dim=1)[1]  # max(dim=1) returns values, indices tuple; only need indices
		label = dataset.y

	pred = pred[test_indices]
	label = dataset.y[test_indices]

	correct = pred.eq(label).sum().item()
	return correct


# ==============================================================================
# Controller
# ==============================================================================
if __name__ == "__main__":
	args = local_arg_parse()

	dataset = get_datasets()
	train(dataset, args)
