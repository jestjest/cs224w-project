#!/usr/bin/env python
#
# CS224W Fall 2019-2020
# @Jason Zheng
#

import argparse
import models
import networkx as nx
import numpy as np
import pickle as pkl
import time
import torch
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
	opt_parser.add_argument('--opt', dest='opt', type=str,
			help='Type of optimizer')
	opt_parser.add_argument('--opt-scheduler', dest='opt_scheduler', type=str,
			help='Type of optimizer scheduler. By default none')
	opt_parser.add_argument('--opt-restart', dest='opt_restart', type=int,
			help='Number of epochs before restart (by default set to 0 which means no restart)')
	opt_parser.add_argument('--opt-decay-step', dest='opt_decay_step', type=int,
			help='Number of epochs before decay')
	opt_parser.add_argument('--opt-decay-rate', dest='opt_decay_rate', type=float,
			help='Learning rate decay ratio')
	opt_parser.add_argument('--lr', dest='lr', type=float,
			help='Learning rate.')
	opt_parser.add_argument('--clip', dest='clip', type=float,
			help='Gradient clipping.')
	opt_parser.add_argument('--weight_decay', type=float,
			help='Optimizer weight decay.')
	opt_parser.add_argument('--batch_size', type=int, default=32)
	# 'model_type': 'RGCN', 'dataset': 'enzymes', 'num_layers': 2, 'batch_size': 32, 'hidden_dim': 32, 'dropout': 0.5,
	 # 'epochs': 500, 'opt': 'adam', 'opt_scheduler': 'none', 'opt_restart': 0, 'weight_decay': 5e-3, 'lr': 0.01}
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
    feat_data = np.zeros((NUM_NODES, num_feats))
    labels = np.empty((NUM_NODES, 1), dtype=np.int64)
    node_map = {}
    label_map = {}

    with open(features) as fp:
        for i, line in enumerate(fp):
            info = line.strip().split()
            feat_data[i, :] = list(map(float, info[1:-1]))
            node_map[info[0]] = i
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)
            labels[i] = label_map[info[-1]]

    adj_lists = defaultdict(set)
    with open(edges) as fp:
        for i, line in enumerate(fp):
            info = line.strip().split()
            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)

    print('Label meanings: ', label_map)
    return feat_data, labels, adj_lists

def get_datasets():
	"""
	Gets the dataset for hateful Twitter users
	"""
	feat_data, labels, adj_lists = load_hate()
	print(feat_data.shape, labels.shape, len(adj_lists))
	train_dataset = []
	validation_dataset = []
	test_dataset = []
	return {
		'train': train_dataset,
		'validation': validation_dataset,
		'test': test_dataset
	}

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
def train(datasets, args):
	train_loader = DataLoader(datasets['train'], batch_size=args.batch_size, shuffle=True)
	validation_loader = DataLoader(datasets['validation'], batch_size=args.batch_size, shuffle=True)
	test_loader = DataLoader(datasets['test'], batch_size=args.batch_size, shuffle=True)

	# build model
	model = models.GNNStack(
		dataset.num_node_features,
		args.hidden_dim,
		dataset.num_classes,
		args,
	)
	scheduler, opt = build_optimizer(args, model.parameters())

	# train
	validation_accuracy = list()
	for epoch in range(args.epochs):
		total_loss = 0
		model.train()
		for batch in loader:
			opt.zero_grad()
			pred = model(batch)
			label = batch.y

			loss = model.loss(pred, label)
			loss.backward()
			opt.step()
			total_loss += loss.item() * batch.num_graphs
		total_loss /= len(loader.dataset)
		print(total_loss)

		if epoch % 10 == 0:
			test_acc = test(test_loader, model)
			validation_accuracy.append(test_acc)
			print(test_acc,   '  test')
	save_to = 'val_%s.pickle' % (args.model_type)
	with open(save_to, 'wb') as handle:
		pkl.dump(validation_accuracy, handle)


def test(loader, model, is_validation=False):
	model.eval()

	correct = 0
	for data in loader:
		with torch.no_grad():
			pred = model(data).max(dim=1)[1]  # max(dim=1) returns values, indices tuple; only need indices
			label = data.y
		correct += pred.eq(label).sum().item()
		total = len(loader.dataset)
	return correct / total


# ==============================================================================
# Controller
# ==============================================================================
if __name__ == "__main__":
	args = local_arg_parse()

	datasets = get_datasets()
	train(datasets, args)
