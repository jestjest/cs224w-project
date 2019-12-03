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
    args = parser.parse_args()
    return args


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
def train(datasets, task, args):
    train_loader = DataLoader(datasets['train'], batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(datasets['validation'], batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(datasets['test'], batch_size=args.batch_size, shuffle=True)

	# build model
	model = models.GNNStack(dataset.num_node_features, args.hidden_dim, dataset.num_classes,
							args, task=task)
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
	save_to = 'val_%s%s.pickle' % (task, args.model_type)
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

    datasets = utils.get_datasets()
    train(datasets, task, args)
