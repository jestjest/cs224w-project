#!/usr/bin/env python
#
# CS224W Fall 2019-2020
# @Jason Zheng
#

import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
from sklearn.metrics import roc_curve, confusion_matrix, recall_score, f1_score, auc, accuracy_score
import time
import torch
import torch_geometric.data as pyg_d
import torch_geometric.nn as pyg_nn
import torch.optim as optim
import os
import random

import models

NUM_NODES = 100386

# ==============================================================================
# Utils
# ==============================================================================


def local_arg_parse():
    opt_parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
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

    if os.path.exists(features + '.tensor'):
        feat_data = torch.load(features + '.tensor')
        labels = torch.load(features + '.labels.tensor')
        edge_tensor = torch.load(edges + '.tensor')
    else:
        with open(features) as fp:
            for i, line in enumerate(fp):
                info = line.strip().split()
                feat_data[i, :] = torch.tensor(list(map(float, info[1:-1])))
                node_map[info[0]] = i
                if not info[-1] in label_map:
                    label_map[info[-1]] = len(label_map)
                labels[i] = label_map[info[-1]]

        edge_list = list()
        with open(edges) as fp:
            for i, line in enumerate(fp):
                info = line.strip().split()
                paper1 = node_map[info[0]]
                paper2 = node_map[info[1]]
                edge_list.append([paper1, paper2])

        edge_tensor = torch.tensor(edge_list, dtype=torch.long)
        labels = labels.squeeze()
        print('Label meanings: ', label_map)
        torch.save(feat_data, features + '.tensor')
        torch.save(labels, features + '.labels.tensor')
        torch.save(edge_tensor, edges + '.tensor')
        print('Saved parsed tensors for next run.')

    return feat_data, labels, edge_tensor


def get_stratified_batches():
    if args.flag_index == "hate":
        df = pd.read_csv("hate/users_anon.csv")
        df = df[df.hate != "other"]
        y = torch.tensor(
            [1 if v == "hateful" else 0 for v in df["hate"].values])
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

    # Assuming train-test ratio of 0.8
    skf = StratifiedShuffleSplit(
        n_splits=10, test_size=0.2, train_size=0.8, random_state=123)
    #skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
    return skf, x, y


def get_datasets():
    """
    Gets the dataset for hateful Twitter users
    """
    # mask not multiple datasets
    feat_data, labels, edges = load_hate()

    dataset = pyg_d.Data(x=feat_data, edge_index=edges,
                         y=labels, batch=feat_data[:, 0])
    return dataset


def build_optimizer(args, params):
    weight_decay = args.weight_decay
    filter_fn = filter(lambda p: p.requires_grad, params)
    if args.opt == 'adam':
        optimizer = optim.Adam(filter_fn, lr=args.lr,
                               weight_decay=weight_decay)
    elif args.opt == 'sgd':
        optimizer = optim.SGD(filter_fn, lr=args.lr,
                              momentum=0.95, weight_decay=weight_decay)
    elif args.opt == 'rmsprop':
        optimizer = optim.RMSprop(
            filter_fn, lr=args.lr, weight_decay=weight_decay)
    elif args.opt == 'adagrad':
        optimizer = optim.Adagrad(
            filter_fn, lr=args.lr, weight_decay=weight_decay)
    if args.opt_scheduler == 'none':
        return None, optimizer
    elif args.opt_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=args.opt_decay_step, gamma=args.opt_decay_rate)
    elif args.opt_scheduler == 'cos':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.opt_restart)
    return scheduler, optimizer


# ==============================================================================
# Train
# ==============================================================================
def train(dataset, args):
    # For reproducibility
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)

    # build model
    model = models.GNNStack(
        320,  # dataset.num_node_features
        args.hidden_dim,  # args.hidden_dim
        3,  # dataset.num_classes
        args,
        torch.tensor([1, 0, 15]).float()    # weights for each class
    )
    scheduler, opt = build_optimizer(args, model.parameters())
    skf, x, y = get_stratified_batches()

    # train
    validation_accuracy = list()
    for epoch in range(args.epochs):
        total_loss = 0
        accs, f1s, aucs, recalls = [], [], [], []
        model.train()
        # No need to loop over batches since we only have one batch
        num_splits = 0
        for train_indices, test_indices in skf.split(x, y):
            train_indices, test_indices = x[train_indices], x[test_indices]
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

            acc_score, f1, auc_score, recall = test(dataset, model, test_indices)
            accs.append(acc_score)
            f1s.append(f1)
            aucs.append(auc_score)
            recalls.append(recall)

        total_loss /= num_splits
        accs = np.array(accs)
        f1s = np.array(f1s)
        aucs = np.array(aucs)
        recalls = np.array(recalls)
        print('Epoch ', epoch)
        print('\tLoss: ', total_loss)
        print('\tACC: %0.4f +- %0.4f' %
                (accs.mean(), accs.std()))
        print('\tF1: %0.4f +- %0.4f' %
                (f1s.mean(), f1s.std()))
        print('\tAUC: %0.4f +- %0.4f' %
                (aucs.mean(), aucs.std()))
        print('\tRecall: %0.4f +- %0.4f' %
                (recalls.mean(), recalls.std()))


def test(dataset, model, test_indices):
    model.eval()

    correct = 0
    with torch.no_grad():
        # max(dim=1) returns values, indices tuple; only need indices
        probs = model(dataset)
        pred = probs.max(dim=1)[1]
        label = dataset.y

    probs = probs[test_indices]
    pred = pred[test_indices]
    label = dataset.y[test_indices]

    probs_score = probs.data.numpy()[:, 2].flatten() - probs.data.numpy()[:, 0].flatten()
    labels_true_test = label.flatten()
    y_true = [1 if v == 2 else 0 for v in labels_true_test]
    fpr, tpr, _ = roc_curve(y_true, probs_score)
    y_pred = [1 if v else 0 for v in probs_score > 0]

    auc_score = auc(fpr, tpr)
    acc_score = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None)
    recall = recall_score(y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None)
    # correct = pred.eq(label).sum().item()

    return acc_score, f1, auc_score, recall


# ==============================================================================
# Controller
# ==============================================================================
if __name__ == "__main__":
    args = local_arg_parse()

    dataset = get_datasets()
    train(dataset, args)
