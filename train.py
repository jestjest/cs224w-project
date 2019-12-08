#!/usr/bin/env python
#
# CS224W Fall 2019-2020
# @Jason Zheng
#

import argparse
from logger_utils import Logger
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
NUM_FEATURES = 320
NUM_ROLX_FEATURES = 44
dev = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

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

    opt_parser.add_argument('--use_refex', action='store_true')
    args = opt_parser.parse_args()
    return args


def load_hate(
        args,
        datadir='hate',
        features='users_hate_all.content',
        edges='users.edges',
        features_rolx='../data/features_refex_normalized.csv'):
    """
    Returns:
            NUM_NODES x num_feates matrix of features
            NUM_NODES x 1 matrix of labels (which are printed out)
            adjacency list of the (directed) edges file (each line being n1 n2 representing n1 -> n2)
                    as a dictionary of n1 to n2.
    """
    num_feats = NUM_FEATURES if not args.use_refex else NUM_FEATURES + NUM_ROLX_FEATURES
    feat_data = torch.zeros((NUM_NODES, num_feats), device=dev)
    labels = torch.empty((NUM_NODES, 1), dtype=torch.long, device=dev)
    node_map = {}
    label_map = {}

    features_tensor_path = os.path.join(datadir, features + '.tensor')
    labels_tensor_path = os.path.join(datadir, features + '.labels.tensor')
    edges_tensor_path = os.path.join(datadir, edges + '.tensor')

    if os.path.exists(features_tensor_path):
        feat_data = torch.load(features_tensor_path, map_location=dev)
        labels = torch.load(labels_tensor_path, map_location=dev)
        edge_tensor = torch.load(edges_tensor_path, map_location=dev)
    else:
        with open(features) as fp:
            for i, line in enumerate(fp):
                info = line.strip().split()
                feat_data[i, :] = torch.tensor(list(map(float, info[1:-1])))
                node_map[info[0]] = i
                if not info[-1] in label_map:
                    label_map[info[-1]] = len(label_map)
                labels[i] = label_map[info[-1]]

        if args.use_refex:
            feat_refex = pd.read_csv(features_rolx)
            for i in range(1, len(feat_refex)):
                feat_data[i, num_features:] = torch.tensor(feat_refex.iloc[i].values[1:])

        edge_src = []
        edge_dst = []
        with open(edges) as fp:
            for i, line in enumerate(fp):
                info = line.strip().split()
                paper1 = node_map[info[0]]
                paper2 = node_map[info[1]]
                edge_src.append(paper1)
                edge_dst.append(paper2)

        edge_tensor = torch.tensor([edge_src, edge_dst], dtype=torch.long, device=dev)
        labels = labels.squeeze()
        print('Label meanings: ', label_map)
        torch.save(feat_data, features_tensor_path)
        torch.save(labels, labels_tensor_path)
        torch.save(edge_tensor, edges_tensor_path)
        print('Saved parsed tensors for next run.')

    return feat_data, labels, edge_tensor


def get_stratified_batches():
    df = pd.read_csv("hate/users_anon.csv")
    df = df[df.hate != "other"]
    y = torch.tensor(
        [1 if v == "hateful" else 0 for v in df["hate"].values])
    x = torch.tensor(df["user_id"].values)
    del df

    # Assuming train-test ratio of 0.8
    #skf = StratifiedShuffleSplit(
    #        n_splits=10, test_size=0.2, train_size=0.8, random_state=123)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
    return skf, x, y


def get_datasets(args):
    """
    Gets the dataset for hateful Twitter users
    """
    # mask not multiple datasets
    datadir = 'hate_with_refex' if args.use_refex else 'hate'
    feat_data, labels, edges = load_hate(args, datadir)

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

    metrics = [
        'total_loss',
        'acc',
        'f1',
        'auc',
        'recall'
    ]
    logger = Logger(model=args.model_type)

    # build model
    model = models.GNNStack(
        320,  # dataset.num_node_features
        args.hidden_dim,  # args.hidden_dim
        3,  # dataset.num_classes
        args,
        torch.tensor([1, 0, 15], device=dev).float()    # weights for each class
    )
    if torch.cuda.is_available():
        model = model.cuda(dev)

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
        log_metrics = {
            'total_loss': total_loss,
            'acc': accs,
            'f1': f1s,
            'auc': aucs,
            'recall': recalls
        }

        logger.log(log_metrics, epoch)
        if epoch % 5 == 0:
            logger.display_status(
                epoch,
                args.epochs,
                total_loss,
                accs,
                f1s,
                aucs,
                recalls
            )
    logger.close()


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

    probs_score = probs.data.cpu().numpy()[:, 2].flatten() - probs.data.cpu().numpy()[:, 0].flatten()
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

    dataset = get_datasets(args)
    train(dataset, args)
