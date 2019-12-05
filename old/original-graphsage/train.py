import torch.nn as nn
from torch.autograd import Variable
from sklearn.metrics import roc_curve, confusion_matrix, recall_score, f1_score, auc, accuracy_score
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict
import argparse
import pandas as pd
import numpy as np
import random
import torch

import utils
import models

NUM_NODES = 100386


def load_dataset(args, features, edges, num_features):
    """
    Input:
        args - args from command line
        features - a filepath with each line being a space-delimited string of [node ID, [features], label name]
        edges - a filepath of the (directed) edges file (each line being "n1 n2" representing n1 -> n2)
    Returns:
        array of length len(nodes) as input
        array of length len(nodes) as binary 0/1 labels for each node, 1 indicating hateful/suspended
        NUM_NODES x num_features matrix of features
        NUM_NODES x 1 matrix of labels (mapping of which is printed out)
        adj list, dictionary of nodes to their set of neighbors
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

    adj_list = defaultdict(set)
    with open(edges) as fp:
        for i, line in enumerate(fp):
            info = line.strip().split()
            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            # TODO why add undirected edges?
            adj_list[paper1].add(paper2)
            adj_list[paper2].add(paper1)

    if args.dataset == "hate":
        df = pd.read_csv("hate/users_anon.csv")
        df = df[df.hate != "other"]     # Filter out 'other' labels.
        y = np.array([1 if v == "hateful" else 0 for v in df["hate"].values]) # Label hateful as 1.
        x = np.array(df["user_id"].values)
        del df
    elif args.dataset == 'suspended':
        df = pd.read_csv("suspended/users_anon.csv")
        np.random.seed(321)
        df2 = df[df["is_63_2"] == True].sample(668, axis=0) # Sample active as well as suspended accounts.
        df3 = df[df["is_63_2"] == False].sample(5405, axis=0)
        df = pd.concat([df2, df3])
        y = np.array([1 if v else 0 for v in df["is_63_2"].values])
        x = np.array(df["user_id"].values)
        del df, df2, df3

    print('Label meanings: ', label_map)
    return x, y, feat_data, labels, adj_list


def train(args, features, weights, edges, num_features):
    """
    args - args from command line
    features - a filepath with each line being a space-delimited string of [node ID, [features], label name]
    weights - array of len(classes) indicating weight of each class when computing loss.
        Higher weight should be assigned to less common classes.
        0 means to ignore a class.
    edges - a filepath of the (directed) edges file (each line being "n1 n2" representing n1 -> n2)
    num_features - number of computed features.
    """
    # For reproducibility
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)

    # Load the data
    x, y, feat_data, labels, adj_list = load_dataset(args, features, edges, num_features)
    print("Loaded dataset")

    # Define embeddings for each node to be used in aggregation (FEATURES DON'T CHANGE)
    features = nn.Embedding(NUM_NODES, num_features)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)

    # build model
    model = models.createGNN(args, features, adj_list, num_features, weights)

    # Train loop
    print("Starting training")
    f1_test = []
    accuracy_test = []
    auc_test = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
    for train_index, test_index in skf.split(x, y):
        train, test = x[train_index], x[test_index]
        total_loss = 0

        # TODO should this be outside the loop?
        _ , opt = utils.build_optimizer(args, model.parameters())

        for batch in range(1000):
            model.train()
            batch_nodes = train[:args.batch_size]
            train = np.roll(train, args.batch_size)      # Prepare train set for next batch.

            opt.zero_grad()
            loss = model.loss(batch_nodes, Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
            loss.backward()
            opt.step()
            total_loss += loss.data.item()

            if batch % 50 == 0:
                model.eval()
                val_output = model(test)
                labels_pred_validation = val_output.data.numpy().argmax(axis=1)
                labels_true_validation = labels[test].flatten()
                if args.dataset == "hate":
                    y_true = [1 if v == 2 else 0 for v in labels_true_validation]       # label 2 is hate
                    y_pred = [1 if v == 2 else 0 for v in labels_pred_validation]
                else:
                    y_true = [1 if v == 1 else 0 for v in labels_true_validation]       # label 1 is suspended
                    y_pred = [1 if v == 1 else 0 for v in labels_pred_validation]

                fscore = f1_score(y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None)
                recall = recall_score(y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None)
                print(confusion_matrix(y_true, y_pred))
                print('F1: {}, Recall: {}'.format(fscore, recall))

                # If we ever reach really good scores...
                if fscore > 0.70 and args.dataset == "hate":
                    break
                if fscore > 0.60 and recall > 0.8 and args.dataset != "hate":
                    break

        # TODO decompose test code?
        # For each split, evaluate AUC, accuracy, and F1 on test split.
        model.eval()
        val_output = model(test)
        if args.dataset == "hate":
            # Prediction score is the difference between hateful and non-hateful scores.
            labels_pred_score = val_output.data.numpy()[:, 2].flatten() - val_output.data.numpy()[:, 0].flatten()
        else:
            # Prediction score is the difference between suspended and active scores.
            labels_pred_score = val_output.data.numpy()[:, 1].flatten() - val_output.data.numpy()[:, 0].flatten()

        labels_true_test = labels[test].flatten()
        if args.dataset == "hate":
            y_true = [1 if v == 2 else 0 for v in labels_true_test]
        else:
            y_true = [1 if v == 1 else 0 for v in labels_true_test]

        fpr, tpr, _ = roc_curve(y_true, labels_pred_score)

        # TODO why is it different inside the training loop?
        # Prediction is the larger of the two hateful/non-hateful or suspended/active scores.
        labels_pred_test = labels_pred_score > 0
        y_pred = [1 if v else 0 for v in labels_pred_test]

        auc_test.append(auc(fpr, tpr))
        accuracy_test.append(accuracy_score(y_true, y_pred))
        f1_test.append(f1_score(y_true, y_pred))

    # Print out final accuracy, F1, AUC results.
    accuracy_test = np.array(accuracy_test)
    f1_test = np.array(f1_test)
    auc_test = np.array(auc_test)

    print("Accuracy   %0.4f +-  %0.4f" % (accuracy_test.mean(), accuracy_test.std()))
    print("F1    %0.4f +-  %0.4f" % (f1_test.mean(), f1_test.std()))
    print("AUC    %0.4f +-  %0.4f" % (auc_test.mean(), auc_test.std()))


def arg_parse():
    parser = argparse.ArgumentParser(description='GNN arguments.')
    utils.parse_optimizer(parser)

    parser.add_argument('--model_type', type=str,
            help='Type of GNN model.')
    parser.add_argument('--batch_size', type=int,
            help='Training batch size')
    parser.add_argument('--num_layers', type=int,
            help='Number of graph conv layers')
    parser.add_argument('--hidden_dim', type=int,
            help='Training hidden size')
    parser.add_argument('--dropout', type=float,
            help='Dropout rate')
    parser.add_argument('--dataset', type=str,
            help='Dataset, either hate or suspended')
    parser.add_argument('--glove_only', action='store_true',
            help='Whether to use only glove features')

    parser.set_defaults(model_type='GraphSage',
            dataset='hate',
            num_layers=1,
            batch_size=128,
            hidden_dim=256,
            dropout=0.0,
            opt='adam',   # opt_parser
            opt_scheduler='none',
            weight_decay=0.0,
            lr=0.01)

    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parse()

    # Determine dataset edges, features, and weights.
    if args.dataset == 'hate':
        edges = "hate/users.edges"
        if args.glove_only:
            features = "hate/users_hate_glove.content"
            num_features = 300
        else:
            features = "hate/users_hate_all.content"
            num_features = 320

        weights = [1, 0, 10] # Weights for non-hate, other, and hate classes.
    elif args.dataset == 'suspended':
        edges = "suspended/users.edges"
        if args.glove_only:
            features = "suspended/users_suspended_glove.content"
            num_features = 300
        else:
            features = "hate/users_hate_all.content"
            num_features = 320

        weights = [1, 15] # Weights for non-suspended, and suspended classes.

    else:
        raise ValueError("dataset provided was {}".format(args.dataset))

    train(args, edges=edges, features=features, num_features=num_features, weights=weights)
