import torch.nn as nn
from torch.nn import init
from torch.autograd import Variable
from sklearn.metrics import roc_curve, confusion_matrix, recall_score, f1_score, auc, accuracy_score
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict
import pandas as pd
import numpy as np
import random
import torch
import time

from graphsage.encoders import Encoder
from graphsage.aggregators import MeanAggregator

NUM_NODES = 100386


class SupervisedGraphSage(nn.Module):
    def __init__(self, num_classes, enc, w):
        """
        w - array of len(num_classes) indicating the weight of each class when computing
            loss.
        """
        super(SupervisedGraphSage, self).__init__()
        self.enc = enc
        self.w = w
        self.xent = nn.CrossEntropyLoss(weight=self.w)
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, enc.embed_dim))
        init.xavier_uniform_(self.weight)

    def forward(self, nodes):
        embeds = self.enc(nodes)
        scores = self.weight.mm(embeds)
        return scores.t()

    def loss(self, nodes, labels):
        scores = self.forward(nodes)
        return self.xent(scores, labels.squeeze())


def load_hate(features, edges, num_features):
    """
    Input:
        features - a filepath with each line being a space-delimited string of [node ID, [features], label name]
        edges - a filepath of the (directed) edges file (each line being "n1 n2" representing n1 -> n2)
    Returns:
        NUM_NODES x num_features matrix of features
        NUM_NODES x 1 matrix of labels (which are printed out)
        adjacency list as an dictionary of nodes to the set of neighbors (UNDIRECTED!).
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


def run_hate(gcn, features, weights, edges, flag_index="hate", num_features=320,
             lr=0.01, batch_size=128):
    """
    gcn - whether to use GCN for encoding
    features - a filepath with each line being a space-delimited string of [node ID, [features], label name]
    weights - array of len(classes) indicating weight of each class when computing loss.
        Higher weight should be assigned to less common classes.
        0 means to ignore a class.
    edges - a filepath of the (directed) edges file (each line being "n1 n2" representing n1 -> n2)
    flag_index - "hate" means to use the hate dataset instead of the suspended dataset.
    num_features - number of computed features.
    """
    # For reproducibility
    torch.manual_seed(1)
    np.random.seed(1)
    random.seed(1)

    # Load the data
    feat_data, labels, adj_lists = load_hate(features, edges, num_features)

    # Define features for each node to be used in aggregation (FEATURES DON'T CHANGE)
    features = nn.Embedding(NUM_NODES, num_features)
    features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)

    # Simple GraphSage Aggregate and encode the data
    agg1 = MeanAggregator(features, cuda=False)
    enc1 = Encoder(features, num_features, 256, adj_lists, agg1, gcn=gcn, cuda=False)
    enc1.num_samples = 25   # Sample 25 neighbors when aggregating.
    graphsage = SupervisedGraphSage(len(weights), enc1, torch.FloatTensor(weights))

    if flag_index == "hate":
        df = pd.read_csv("hate/users_anon.csv")
        df = df[df.hate != "other"]     # Filter out 'other' labels.
        y = np.array([1 if v == "hateful" else 0 for v in df["hate"].values]) # Label hateful as 1.
        x = np.array(df["user_id"].values)
        del df

    else:
        df = pd.read_csv("suspended/users_anon.csv")
        np.random.seed(321)
        df2 = df[df["is_63_2"] == True].sample(668, axis=0) # Sample active as well as suspended accounts.
        df3 = df[df["is_63_2"] == False].sample(5405, axis=0)
        df = pd.concat([df2, df3])
        y = np.array([1 if v else 0 for v in df["is_63_2"].values])
        x = np.array(df["user_id"].values)
        del df, df2, df3

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)

    f1_test = []
    accuracy_test = []
    auc_test = []
    for train_index, test_index in skf.split(x, y):
        train, test = x[train_index], x[test_index]

        # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, graphsage.parameters()))
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, graphsage.parameters()), lr=lr)
        times = []
        cum_loss = 0

        for batch in range(1000):
            batch_nodes = train[:batch_size]
            train = np.roll(train, batch_size)      # Prepare train set for next batch.

            start_time = time.time()
            optimizer.zero_grad()
            loss = graphsage.loss(batch_nodes, Variable(torch.LongTensor(labels[np.array(batch_nodes)])))
            loss.backward()
            optimizer.step()
            end_time = time.time()
            times.append(end_time - start_time)
            cum_loss += loss.data.item()

            if batch % 50 == 0:
                val_output = graphsage.forward(test)
                labels_pred_validation = val_output.data.numpy().argmax(axis=1)
                labels_true_validation = labels[test].flatten()
                if flag_index == "hate":
                    y_true = [1 if v == 2 else 0 for v in labels_true_validation]       # label 2 is hate
                    y_pred = [1 if v == 2 else 0 for v in labels_pred_validation]
                else:
                    y_true = [1 if v == 1 else 0 for v in labels_true_validation]       # label 1 is suspended
                    y_pred = [1 if v == 1 else 0 for v in labels_pred_validation]

                fscore = f1_score(y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None)
                recall = recall_score(y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None)
                print(confusion_matrix(y_true, y_pred))
                print('F1: {}, Recall: {}'.format(fscore, recall))

                # Stop on current split when satisfactory.
                if fscore > 0.65 and flag_index == "hate":
                    break
                if fscore >= 0.50 and recall > 0.8 and flag_index != "hate":
                    break

        # For each split, evaluate AUC, accuracy, and F1 on test split.
        val_output = graphsage.forward(test)
        if flag_index == "hate":
            labels_pred_score = val_output.data.numpy()[:, 2].flatten() - val_output.data.numpy()[:, 0].flatten()
        else:
            labels_pred_score = val_output.data.numpy()[:, 1].flatten() - val_output.data.numpy()[:, 0].flatten()

        labels_true_test = labels[test].flatten()
        if flag_index == "hate":
            y_true = [1 if v == 2 else 0 for v in labels_true_test]
        else:
            y_true = [1 if v else 0 for v in labels_true_test]

        fpr, tpr, _ = roc_curve(y_true, labels_pred_score)

        labels_pred_test = labels_pred_score > 0
        y_pred = [1 if v else 0 for v in labels_pred_test]

        auc_test.append(auc(fpr, tpr))
        accuracy_test.append(accuracy_score(y_true, y_pred))
        f1_test.append(f1_score(y_true, y_pred))
        print(confusion_matrix(y_true, y_pred))

    # Print out final accuracy, F1, AUC results.
    accuracy_test = np.array(accuracy_test)
    f1_test = np.array(f1_test)
    auc_test = np.array(auc_test)

    print("Accuracy   %0.4f +-  %0.4f" % (accuracy_test.mean(), accuracy_test.std()))
    print("F1    %0.4f +-  %0.4f" % (f1_test.mean(), f1_test.std()))
    print("AUC    %0.4f +-  %0.4f" % (auc_test.mean(), auc_test.std()))


if __name__ == "__main__":
    print("GraphSage all hate")
    run_hate(gcn=False, edges="hate/users.edges", features="hate/users_hate_all.content",
             num_features=320, weights=[1, 0, 10])

    print("GraphSage glove hate")
    run_hate(gcn=False,  edges="hate/users.edges", features="hate/users_hate_glove.content",
             num_features=300, weights=[1, 0, 10])

    print("GraphSage all suspended")
    run_hate(gcn=False, edges="suspended/users.edges", features="suspended/users_suspended_all.content",
             flag_index="suspended", num_features=320, weights=[1, 15], batch_size=128)

    print("GraphSage glove suspended")
    run_hate(gcn=False, edges="suspended/users.edges", features="suspended/users_suspended_glove.content",
             flag_index="suspended",  num_features=300, weights=[1, 15], batch_size=128)


