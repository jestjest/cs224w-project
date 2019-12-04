import torch
import torch.nn as nn
from torch.nn import init

from encoders import *
from aggregators import *

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


def createGNN(args, features, adj_list, num_features, class_weights):
    if args.model_type == 'GraphSage':
        agg1 = MeanAggregator(features, cuda=False)
        enc1 = Encoder(features, num_features, args.hidden_dim, adj_list, agg1, gcn=False, cuda=False)
        enc1.num_samples = 25   # Sample 25 neighbors when aggregating.
        return SupervisedGraphSage(len(class_weights), enc1, torch.FloatTensor(class_weights))
