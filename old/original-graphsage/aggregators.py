import torch
import torch.nn as nn
from torch.autograd import Variable

import random

"""
Set of modules for aggregating embeddings of neighbors.
"""

class MeanAggregator(nn.Module):
    """
    Aggregates a node's embeddings using mean of neighbors' embeddings
    """
    def __init__(self, features, cuda=False, gcn=False):
        """
        Initializes the aggregator for a specific graph.

        features -- function mapping LongTensor of node ids to FloatTensor of feature values.
        cuda -- whether to use GPU
        gcn --- whether to perform concatenation GraphSAGE-style, or add self-loops GCN-style
        """

        super(MeanAggregator, self).__init__()

        self.features = features
        self.cuda = cuda
        self.gcn = gcn

    def forward(self, nodes, to_neighs, num_sample=10):
        """
        nodes --- list of nodes in a batch
        to_neighs --- list of sets, each set is the set of neighbors for node in batch
        num_sample --- number of neighbors to sample. No sampling if None.
        """
        # Local pointers to functions (speed hack)
        # Sample neighbors if necessary.
        _set = set
        if not num_sample is None:
            _sample = random.sample
            samp_neighs = [_set(_sample(to_neigh,
                            num_sample,
                            )) if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs]
        else:
            samp_neighs = to_neighs

        # GCN - add self loops
        if self.gcn:
            samp_neighs = [samp_neigh + set([nodes[i]]) for i, samp_neigh in enumerate(samp_neighs)]

        # Map neighbors to index
        unique_nodes_list = list(set.union(*samp_neighs))
        unique_nodes = {n:i for i,n in enumerate(unique_nodes_list)}
        # Mask, (i, j) is a value for node i with neighbor j. [num_nodes, num_unique_nodes]
        # Determine j as column_indices, i as row_indices
        mask = Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
        # Looks like [v1.n1, v1.n2, ..., v100.n1, v100.n2, ...]
        column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
        # Looks like [v1 v1 v1 ... v100 v100 v100 ...]
        row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        mask[row_indices, column_indices] = 1
        if self.cuda:
            mask = mask.cuda()

        # Get rows of 1/num_neighbors for each node.
        num_neigh = mask.sum(1, keepdim=True)
        mask = mask.div(num_neigh)

        # embed_matrix [num_unique_nodes, num_features] of all neighbors
        if self.cuda:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list).cuda())
        else:
            embed_matrix = self.features(torch.LongTensor(unique_nodes_list))

        # Perform averaging
        # to_feats [num_nodes, num_features]
        to_feats = mask.mm(embed_matrix)
        return to_feats
