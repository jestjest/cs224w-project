import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils

class GNNStack(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, args, weights):
        super(GNNStack, self).__init__()

        # Special case for GatedGraphConv where hidden_dim > input_dim, just increase it a little.
        if args.model_type == 'Gate':
            hidden_dim = input_dim + 16

        self.convs = nn.ModuleList()
        self.convs.append(self.build_layer(args.model_type, input_dim, hidden_dim))
        assert (args.num_layers >= 1), 'Number of layers is not >=1'
        for l in range(args.num_layers-1):
            self.convs.append(self.build_layer(args.model_type, hidden_dim, hidden_dim))

        # post-message-passing
        self.post_mp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Dropout(args.dropout),
            nn.Linear(hidden_dim, output_dim))

        self.dropout = args.dropout
        self.num_layers = args.num_layers
        self.weights = weights

    def build_layer(self, model_type, input_dim, hidden_dim):
        if model_type == 'GCN':
            return pyg_nn.GCNConv(input_dim, hidden_dim)
        elif model_type == 'GraphSage':
            return GraphSage(input_dim, hidden_dim)
        elif model_type == 'GAT':
            return GAT(input_dim, hidden_dim)
        elif model_type == 'Gate':
            return pyg_nn.GatedGraphConv(hidden_dim, 2)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        for i in range(self.num_layers):
            conv_out = self.convs[i](x, edge_index)
            relu_out = F.relu(conv_out)
            x = F.dropout(relu_out, p=self.dropout, training=self.training)
        x = self.post_mp(x)

        return F.log_softmax(x, dim=1)

    def loss(self, pred, label):
        return F.nll_loss(pred, label, weight=self.weights)


class GraphSage(pyg_nn.MessagePassing):
    """Non-minibatch version of GraphSage."""
    def __init__(self, in_channels, out_channels, reducer='mean',
                 normalize_embedding=True):
        super(GraphSage, self).__init__(aggr='mean')

        self.lin = nn.Linear(in_channels, out_channels)
        self.agg_lin = nn.Linear(in_channels + out_channels, out_channels)
        if normalize_embedding:
            self.normalize_emb = True

    def forward(self, x, edge_index):
        num_nodes = x.size(0)
        return self.propagate(edge_index, size=(num_nodes, num_nodes), x=x)

    def message(self, x_j, edge_index, size):
        x_j = self.lin(x_j)
        x_j = F.relu(x_j)

        return x_j

    def update(self, aggr_out, x):
        concat_out = torch.cat([x, aggr_out], dim=-1)
        aggr_out = F.relu(self.agg_lin(concat_out))
        if self.normalize_emb:
            aggr_out = F.normalize(aggr_out, p=2)
        return aggr_out


class GAT(pyg_nn.MessagePassing):

    def __init__(self, in_channels, out_channels, num_heads=1, concat=True,
                 dropout=0, bias=True, **kwargs):
        super(GAT, self).__init__(aggr='add', **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = num_heads
        self.concat = concat
        self.dropout = dropout
        self.lin = nn.Linear(in_channels, num_heads * out_channels)
        self.att = nn.Parameter(torch.empty([1, num_heads, 2 * out_channels]))
        if bias and concat:
            self.bias = nn.Parameter(torch.empty([self.heads * out_channels]))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.empty([out_channels]))
        else:
            self.register_parameter('bias', None)
        nn.init.xavier_uniform_(self.att)
        nn.init.zeros_(self.bias)


    def forward(self, x, edge_index, size=None):
        x = self.lin(x)
        return self.propagate(edge_index, size=size, x=x)

    def message(self, edge_index_i, x_i, x_j, size_i):
        x_j = x_j.view(-1, self.heads, self.out_channels)
        x_i = x_i.view(-1, self.heads, self.out_channels)
        e_ij = (self.att * torch.cat([x_i, x_j], dim=-1)).sum(dim=-1)
        relu_out = F.leaky_relu(e_ij, 0.2)
        alpha = pyg_utils.softmax(relu_out, edge_index_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return x_j * alpha.view(-1, self.heads, 1)

    def update(self, aggr_out):
        if self.concat is True:
            aggr_out = aggr_out.view(-1, self.heads * self.out_channels)
        else:
            aggr_out = aggr_out.mean(dim=1)

        if self.bias is not None:
            aggr_out = aggr_out + self.bias
        return aggr_out
