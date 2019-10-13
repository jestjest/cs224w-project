#
# CS224W Fall 2019-2020
# @Jason Zheng, Guillaume Nervo, Jestin Ma
#
import numpy as np
import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import snap
import sys

DATASETS = {
    'basic': [
        'ira_tweets_csv_hashed.csv',
    ],
}
INTERACTIONS = {
    'retweet': 'retweet_userid',
    'reply': 'in_reply_to_userid',
    'mention': 'user_mentions',
}


def load_datasets(dataset_grouping):
    """
    @params: [dataset_grouping (str)]
    @returns: (Pandas Dataframe)

    Reads all csv's from dataset_grouping's input partition in DATASETS, and
    concatenates these to a single pandas dataframe. Returns the dataframe.
    """
    li = []
    for dataset in DATASETS[dataset_grouping]:
        path = './datasets/%s' % dataset
        df = pd.read_csv(path)
        li.append(df)
    return pd.concat(li, axis=0, ignore_index=True)


def generate_network(dataset, graph_out):
    """
    @params: [dataset (Pandas Dataframe), graph_out (str)]
    @returns: None

    Generates a snap.py graph between users with edges defined by interaction
    types from INTERACTIONS using Pandas dataframe dataset. Saves the graph to
    filepath graph_out. Currently not decomposed, to reduce memory payload.
    """
    interactions_graph = snap.TUNGraph.New()
    userid_to_node_map = dict()
    i = 0
    for row in dataset.iterrows():
        data = row[1]
        userid = data['userid']
        retweet_id = data[INTERACTIONS['retweet']]
        reply_id = data[INTERACTIONS['reply']]
        mention_id = data[INTERACTIONS['mention']]

        if userid not in userid_to_node_map:
            user_node_id = i
            userid_to_node_map[userid] = i
            interactions_graph.AddNode(user_node_id)
            i += 1
        else:
            user_node_id = userid_to_node_map[userid]

        if retweet_id == retweet_id:
            if retweet_id not in userid_to_node_map:
                retweet_node_id = i
                userid_to_node_map[retweet_id] = i
                interactions_graph.AddNode(retweet_node_id)
                i += 1
            else:
                retweet_node_id = userid_to_node_map[retweet_id]
            interactions_graph.AddEdge(user_node_id, retweet_node_id)

        if reply_id == 'nan':
            if reply_id not in userid_to_node_map:
                reply_node_id = i
                userid_to_node_map[reply_id] = i
                interactions_graph.AddNode(reply_node_id)
                i += 1
            else:
                reply_node_id = userid_to_node_map[reply_id]
            interactions_graph.AddEdge(user_node_id, reply_node_id)

        if mention_id == 'nan':
            if mention_id not in userid_to_node_map:
                mention_node_id = i
                userid_to_node_map[mention_id] = i
                interactions_graph.AddNode(mention_node_id)
                i += 1
            else:
                mention_node_id = userid_to_node_map[mention_id]
            interactions_graph.AddEdge(user_node_id, mention_node_id)

    FOut = snap.TFOut(graph_out)
    interactions_graph.Save(FOut)
    FOut.Flush()


def visualize_k_random_users(k, graph):
    """
    @params: [k (int), graph (snap.TUNGraph)]
    @returns: None

    Loads the snap.py graph from graph, and samples k edges from the network to
    visualize using networkx.
    """
    sample_graph = snap.GetRndESubGraph(graph, k)
    snap.PrintInfo(
        sample_graph,
        'Sampled Graph Information',
        '/dev/stdout',
        False
    )
    nx_graph = nx.Graph()
    for node in sample_graph.Nodes():
        nx_graph.add_node(node.GetId())
    for edge in sample_graph.Edges():
        n1 = edge.GetSrcNId()
        n2 = edge.GetDstNId()
        nx_graph.add_edge(n1, n2)
    edges_list = [edge for edge in nx_graph.edges()]

    pos = nx.spring_layout(nx_graph)
    nx.draw_networkx_nodes(
        nx_graph,
        pos,
        node_color='b',
        node_size=10,
        alpha=0.6
    )
    nx.draw_networkx_edges(nx_graph, pos, edgelist=edges_list, arrows=False)
    plt.show()


def generate_snap_dataset(
    dataset_grouping='basic',
    graph_out_path='network.graph'
):
    """
    @params: [graph_out_path (str)]
    @returns: None

    Loads datasets specified in macros, and writes a user-connection graph to
    graph_out_path
    """
    dataset = load_datasets(dataset_grouping)
    graph = generate_network(dataset, graph_out_path)


def analyze_dataset_network(k=1000, graph_in_path='network.graph'):
    """
    @params: [k (int), graph_in_path (str)]
    @returns: None

    Loads a network from 'graph_in_path' and prints basic information about the
    network. Samples k edges from the network to visualize using networkx.
    """
    graph = snap.TUNGraph.Load(snap.TFIn(graph_in_path))
    snap.PrintInfo(graph, 'Basic Graph Information', '/dev/stdout', False)
    MxScc = snap.GetMxScc(graph)
    print('Nodes in largest strongly-connected subcomponent: %d' %
        MxScc.GetNodes()
    )
    visualize_k_random_users(k, graph)


if __name__ == '__main__':
    """
    @flags: [
        '--gen [dataset-grouping] [graph_out_path]':
            generate a new graph using the dataset-grouping, and graph_out_path
    ]
    """
    if len(sys.argv) > 1 and '--gen' in sys.argv:
        print("Generating a new graph")
        flag_idx = sys.argv.index('--gen')
        if len(sys.argv) > 2:
            dataset_grouping = sys.argv[flag_idx + 1]
            graph_out_path = sys.argv[flag_idx + 2]
            generate_snap_dataset(dataset_grouping, graph_out_path)
        else:
            generate_snap_dataset()
    else:
        print("Analyzing existing graph")
        analyze_dataset_network()
