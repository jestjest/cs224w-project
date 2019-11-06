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
import random
import snap
import sys

DATASETS = {
    'bad actors': [
        'ira_tweets_csv_hashed.csv',
    ],
    'benign': [
        'json/democratic_party_timelines',
        'json/republican_party_timelines',
    ]
}

# ==============================================================================
# Dataset code
# ==============================================================================
def load_datasets():
    """
    @params: [dataset_grouping (str)]
    @returns: (Pandas Dataframe)

    Reads all csv's from dataset_grouping's input partition in DATASETS, and
    concatenates these to a single pandas dataframe. Returns the dataframe.
    """
    li = []
    for dataset in DATASETS['bad actors']:
        path = './datasets/%s' % dataset
        print('Reading data from %s' % path)
        df = pd.read_csv(path)
        df = format_csv_df(df)
        li.append(df)
    return pd.concat(li, axis=0, ignore_index=True)


def load_json():
    """
    @params: []
    @returns: (Pandas Dataframe)

    Reads all json's from dataset_grouping's input partition in DATASETS, and
    concatenates these to a single pandas dataframe. Returns the dataframe.
    """
    li = []
    for dataset in DATASETS['benign']:
        path = './datasets/%s' % dataset
        print('Reading data from %s' % path)
        df = pd.read_json(path, lines=True)
        df = convert_to_csv_df(df)
        li.append(df)
    return pd.concat(li, axis=0, ignore_index=True)


def format_csv_df(df):
    """
    @params: [df (Pandas Dataframe)]
    @returns: [df (Pandas Dataframe)]

    Selects the relevant fields from csv derived tweet dataframe
    """
    converted_struct = {
        'userid': df['userid'],
        'in_reply_to_userid': df['in_reply_to_userid'],
        'user_mentions': df['user_mentions'],
        'full_text': df['tweet_text'],
    }
    return pd.DataFrame(converted_struct)


def convert_to_csv_df(df):
    """
    @params: [df (Pandas Dataframe)]
    @returns: [df (Pandas Dataframe)]

    Converts the json structured tweet dataframe to match the CSV bad actors
    dataframe structure
    """
    user_ids = [user.get('id') for user in df['user']]
    user_mentions = [entity.get('user_mentions') for entity in df['entities']]

    converted_struct = {
        'userid': user_ids,
        'in_reply_to_userid': df['in_reply_to_user_id'],
        'user_mentions': user_mentions,
        'full_text': df['full_text'],
    }
    return pd.DataFrame(converted_struct)


# ==============================================================================
# Snap.py network generation code
# ==============================================================================
def generate_network(dataset, graph_out):
    """
    OUTDATED
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
        reply_id = data['in_reply_to_userid']
        mention_id = data['user_mentions']

        if userid not in userid_to_node_map:
            user_node_id = i
            userid_to_node_map[userid] = i
            interactions_graph.AddNode(user_node_id)
            i += 1
        else:
            user_node_id = userid_to_node_map[userid]

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


def get_k_graph_egonet(k, num_sampled, subgraph, graph):
    """
    OUTDATED
    @params: [
        k (int),
        num_sampled (int),
        subgraph (snap.TUNGraph),
        graph (snap.TUNGraph)
    ]
    @returns: k_neighborhood (snap.TUNGraph)

    Takes a subgraph from graph, and returns an egonet graph that contains the
    node as well as all of its neighbors of a distance k. Selects num_sampled
    nodes to fanout.
    """
    for i in range(k):
        nodes_arr = {node.GetId() for node in subgraph.Nodes()} #Prevent memory corruption
        nodes_arr = random.sample(nodes_arr, num_sampled)
        for graph_node in nodes_arr:
            graph_node_iter = graph.GetNI(graph_node)
            for neighbor in graph_node_iter.GetOutEdges():
                if not subgraph.IsNode(neighbor):
                    subgraph.AddNode(neighbor)
                if not subgraph.IsEdge(graph_node, neighbor):
                    subgraph.AddEdge(graph_node, neighbor)
    return subgraph

# ==============================================================================
# Networkx Visualization Code
# ==============================================================================
def visualize_k_random_users(k, fanout, fanout_samples, graph):
    """
    OUTDATED
    @params: [k (int), fanout_samples (int), graph (snap.TUNGraph)]
    @returns: None

    Loads the snap.py graph from graph, and samples k edges from the network to
    visualize using networkx. Samples fanout_samples nodes to fanout, to
    prevent intractibly large sample graphs.
    """
    sample_graph = snap.GetRndESubGraph(graph, k)
    sample_graph = get_k_graph_egonet(
        fanout,
        fanout_samples,
        sample_graph,
        graph
    )
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


def analyze_dataset_network(
    k=1000,
    fanout=1,
    fanout_samples=1,
    graph_in_path='bad_actors.graph'
):
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
    visualize_k_random_users(k, fanout, fanout_samples, graph)

# ==============================================================================
# Controller code
# ==============================================================================
def generate_snap_dataset(
    generate_network_flag=False
):
    """
    @params: [graph_out_path (str)]
    @returns: benign_dataset, bad_dataset

    Loads datasets specified in macros, and writes a user-connection graph to
    graph_out_path
    """
    for grouping in DATASETS:
        if grouping == 'benign':
            benign_dataset = load_json()
        else:
            bad_dataset = load_datasets()
    if generate_network_flag:
        print('Generating graph networks')
        bad_actor_graph = generate_network(bad_dataset, 'bad_actors.graph')
        benign_graph = generate_network(benign_dataset, 'benign.graph')
    return benign_dataset, bad_dataset


if __name__ == '__main__':
    """
    NOTE: This graph implementation assumes undirected edges.

    @flags: [
        '--gen':
            generate a new graph in addition to creating the datasets
        '--analyze'
            analyze a graph
    ]
    """
    if len(sys.argv) > 1 and '--gen' in sys.argv:
        print("Generating new graphs")
        benign, bad = generate_snap_dataset(generate_network_flag=True)
    else:
        print('Creating datasets without graphs')
        benign, bad = generate_snap_dataset(generate_network_flag=False)
    benign.to_csv('./datasets/compiled/benign_actors.csv', encoding='utf-8', index=False)
    bad.to_csv('./datasets/compiled/bad_actors.csv', encoding='utf-8', index=False)

    if len(sys.argv) > 1 and '--analyze' in sys.argv:
        print("Analyzing existing graph")
        analyze_dataset_network(
            k=100,
            fanout=1,
            fanout_samples=0,
        )
