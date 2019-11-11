#!/usr/bin/python3
#
# CS224W Fall 2019-2020
# @Jason Zheng, Guillaume Nervo, Jestin Ma
#
import argparse
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import os
import pandas as pd
import pathlib
import random
import snap
import sys
import dataset_utils

# Where graphs will be located.
PROCESSED_GRAPHS_DIR = './graphs'


# ==============================================================================
# Snap.py network generation code
# ==============================================================================
def generate_graphs(dataset_name):
    """
    @params: [dataset_path (csv path of a Pandas dataframe)]
    @returns: None

    Generates snap.py graphs between users with edges defined by retweets/mentions/replies
    using Pandas dataframe dataset. Saves graphs to
    the directory PROCESSED_GRAPHS_DIR.
    """

    dataset = dataset_utils.load_dataset(dataset_name)

    print("Generating graphs for %s, total rows %s" % (dataset_name, len(dataset.index)))

    mentions_graph = snap.TNGraph.New()
    reply_graph = snap.TNGraph.New()
    retweet_graph = snap.TNGraph.New()

    userid_to_node_map = dict()
    i = 0
    for row in dataset.iterrows():
        data = row[1]
        userid = data['userid']
        reply_id = data['in_reply_to_userid']
        retweet_id = data['retweet_of']

        if userid not in userid_to_node_map:
            user_node_id = i
            userid_to_node_map[userid] = i
            mentions_graph.AddNode(user_node_id)
            reply_graph.AddNode(user_node_id)
            retweet_graph.AddNode(user_node_id)
            i += 1
        else:
            user_node_id = userid_to_node_map[userid]

        # Equality checks whether it is NaN or not.
        if reply_id == reply_id:
            if reply_id not in userid_to_node_map:
                reply_node_id = i
                userid_to_node_map[reply_id] = i
                mentions_graph.AddNode(reply_node_id)
                reply_graph.AddNode(reply_node_id)
                retweet_graph.AddNode(reply_node_id)
                i += 1
            else:
                reply_node_id = userid_to_node_map[reply_id]
            reply_graph.AddEdge(user_node_id, reply_node_id)

        if retweet_id == retweet_id:
            if retweet_id not in userid_to_node_map:
                retweet_node_id = i
                userid_to_node_map[retweet_id] = i
                mentions_graph.AddNode(retweet_node_id)
                reply_graph.AddNode(retweet_node_id)
                retweet_graph.AddNode(retweet_node_id)
                i += 1
            else:
                retweet_node_id = userid_to_node_map[retweet_id]
            retweet_graph.AddEdge(user_node_id, retweet_node_id)

        if data['user_mentions'] == data['user_mentions']:
            for mention_id in data['user_mentions']:
                if mention_id not in userid_to_node_map:
                    mention_node_id = i
                    userid_to_node_map[mention_id] = i
                    mentions_graph.AddNode(mention_node_id)
                    reply_graph.AddNode(mention_node_id)
                    retweet_graph.AddNode(mention_node_id)
                    i += 1
                else:
                    mention_node_id = userid_to_node_map[mention_id]
                mentions_graph.AddEdge(user_node_id, mention_node_id)

        if row[0] % 10000 == 0:
            print('Processed %s rows so far' % (row[0] + 1))

    path = os.path.join(PROCESSED_GRAPHS_DIR, '%s-mentions.graph' % dataset_name)
    fout = snap.TFOut(path)
    mentions_graph.Save(fout)
    fout.Flush()

    path = os.path.join(PROCESSED_GRAPHS_DIR, '%s-reply.graph' % dataset_name)
    fout = snap.TFOut(path)
    reply_graph.Save(fout)
    fout.Flush()

    path = os.path.join(PROCESSED_GRAPHS_DIR, '%s-retweet.graph' % dataset_name)
    fout = snap.TFOut(path)
    retweet_graph.Save(fout)
    fout.Flush()


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


def analyze_network(
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
    graph = snap.TNGraph.Load(snap.TFIn(graph_in_path))
    snap.PrintInfo(graph, 'Basic Graph Information', '/dev/stdout', False)
    MxScc = snap.GetMxScc(graph)
    print('Nodes in largest strongly-connected subcomponent: %d' %
        MxScc.GetNodes()
    )
    visualize_k_random_users(k, fanout, fanout_samples, graph)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip_graph', action='store_true', help="whether to skip graph generation")
    parser.add_argument('--analyze', required=False)
    args = parser.parse_args()

    if not args.skip_graph:
        pathlib.Path(PROCESSED_GRAPHS_DIR).mkdir(parents=True, exist_ok=True)
        generate_graphs('benign')
        generate_graphs('bad')


    if args.analyze:
        print("Analyzing graph %s" % args.analyze)
        analyze_network(
            k=5,
            fanout=2,
            fanout_samples=3,
            graph_in_path=args.analyze
        )
