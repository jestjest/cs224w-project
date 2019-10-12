#
# CS224W Fall 2019-2020
# @jzzheng
#

import numpy as np
import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
import snap

DATASETS = [
    'ira_tweets_csv_hashed.csv'
]
GRAPH_OUT = 'network.graph'

INTERACTIONS = {
    'retweet': 'retweet_userid',
    'reply': 'in_reply_to_userid',
    'mention': 'user_mentions',
}

def load_datasets():
    path = './datasets/%s' % DATASETS[0]
    return pd.read_csv(path)

def generate_network(dataset, graph_out):
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

    print(interactions_graph.GetNodes())
    print(interactions_graph.GetEdges())
    MxScc = snap.GetMxScc(interactions_graph)
    print(MxScc.GetNodes())

    return interactions_graph

def visualize_k_random_users(k, graph):
    for i in range(k):
        random_user = GetRndNI()



def main():
    # dataset = load_datasets()
    # graph = generate_network(dataset, GRAPH_OUT)
    # visualize_k_random_users(k, graph)
    # get graph

    # Graph = snap.TUNGraph.New()
    # snap.PrintInfo(Graph, '')

    Graph = snap.TUNGraph.Load(snap.TFIn("network.graph"))
    snap.PrintInfo(Graph, '')

    # snap.PlotOutDegDistr(Graph, "deg", "Degree Distribution")
    # snap.PlotSccDistr(Graph, "scc", "Scc distribution")

    # Sampled_graph = snap.GetRndSubGraph(Graph,50)
    # snap.DrawGViz(Sampled_graph, snap.gvlDot, "sampled_graph.png", "Sample Graph Random Nodes")



if __name__ == '__main__':
    main()
