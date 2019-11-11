#!/usr/bin/python3
#
# CS224W Fall 2019-2020
# @Jason Zheng, Guillaume Nervo, Jestin Ma
#
import datetime as datetime
import numpy as np
import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import networkx as nx
import os
import pandas as pd
import pathlib
import random
import snap
import sys


# Where files listed in DATASETS are located.
DATASET_DIR = '/shared/data'

DATASETS = {
    'bad actors': [
        'iran_201906_1_tweets_csv_hashed.csv',
    ],
    'benign': [
        'json/democratic_party_timelines',
        'json/republican_party_timelines',
    ]
}

# Where processed datasets will be located.
PROCESSED_DATA_DIR = './datasets/compiled'

# ==============================================================================
# Dataset code
# ==============================================================================

def reformat_datetime(datetime_str, out_format):
    """
    @params [a UTC datetime string with a specific format (see below)]
    @returns: [a date string in the format of out_format]

    Reformats a UTC datetime string returned by the Twitter API for compatibility.
    """

    in_format = "%a %b %d %H:%M:%S %z %Y"
    parsed_datetime = datetime.datetime.strptime(datetime_str, in_format)
    return datetime.datetime.strftime(parsed_datetime, out_format)


def load_datasets():
    """
    @params: [dataset_grouping (str)]
    @returns: (Pandas Dataframe)

    Reads all csv's from dataset_grouping's input partition in DATASETS, and
    concatenates these to a single pandas dataframe. Returns the dataframe.
    """
    li = []
    for dataset in DATASETS['bad actors']:
        path = os.path.join(DATASET_DIR, dataset)
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
        path = os.path.join(DATASET_DIR, dataset)
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
        'followers_count': df['follower_count'],
        'following_count': df['following_count'],
        'account_creation_date': df['account_creation_date'],    # Format: YYYY-MM-DD

        'tweet_time': df['tweet_time'],
        'full_text': df['tweet_text'],
        'like_count': df['like_count'],
        'user_mentions': df['user_mentions'],
        'in_reply_to_userid': df['in_reply_to_userid'],     # NaN if not a reply.

        'is_retweeted': df['is_retweet'],
        'retweet_count': df['retweet_count'],
        'retweet_of': df['retweet_userid']
    }
    return pd.DataFrame(converted_struct)


def convert_to_csv_df(df):
    """
    @params: [df (Pandas Dataframe)]
    @returns: [df (Pandas Dataframe)]

    Converts the json structured tweet dataframe to match the CSV bad actors
    dataframe structure
    """
    out_format = "%Y-%m-%d"
    # print(df['retweeted_status'])
    # print(df['retweeted_status']['user'].get('id'))

    user_metadata = [(
        user.get('id'),
        user.get('followers_count'),
        user.get('friends_count'),
        reformat_datetime(user.get('created_at'), "%Y-%m-%d"))
        for user in df['user']]

    unzipped_user_metadata = list(zip(*user_metadata))
    # A list where the Nth item is a list of user mention in the Nth tweet.
    user_mentions = list()
    for entity in df['entities']:
        tweet_mentions = [mention['id'] for mention in entity.get('user_mentions')]
        user_mentions.append(tweet_mentions)

    retweet_status = list()
    for status in df['retweeted_status']:
        if status == status:
            retweet_status.append(status['user'].get('id'))
        else:
            retweet_status.append(None)

    converted_struct = {
        'userid': unzipped_user_metadata[0],
        'followers_count': unzipped_user_metadata[1],
        'following_count': unzipped_user_metadata[2],
        'account_creation_date': unzipped_user_metadata[3],

        'tweet_time': df['created_at'].dt.strftime("%Y-%m-%d %H:%M"),
        'full_text': df['full_text'],
        'like_count': df['favorite_count'],
        'user_mentions': user_mentions,
        'in_reply_to_userid': df['in_reply_to_user_id'],        # NaN if not a reply.

        # Existence of this attribute determines whether a tweet is a retweet.
        'is_retweeted': pd.isnull(df['retweeted_status']),
        'retweet_count': df['retweet_count'],
        'retweet_of': retweet_status        # None if no retweet
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
        retweet_id = data['retweet_of']
        if graph_out == 'benign.graph' and reply_id == reply_id: #hack for detecting obfuscated id's
            reply_id = int(reply_id)
            retweet_id = int(retweet_id)

        if userid not in userid_to_node_map:
            user_node_id = i
            userid_to_node_map[userid] = i
            interactions_graph.AddNode(user_node_id)
            i += 1
        else:
            user_node_id = userid_to_node_map[userid]

        if reply_id == reply_id:
            if reply_id not in userid_to_node_map:
                reply_node_id = i
                userid_to_node_map[reply_id] = i
                interactions_graph.AddNode(reply_node_id)
                i += 1
            else:
                reply_node_id = userid_to_node_map[reply_id]
            interactions_graph.AddEdge(user_node_id, reply_node_id)

        if retweet_id and retweet_id == retweet_id:
            if retweet_id not in userid_to_node_map:
                retweet_node_id = i
                userid_to_node_map[retweet_id] = i
                interactions_graph.AddNode(retweet_node_id)
                i += 1
            else:
                retweet_node_id = userid_to_node_map[retweet_id]
            interactions_graph.AddEdge(user_node_id, retweet_node_id)

        if data['user_mentions'] == data['user_mentions']:
            for mention_id in data['user_mentions']:
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
        generate_network(bad_dataset, 'bad_actors.graph')
        generate_network(benign_dataset, 'benign.graph')
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
    elif len(sys.argv) > 1 and '--analyze' in sys.argv:
        print("Analyzing existing graph")
        analyze_dataset_network(
            k=5,
            fanout=2,
            fanout_samples=3,
        )
    else:
        print('Creating datasets without graphs')
        benign, bad = generate_snap_dataset(generate_network_flag=False)
    pathlib.Path(PROCESSED_DATA_DIR).mkdir(parents=True, exist_ok=True)
    benign.to_csv(
        os.path.join(PROCESSED_DATA_DIR, 'benign_actors.csv'), encoding='utf-8', index=False)
    bad.to_csv(
        os.path.join(PROCESSED_DATA_DIR, 'bad_actors.csv'), encoding='utf-8', index=False)
