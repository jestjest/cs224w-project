# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 09:48:14 2019

@author: guill
"""

#
# CS224W Fall 2019-2020
# @Jason Zheng, Guillaume Nervo, Jestin Ma
#
import numpy as np
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import networkx as nx
from networkx.readwrite import json_graph
import pandas as pd
from pprint import pprint
import random
#import seaborn as sns
import snap
import sys
import warnings

from graphrole import RecursiveFeatureExtractor, RoleExtractor


DATASETS = {
    'basic': [
        'iran_201901_1_tweets_csv_hashed_1.csv',
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
        path = './Iran/%s' % dataset
        #cast column type because some id are of str and others 
        #float
        type_col={"userid":str,'retweet_userid':str,
                  "in_reply_to_userid":str}
        df = pd.read_csv(path,dtype=type_col)
        li.append(df)
    return pd.concat(li, axis=0, ignore_index=True)


def generate_network(dataset, graph_out):
    """
    @params: [dataset (Pandas Dataframe), graph_out (str)]
    @returns: None

    Generates a directed graph networkx between users with edges defined by interaction
    types from INTERACTIONS using Pandas dataframe dataset. Saves the graph to
    filepath graph_out. Currently not decomposed, to reduce memory payload.
    """
    interactions_graph = nx.DiGraph()
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
            interactions_graph.add_node(user_node_id)
            i += 1
        else:
            user_node_id = userid_to_node_map[userid]

        if not pd.isna(retweet_id):
            if retweet_id not in userid_to_node_map:
                retweet_node_id = i
                userid_to_node_map[retweet_id] = i
                interactions_graph.add_node(retweet_node_id)
                i += 1
            else:
                retweet_node_id = userid_to_node_map[retweet_id]
                
            interactions_graph.add_edge(user_node_id, retweet_node_id)

        if not pd.isna(reply_id):
            if reply_id not in userid_to_node_map:
                reply_node_id = i
                userid_to_node_map[reply_id] = i
                interactions_graph.add_node(reply_node_id)
                i += 1
            else:
                reply_node_id = userid_to_node_map[reply_id]
            interactions_graph.add_edge(user_node_id, reply_node_id)

        #user mention is a string of the form
        #"[user1,user2]" (it coulb be empty or 'nan')
        if (isinstance(mention_id,str) and mention_id!='[]'):
            l_mention=mention_id[1:-1].split(', ')
            for ind_user in range(len(l_mention)):
                if l_mention[ind_user] not in userid_to_node_map:
                    mention_node_id = i
                    userid_to_node_map[l_mention[ind_user]] = i
                    interactions_graph.add_node(mention_node_id)
                    i += 1
                else:
                    mention_node_id = userid_to_node_map[l_mention[ind_user]]
                interactions_graph.add_edge(user_node_id, mention_node_id)

    #nx.write_graphml_lxml(interactions_graph, "networkx_grapgh_iran") 
    
    return interactions_graph


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
    graph,user_dict = generate_network(dataset, graph_out_path)
    return graph

g=generate_snap_dataset()

############ Extract features from the graph #############
feature_extractor = RecursiveFeatureExtractor(g)
features = feature_extractor.extract_features()

print(f'\nFeatures extracted from {feature_extractor.generation_count} recursive generations:')


##################### Role extractions #################
role_extractor = RoleExtractor(n_roles=None)
role_extractor.extract_role_factors(features)
node_roles = role_extractor.roles


#################### if you want to have access to what 
#################### are the "probabilitis for each class 
"""
print('\nNode role membership by percentage:')
print(role_extractor.role_percentage.round(2))
"""