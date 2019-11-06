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
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd
from pprint import pprint
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

    return interactions_graph


def generate_snap_dataset(
    dataset_grouping='basic',
    graph_out_path='network'
):
    """
    @params: [graph_out_path (str)]
    @returns: None

    Loads datasets specified in macros, and writes a user-connection graph to
    graph_out_path
    """
    dataset = load_datasets(dataset_grouping)
    graph= generate_network(dataset, graph_out_path)
    nx.write_gpickle(graph, graph_out_path+".gpickle")
    
    return graph


def get_RolX_feat(graph):
    """
    @params: [graph (networkx)]
    @returns: features vectors (matrix Node*nbr_feat)

    Create the node/features matrix for role assignements and for our GNN
    """
    
    feature_extractor = RecursiveFeatureExtractor(g)
    features = feature_extractor.extract_features()
    return features
    
def get_RolX_roles(features):
    """
    @params: [feat (pandas)]
    @returns: roles for each nodes

    Compute roles for every nodes
    """    
    role_extractor = RoleExtractor(n_roles=None)
    role_extractor.extract_role_factors(features)
    node_roles = role_extractor.roles
    
    #################### if you want to have access to what 
    #################### are the "probabilitis for each class 
    """
    print('\nNode role membership by percentage:')
    print(role_extractor.role_percentage.round(2))
    """
    
    return node_roles


def draw_Rolx(g,node_roles):
    """
    @params: [role (str), graph (networkx), node_roles (list)]
    @returns: 

    Draw the whole graph
    """  
    
    unique_roles = sorted(set(node_roles.values()))
    color_map = ["#9d6d00", "#903ee0", "#11dc79", "#f568ff", "#419500", "#013fb0", 
          "#f2b64c", "#007ae4", "#ff905a", "#33d3e3", "#9e003a", "#019085", 
          "#950065", "#afc98f", "#ff9bfa", "#83221d", "#01668a", "#ff7c7c", 
          "#643561", "#75608a"]
    
    # map roles to colors
    role_colors = {role: color_map[i] for i, role in enumerate(unique_roles)}
    # build list of colors for all nodes in G
    node_colors = [role_colors[node_roles[node]] for node in list_non_0]
    
    plt.figure()
    
    with warnings.catch_warnings():
        # catch matplotlib deprecation warning
        warnings.simplefilter('ignore')
        nx.draw(
            g,
            pos=nx.spring_layout(g, seed=42),
            with_labels=True,
            node_color=node_colors,
        )
    
    return node_roles

def draw_Rolx_per_role(role,g,node_roles):
    """
    @params: [role (str), graph (networkx), node_roles (list)]
    @returns: 

    Draw the egonet for nodes belonging to role
    """  
    
    unique_roles = sorted(set(node_roles.values()))
    color_map = ["#9d6d00", "#903ee0", "#11dc79", "#f568ff", "#419500", "#013fb0", 
          "#f2b64c", "#007ae4", "#ff905a", "#33d3e3", "#9e003a", "#019085", 
          "#950065", "#afc98f", "#ff9bfa", "#83221d", "#01668a", "#ff7c7c", 
          "#643561", "#75608a"]
    
    list_node_role=[]
    for i in node_roles.keys():
        if(node_roles[i]==role):
            list_node_role+=[i]
    
    #create subgraph with only non role 0
    sub_g=nx.DiGraph()
    sub_g.add_nodes_from(list_node_role)
    
    for edge in g.edges():
        if(sub_g.has_node(edge[0]) and sub_g.has_node(edge[1])):
            sub_g.add_edge(edge[0],edge[1])
    
    
    
    # map roles to colors
    role_colors = {role: color_map[i] for i, role in enumerate(unique_roles)}
    # build list of colors for all nodes in G
    node_colors = [role_colors[node_roles[node]] for node in list_node_role]
    
    plt.figure()
    
    with warnings.catch_warnings():
        # catch matplotlib deprecation warning
        warnings.simplefilter('ignore')
        nx.draw(
            sub_g,
            nodelist=list_node_role,
            pos=nx.spring_layout(sub_g, seed=42),
            with_labels=True,
            node_color=node_colors,
        )
    
    return node_roles


g=nx.read_gpickle("network.gpickle")
#g=generate_snap_dataset()
features=get_RolX_feat(g)
node_roles=get_RolX_roles(features)

draw_Rolx_per_role('role_2',g,node_roles)