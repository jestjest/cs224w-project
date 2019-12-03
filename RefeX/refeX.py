# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 19:48:37 2019

@author: guill
"""

import math
import random
import functools
import scipy.stats
import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm



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
    path = './Iran/%s' % dataset_grouping
    #cast column type because some id are of str and others 
    #float
    type_col={"userid":str,'retweet_userid':str,
              "in_reply_to_userid":str}
    df = pd.read_csv(path,dtype=type_col)
    li.append(df)
    return pd.concat(li, axis=0, ignore_index=True)


def generate_network(dataset):
    """
    @params: [dataset (Pandas Dataframe)]
    @returns: None

    Generates a directed graph networkx between users with edges defined by interaction
    types from INTERACTIONS using Pandas dataframe dataset. Saves the graph to
    filepath graph_out. Currently not decomposed, to reduce memory payload.
    """
    interactions_graph = nx.Graph()
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


def inducer(graph, node):
    nebs = list(nx.neighbors(graph, node))
    sub_nodes = nebs + [node]
    sub_g = nx.subgraph(graph, sub_nodes)
    out_counts = np.sum(np.array([len([list(nx.neighbors(graph,x))]) for x in sub_nodes]))
    return sub_g, out_counts, nebs

def complex_aggregator(x):
    return [np.min(x),np.std(x),np.var(x),np.mean(x),np.percentile(x,25),np.percentile(x,50),np.percentile(x,100),scipy.stats.skew(x),scipy.stats.kurtosis(x)]

def aggregator(x):
    return [np.sum(x),np.mean(x)]

def state_printer(x):
    print("-"*80)
    print(x)
    print("")

def sub_selector(old_features, new_features, pruning_threshold):

    print("Cross-temporal feature pruning started.")
    indices = set()
    for i in tqdm(range(0,old_features.shape[1])):
        for j in range(0, new_features.shape[1]):
            c = np.corrcoef(old_features[:,i], new_features[:,j])
            if abs(c[0,1]) > pruning_threshold:
                indices = indices.union(set([j]))
        keep = list(set(range(0,new_features.shape[1])).difference(indices))
        new_features = new_features[:,keep]
        indices = set()
    return new_features


class RecursiveExtractor:
    
    def __init__(self, args):
        self.args = args
        if self.args.aggregator == "complex":
            self.aggregator = complex_aggregator
        else:
            self.aggregator = aggregator
        self.multiplier = len(self.aggregator(0))
        
        #if the graph is already created
        if(args.graph):
            self.graph=nx.read_gpickle(args.graph)
        else:
            self.graph = generate_network(load_datasets(args.input))
            
        self.nodes = nx.nodes(self.graph)
        self.create_features()

    def basic_stat_extractor(self):
        self.base_features = []
        self.sub_graph_container = {}
        for node in tqdm(range(0,len(self.nodes))):
            sub_g, overall_counts, nebs = inducer(self.graph, node)
            in_counts = len(nx.edges(sub_g))
            self.sub_graph_container[node] = nebs
            deg = nx.degree(sub_g, node)
            trans = nx.clustering(sub_g, node)
            self.base_features.append([in_counts, overall_counts, float(in_counts)/float(overall_counts), float(overall_counts - in_counts)/float(overall_counts),deg, trans])
        self.features = {}
        self.features[0] = np.array(self.base_features)
        print("") 
        del self.base_features
    
    def single_recursion(self, i):
        features_from_previous_round = self.features[i].shape[1]
        new_features = np.zeros((len(self.nodes), features_from_previous_round*self.multiplier))
        for k in tqdm(range(0,len(self.nodes))):
            selected_nodes = self.sub_graph_container[k]
            main_features = self.features[i][selected_nodes,:]
            #if no neighbors just return a zero array of the right size
            try:
                new_features[k,:]= functools.reduce(lambda x,y: x+y,[self.aggregator(main_features[:,j]) for j in range(0,features_from_previous_round)]) 
            except:
                new_features[k,:]= np.zeros(new_features[k,:].shape)
        return new_features

    def do_recursions(self):
        for recursion in range(0,self.args.recursive_iterations):
            state_printer("Recursion round: " + str(recursion+1) + ".")
            new_features = self.single_recursion(recursion)
            new_features = sub_selector(self.features[recursion], new_features, self.args.pruning_cutoff)
            self.features[recursion+1] = new_features
            
        
        self.features=np.concatenate(([x for x in self.features.values()]),1)
        self.features = self.features / (np.max(self.features)-np.min(self.features))

    def binarize(self):
        self.new_features = []
        for x in tqdm(range(0,self.features.shape[1])):
            try:
                self.new_features = self.new_features + [pd.get_dummies(pd.qcut(self.features[:,x],self.args.bins, 
                                                                                labels = range(0,self.args.bins), duplicates = "drop"))]
            except:
                pass
        print(self.new_features)
        self.new_features = pd.concat(self.new_features, axis = 1)
        
    def dump_to_disk(self):
        #self.new_features.columns = map(lambda x: "x_" + str(x), range(0,self.new_features.shape[1]))
        #self.new_features.to_csv(self.args["recursive_features_output"], index = None)
        np.save(self.args.recursive_features_output,self.features)

    def create_features(self):
        state_printer("Basic node level feature extraction and induced subgraph creation started.")
        self.basic_stat_extractor()
        state_printer("Recursion started.")
        self.do_recursions()
        state_printer("Binary feature quantization started.")
        #self.binarize()
        #state_printer("Saving the raw features.")
        self.dump_to_disk()
        state_printer("The number of extracted features is: " + str(self.features.shape[1]) + ".")
        
    

args={"input":"iran_201901_1_tweets_csv_hashed_1.csv",
      "recursive_features_output": "features_iran_201901_1_tweets_csv_hashed_1",
      "recursive_iterations": 2,
      "aggregator": "complex",
      "pruning_cutoff": 0.5}
