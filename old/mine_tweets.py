#
# CS224W Fall 2019-2020
# @Jason Zheng, Guillaume Nervo, Jestin Ma
#
import numpy as np
import math
import networkx as nx
import pandas as pd
import random
import sys

def mine_tweets(k_users, f_fanout):
    """
    @params: [k_users (int), f_fanout (int)]
    @returns: None

    Mines twitter API to obtain users and tweets from random threads
    """
    # 18 digits
    # https://developer.twitter.com/en/docs/tweets/sample-realtime/api-reference


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: mine_tweets.py [k_users] [f_fanout]')
    else:
        mine_tweets(sys.argv[1], sys.argv[2])
