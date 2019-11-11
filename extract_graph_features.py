#!/usr/bin/python3
"""
Extract several per-node features from a given snap TNEANet graph object.

@author: jestinm
"""

import argparse
import snap


def extract_node_features(graph, no_graph_info=False):
    """
    """
    for node in graph.Nodes():
        print(graph.GetIntAttrDatN(node, "followers_count"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", help="snap_graph")
    parser.add_argument('--no_graph_info', action='store_true', help="whether to omit graph-level features")
    parser.add_argument("outfile", help="file to write the csv of node-to-features")
    args = parser.parse_args()

    graph = snap.TNEANet.Load(snap.TFIn(args.graph))
    df = extract_node_features(graph, args.no_graph_info)
    # df.to_csv(args.outfile)
