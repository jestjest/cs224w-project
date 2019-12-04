# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 10:17:37 2019

@author: guill
"""

import argparse

def parameter_parser():

    """
    A method to parse up command line parameters. By default it gives an embedding of the Facebook tvshow network.
    The default hyperparameters give a good quality representation and good candidate cluster means without grid search.
    """

    parser = argparse.ArgumentParser(description = "Run RefeX.")

    #------------------------------------------------------------------
    # Input and output file parameters.
    #------------------------------------------------------------------

    parser.add_argument("--input",
                        nargs = "?",
                        default = "./input/tvshow_edges.csv",
	                help = "Input graph path.")
    
    parser.add_argument("--graph",
                        nargs = "?",
                        default = None,
	                help = "Input graph.")

    parser.add_argument("--recursive_features_output",
                        nargs = "?",
                        default = "./output/features/tvhsow_features.csv",
	                help = "Embeddings path.")


    #-----------------------------------------------------------------------
    # Recursive feature extraction parameters.
    #-----------------------------------------------------------------------

    parser.add_argument("--recursive_iterations",
                        type = int,
                        default = 3,
	                help = "Number of recursions.")

    parser.add_argument("--aggregator",
                        nargs = "?",
                        default = "simple",
    	                 help = "Aggregator statistics extracted.")

    parser.add_argument("--bins",
                        type = int,
                        default = 4,
	                help = "Number of quantization bins.")

    parser.add_argument("--pruning_cutoff",
                        type = float,
                        default = 0.5,
	                help = "Absolute correlation for feature pruning.")

    
    return parser.parse_args()

from refeX import RecursiveExtractor

if __name__ == "__main__":
    args = parameter_parser()
    RecursiveExtractor(args)