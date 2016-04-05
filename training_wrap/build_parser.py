import argparse
import os

def build_parser():
    """
    Function for parsing arguments
    """

    parser = argparse.ArgumentParser(description='Train the networks using caffe')
    parser.add_argument(
        "--model_def",
        default='/om/user/chengxuz/sparse_CNN/alex_mine/train_val.prototxt_lmdb',
        help="Model definition file.")
    parser.add_argument("--data_path", 
        default='', 
        help="")
