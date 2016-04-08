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
    parser.add_argument(
        "--data_path",
        default='/om/user/yamins/.skdata/imagenet/ChallengeSynsets2013_offline_23d6ee636ade8ad3912204410d1acc23c10357cf/cache/images_cache_e86d39462641ebc8870926aa16629eae5ca11c78_random_0_hdf5', 
        help="Training data path")
