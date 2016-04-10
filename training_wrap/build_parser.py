from python_util.util import *
from python_util.data import *
from python_util.options import *
from python_util.gpumodel import *
from optparse import OptionParser
import sys

from convdata import (ImageDataProvider, CIFARDataProvider, DummyConvNetLogRegDataProvider, 
                      CroppedGeneralDataProvider, CroppedGeneralDataRandomProvider, 
                      CroppedGeneralDataMapProvider, CroppedImageAndVectorProvider)

def build_parser():
    """
    Function for parsing arguments
    Parser borrowed from archconvnet
    """

    '''
    parser = argparse.ArgumentParser(description='Train the networks using caffe')
    parser.add_argument(
        "--model_def",
        default='/om/user/chengxuz/sparse_CNN/alex_mine/train_val.prototxt_lmdb',
        help="Model definition file.")
    parser.add_argument(
        "--data_path",
        default='/om/user/yamins/.skdata/imagenet/ChallengeSynsets2013_offline_23d6ee636ade8ad3912204410d1acc23c10357cf/cache/images_cache_e86d39462641ebc8870926aa16629eae5ca11c78_random_0_hdf5', 
        help="Training data path")
    parser.add_argument("--imageSize", default=256, help="Image size")
    parser.add_argument("--imageChannel", default=3, help="Image channel")
    parser.add_argument("", default=, help="")
    '''

    op = OptionsParser()
    op.add_option("load-file", "load_file", StringOptionParser, "Load file", default="", excuses=OptionsParser.EXCUSE_ALL)
    op.add_option("train-range", "train_batch_range", RangeOptionParser, "Data batch range: training")
    op.add_option("test-range", "test_batch_range", RangeOptionParser, "Data batch range: testing")
    op.add_option("data-provider", "dp_type", StringOptionParser, "Data provider", default="default")
    op.add_option("test-freq", "testing_freq", IntegerOptionParser, "Testing frequency", default=25)
    op.add_option("epochs", "num_epochs", IntegerOptionParser, "Number of epochs", default=500)
    op.add_option("start-epoch", "starting_epoch", IntegerOptionParser, "Epoch to begin with", default=1)
    op.add_option("start-batch", "starting_batch", IntegerOptionParser, "Batch to begin with", default=0)
    op.add_option("data-path", "data_path", StringOptionParser, "Data path")
    
    op.add_option("max-test-err", "max_test_err", FloatOptionParser, "Maximum test error for saving")
    op.add_option("test-only", "test_only", BooleanOptionParser, "Test and quit?", default=0)
    op.add_option("test-one", "test_one", BooleanOptionParser, "Test on one batch at a time?", default=1)
    op.add_option("save-initial", "save_initial", BooleanOptionParser, "Save initial state as checkpoint before training?", default=0)
    op.add_option("force-save", "force_save", BooleanOptionParser, "Force save before quitting", default=0)
    op.add_option("gpu", "gpu", ListOptionParser(IntegerOptionParser), "GPU override")
    ####### db configs #######
    op.add_option("save-db", "save_db", BooleanOptionParser, "Save checkpoints to mongo database?", default=0)
    op.add_option("save-filters", "save_filters", BooleanOptionParser, "Save filters to database?", default=1)
    op.add_option("save-recent-filters", 'save_recent_filters', BooleanOptionParser, "Save recent filters to database?", default=1)
    op.add_option("saving-freq", "saving_freq", IntegerOptionParser, 
                  "Frequency for saving filters to db filesystem, as a multiple of testing-freq", 
                  default=1)
    op.add_option("checkpoint-fs-host", "checkpoint_fs_host", StringOptionParser, "Host for Saving Checkpoints to DB", default="localhost")
    op.add_option("checkpoint-fs-port", "checkpoint_fs_port", IntegerOptionParser, "Port for Saving Checkpoints to DB", default=27017)
    op.add_option("checkpoint-db-name", "checkpoint_db_name", StringOptionParser,
            "Name for mongodb database for saved checkpoints", default="convnet_checkpoint_db")
    op.add_option("checkpoint-fs-name", "checkpoint_fs_name", StringOptionParser,
            "Name for gridfs FS for saved checkpoints", default="convnet_checkpoint_fs")
    op.add_option("experiment-data", "experiment_data", JSONOptionParser, "Data for grouping results in database", default="")
    op.add_option("load-query", "load_query", JSONOptionParser, "Query for loading checkpoint from database", default="", excuses=OptionsParser.EXCUSE_ALL)        

    op.add_option("mini", "minibatch_size", IntegerOptionParser, "Minibatch size", default=128)
    op.add_option("layer-def", "layer_def", StringOptionParser, "Layer definition file", set_once=False)
    op.add_option("layer-params", "layer_params", StringOptionParser, "Layer parameter file")
    op.add_option("layer-path", "layer_path", StringOptionParser, "Layer file path prefix", default="")
    op.add_option("check-grads", "check_grads", BooleanOptionParser, "Check gradients and quit?", default=0, excuses=['data_path', 'train_batch_range','test_batch_range'])
    op.add_option("crop-border", "crop_border", ListOptionParser(IntegerOptionParser), "Cropped DP: crop border size", default=[4], set_once=True)        
    op.add_option("multiview-test", "multiview_test", BooleanOptionParser, "Cropped DP: test on multiple patches?", default=0)
    op.add_option("inner-size", "inner_size", IntegerOptionParser, "Cropped DP: crop size (0 = don't crop)", default=0, set_once=True)
    op.add_option("conv-to-local", "conv_to_local", ListOptionParser(StringOptionParser), "Convert given conv layers to unshared local", default=[])
    op.add_option("unshare-weights", "unshare_weights", ListOptionParser(StringOptionParser), "Unshare weight matrices in given layers", default=[])
    op.add_option("conserve-mem", "conserve_mem", BooleanOptionParser, "Conserve GPU memory (slower)?", default=0)
    op.add_option("color-noise", "color_noise", FloatOptionParser, "Add PCA noise to color channels with given scale", default=0.0)
    op.add_option("test-out", "test_out", StringOptionParser, "Output test case predictions to given path", default="", requires=['logreg_name', 'multiview_test'])
    op.add_option("logreg-name", "logreg_name", StringOptionParser, "Logreg cost layer name (for --test-out)", default="")
    op.add_option("scalar-mean", "scalar_mean", FloatOptionParser, "Subtract this scalar from image (-1 = don't)", default=-1)

    # ----------options related with general data provider----
    op.add_option("random-seed", "random_seed", IntegerOptionParser,
            "Random Seed", default=0)
    op.add_option("dp-params", "dp_params", JSONOptionParser, "Data provider paramers", default='')
    op.add_option("write-features", "write_features", ListOptionParser(StringOptionParser),
          "Write features from given layer(s)", default="", requires=['feature-path'])                    
    op.add_option("feature-path", "feature_path", StringOptionParser,
          "Write features to this path (to be used with --write-features)", default="")

    op.add_option('load-layers', "load_layers", ListOptionParser(StringOptionParser), 
                  "Load these layers", default="")

    op.add_option("img-flip", "img_flip", BooleanOptionParser,
            "Whether flip training image", default=True )

    op.delete_option('max_test_err')
    op.options["testing_freq"].default = 57
    op.options["num_epochs"].default = 50000
    op.options['dp_type'].default = None

    DataProvider.register_data_provider('dummy-lr-n', 'Dummy ConvNet logistic regression', DummyConvNetLogRegDataProvider)
    DataProvider.register_data_provider('labeled-data', 'Labeled Data', LabeledDataProvider)
    DataProvider.register_data_provider('labeled-data-trans', 'Labeled Data Trans', LabeledDataProviderTrans)        
    DataProvider.register_data_provider('image', 'JPEG-encoded image data provider', ImageDataProvider)
    DataProvider.register_data_provider('cifar', 'CIFAR-10 data provider', CIFARDataProvider)
    DataProvider.register_data_provider('general-cropped', 'Cropped General', CroppedGeneralDataProvider)
    DataProvider.register_data_provider('general-map-cropped', 'Cropped General Mop', CroppedGeneralDataMapProvider)
    DataProvider.register_data_provider('general-cropped-rand', 'Cropped General Random',
                                        CroppedGeneralDataRandomProvider)                
    DataProvider.register_data_provider('general-cropped-vectors', 'Cropped General_Vector',
                                        CroppedImageAndVectorProvider)        


    return op

if __name__ == "__main__":
    op = build_parser()
    op, load_dic = IGPUModel.parse_options(op)

    dp_params = {}
    filename_options = []
    if op.options["dp_params"].value_given:
        dp_params.update(op.options["dp_params"].value)        
    for v in ('crop_border', 'img_flip', 'color_noise', 'multiview_test', 'inner_size', 'scalar_mean', 'minibatch_size'):
        dp_params[v] = op.get_value(v)

    dp_model    = IGPUModel("ConvNet", op, load_dic, filename_options, dp_params=dp_params)
    test_data   = dp_model.get_next_batch()
