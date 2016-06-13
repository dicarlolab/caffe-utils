import numpy as np
import h5py
import cPickle
import caffe
#caffe_root = "/om/user/hyo/usr7/pkgs/caffe/"
caffe_root = "/om/user/chengxuz/caffe_install/caffe/"
import sys
sys.path.insert(0, caffe_root + 'python')
from optparse import OptionParser
import ast
from data import DataProvider
import time
import os
import subprocess
import pdb
import multiprocessing
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool

caffe.set_mode_gpu()
caffe.set_device(0)

options     = ''
N           = ''
dp          = ''
loss        = ''
prevs       = ''
lock        = multiprocessing.Lock()
data_time   = time.time()

def train_iter(iter):
    global data_time

    # clear param diffs
    for layer_name, lay in zip(N._layer_names, N.layers):
        #print(layer_name, lay)
        for blobind, blob in enumerate(lay.blobs):
            blob.diff[:] = 0
    # clear loss  
    for k in loss.keys():
        loss[k] = 0
   
    #print('Clear finished!')
    # update weights at every <iter_size> iterations 
    for i in range(dp.iter_size):
    # load data batch
        ind = iter * dp.batch_size * dp.iter_size + i * dp.batch_size
        _data, _labels = dp.get_batch(ind)

        lock.acquire()
        load_time = time.time() - data_time

    # set data as input
        N.blobs['data'].data[...] = _data
        for label_key in _labels.keys():
            N.blobs[label_key].data[...] = _labels[label_key].reshape(N.blobs[label_key].data.shape)

    # Forward
        t0 = time.time()
        out = N.forward()
        forward_time = time.time()
    # Backward
        N.backward()
        backward_time = time.time() - forward_time
        forward_time -= t0

        for k in out.keys():
            loss[k] += np.copy(out[k])

    # learning rate schedule
    if options.lr_policy == "step":
        learning_rate = options.learning_rate * (options.gamma**(iter/options.stepsize))

    # print output loss
    print "Iteration", iter, "(lr: {0:.4f})".format( learning_rate )
    for k in np.sort(out.keys()):
        loss[k] /= dp.iter_size
        print "Iteration", iter, ", ", k, "=", loss[k]
    sys.stdout.flush()

    # update filter parameters
    t0 = time.time()
    #print('Blob diff related!')
    for layer_name, lay in zip(N._layer_names, N.layers):
        #print(layer_name, lay)
        #pdb.set_trace()
        for blobind, blob in enumerate(lay.blobs):
            #print(blobind, blob)
            diff = blob.diff[:]
            key = (layer_name, blobind)
            if key in prevs:
                previous_change = prevs[key]
            else:
                previous_change = 0
            #print('Blob diff related finished!')
            lr = learning_rate
            wd = options.weight_decay
            if blobind == 1:
                lr = 2 *lr
                wd = 0
            if lay.type == "BatchNorm":
                lr = 0
                wd = 0
            change = options.momentum * previous_change - lr * diff / dp.iter_size - lr * wd * blob.data[:]

            blob.data[:] += change
            prevs[key] = change
    update_time = time.time() - t0
    print "loading: {0:.2f}, forward: {1:.2f}, backward: {2:.2f}, update: {3:.2f}".format(load_time, forward_time, backward_time, update_time)
    
    # save weights 
    if iter % options.snapshot == 0:
        print 'Snapshotting for iteration ' + str(iter)
        N.save(options.snapshot_prefix + '_iter_' +  str(iter) + '.caffemodel')

    # test on validation set
    if iter % options.test_interval == 0:
        print 'Test for iteration ' + str(iter)
        weights_test = options.snapshot_prefix + '_iter_' +  str(iter) + '.caffemodel'
        if not os.path.exists(weights_test):
            N.save(options.snapshot_prefix + '_iter_' +  str(iter) + '.caffemodel')
        command_for_test = options.command_for_test + ' test.sh ' + options.net + ' ' + weights_test + ' "' + options.dp_params + '" "' + options.preproc + '"'
        subprocess.call(command_for_test, shell=True)

    
    data_time = time.time()
    lock.release()

def main(parser):
    # retrieve options
    global options
    global N
    global dp
    global loss
    global prevs

    (options, args) = parser.parse_args()
    '''
    net = options.net
    display = options.display
    base_lr = options.learning_rate
    lr_policy = options.lr_policy
    gamma = options.gamma
    stepsize = options.stepsize
    max_iter = options.max_iter
    momentum = options.momentum
    iter_size = options.iter_size
    weight_decay = options.weight_decay
    iter_snapshot = options.iter_resume
    snapshot = options.snapshot
    snapshot_prefix = options.snapshot_prefix
    test_interval = options.test_interval
    command_for_test = options.command_for_test
    weights = options.weights
    '''

    dp_params = ast.literal_eval(options.dp_params)
    preproc = ast.literal_eval(options.preproc)

    #multi_core = options.multi_core

    # Data provider
    dp = DataProvider(dp_params, preproc)

    '''
    isize = dp.batch_size
    iter_size = dp.iter_size
    bsize = isize * iter_size
    '''

    '''
    if multi_core>1:
        # iter_size bigger than 1 not supported in multi_core version
        iter_size   = 1
    '''

    # Load the net
    if options.weights is not None:
        print "Loading weights from " + options.weights
        N = caffe.Net(options.net, options.weights, caffe.TRAIN)
    else:
        if options.iter_resume==0:
            N = caffe.Net(options.net, caffe.TRAIN)
        else:
            print "Resuming training from " + options.snapshot_prefix + '_iter_' + str(options.iter_resume) + '.caffemodel'
            N = caffe.Net(options.net, options.snapshot_prefix + '_iter_' +  str(options.iter_resume) + '.caffemodel', caffe.TRAIN)

    print('Now loading finished!')
    # Save the net if training gets stopped
    import signal
    def sigint_handler(signal, frame):
        print 'Training paused...'
        print 'Snapshotting for iteration ' + str(iter)
        N.save(snapshot_prefix + '_iter_' +  str(iter) + '.caffemodel')
        sys.exit(0)
    signal.signal(signal.SIGINT, sigint_handler)


    print('Now Training started!')
    # Start training
    loss = { key: 0 for key in N.outputs }
    prevs = {}

    if options.multi_core==1:
        for iter in range(options.iter_resume, options.max_iter):
            train_iter(iter)
    else:
        pool = ThreadPool(options.multi_core)
        results = pool.map(train_iter, range(options.iter_resume, options.max_iter))
        pool.close()
        pool.join()

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-n", "--net", dest="net")
    parser.add_option("-d", "--display", dest="display", default=1, type=int)
    parser.add_option("-l", "--base_lr", dest="learning_rate", type=float)
    parser.add_option("-c", "--lr_policy", dest="lr_policy", default=None)
    parser.add_option("-g", "--gamma", dest="gamma", default=None, type=float)
    parser.add_option("-z", "--stepsize", dest="stepsize", default=None, type=int)
    parser.add_option("-e", "--max_iter", dest="max_iter", type=int)
    parser.add_option("-m", "--momentum", dest="momentum", type=float)
    parser.add_option("-i", "--iter_size", dest="iter_size", default=1, type=int)
    parser.add_option("-o", "--multi_core", dest="multi_core", default=1, type=int)
    parser.add_option("-w", "--weight_decay", dest="weight_decay", type=float)
    parser.add_option("-s", "--snapshot", dest="snapshot", default=500, type=int)
    parser.add_option("-p", "--snapshot_prefix", dest="snapshot_prefix", default=None)
    parser.add_option("-t", "--test-interval", dest="test_interval", default=1000, type=int)
    parser.add_option("--command-for-test", dest="command_for_test", default='sbatch --gres=gpu:1 --mem=1500')
    parser.add_option("-r", "--iter_resume", dest="iter_resume", default=0, type=int)
    parser.add_option("-f", "--weights", dest="weights", default=None)
    parser.add_option("--dp-params", dest="dp_params", default='', help="Data provider params")
    parser.add_option("--preproc", dest="preproc", default='', help="Data preprocessing params")

    main(parser)
