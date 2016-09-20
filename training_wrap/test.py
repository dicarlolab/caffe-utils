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
import os
import cPickle
#from lockfile import LockFile

caffe.set_mode_gpu()
caffe.set_device(0)

def main(parser):
    # retrieve options
    (options, args) = parser.parse_args()
    net = options.net
    weights = options.weights
    dp_params = ast.literal_eval(options.dp_params)
    preproc = ast.literal_eval(options.preproc)
    max_iter = options.max_iter

    # output filename
    filename = 'output_test/' + (weights.split('/')[-1]).split('_iter')[0] + '.p'
    iter_test = int(((weights.split('/')[-1]).split('_iter_')[1]).split('.')[0])
    filename_temp = 'output_test/temp_' + (weights.split('/')[-2]) + (weights.split('/')[-1]).split('.')[0] + '.txt'
    f = open(filename_temp, 'w')

    # Data provider
    dp = DataProvider(dp_params, preproc)
    bsize = dp.batch_size
    if max_iter is None:
        max_iter = dp.val_len / bsize

    # Load the net
    N = caffe.Net(net, weights, caffe.TEST)
    print >> f, "Testing from " + weights

    # Save the net if training gets stopped
    import signal
    def sigint_handler(signal, frame):
        print >> f, 'Training paused...'
        print >> f, 'Snapshotting for iteration ' + str(iter)
        N.save(snapshot_prefix + '_iter_' +  str(iter) + '.caffemodel')
        sys.exit(0)
    signal.signal(signal.SIGINT, sigint_handler)

    # Start training
    loss = { key: 0 for key in N.outputs }
    prevs = {}
    for iter in range(max_iter):
	# load data batch
        ind = iter * bsize
        _data, _labels = dp.get_batch_test(ind)

	# set data as input
        N.blobs['data'].data[...] = _data
        for label_key in _labels.keys():
            N.blobs[label_key].data[...] = _labels[label_key].reshape(N.blobs[label_key].data.shape)

	# Forward
        out = N.forward()

        # add to loss
        for k in out.keys():
            loss[k] += np.copy(out[k])

        # print output loss to temp file
        print >> f, "Batch #", iter
        for k in np.sort(out.keys()):
            print >> f, "Batch #", iter, ", ", k, "=", out[k]
        sys.stdout.flush()

    # print output loss to temp file
    for k in np.sort(loss.keys()):
        print >> f, "Mean test loss,", k, ": {0:.8f}".format(loss[k] / max_iter)
    f.close()

    # Save to test output file
    #lock = LockFile(filename)
    #lock.acquire()
    if os.path.exists(filename):
        r = cPickle.load(open(filename))
    else:
        r = {'iter_test': {}, 'test': {}}
    for k in loss.keys():
        if k not in r['iter_test']:
            r['iter_test'][k] = np.array([iter_test])
            r['test'][k] = np.array([loss[k] / max_iter])
        else:
            r['iter_test'][k] = np.append(r['iter_test'][k], iter_test)
            r['test'][k] = np.append(r['test'][k], loss[k] / max_iter)[np.argsort(r['iter_test'][k])]
            r['iter_test'][k] = np.sort(r['iter_test'][k])
    with open(filename, 'w') as f:
        cPickle.dump(r, f)
    #lock.release()

    # remove temp file
    #os.remove(filename_temp)

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-n", "--net", dest="net")
    parser.add_option("-f", "--weights", dest="weights", default=None)
    parser.add_option("--dp-params", dest="dp_params", default='', help="Data provider params")
    parser.add_option("--preproc", dest="preproc", default='', help="Data preprocessing params")
    parser.add_option("--max-iter", dest="max_iter", default=None, type=int)

    main(parser)
