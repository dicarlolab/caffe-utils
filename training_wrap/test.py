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

    if not options.save_file==None:
        all_data_dict   = {}
        #names           = ['conv4']
        names 	= ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7', 'fc8'] # for sparsity network
        img_num         = 5760
        
    for iter in range(max_iter):
	# load data batch
        ind = iter * bsize
        _data, _labels = dp.get_batch_test(ind, perm_flag=(options.perm_flag==1))

	# set data as input
        #N.blobs['data'].data[...] = _data
        N.blobs['data'].data[:bsize] = _data
        for label_key in _labels.keys():
            #N.blobs[label_key].data[...] = _labels[label_key].reshape(N.blobs[label_key].data.shape)
            N.blobs[label_key].data[:bsize] = _labels[label_key].reshape(N.blobs[label_key].data[:bsize].shape)

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

        if not options.save_file==None:
            for layer_name in names:
                now_data_tmp    = N.blobs[layer_name].data[:bsize]
                now_data_shape  = now_data_tmp.size/bsize

                #print(layer_name)
                #print(now_data_shape)
                #print(net.blobs[layer_name].data.shape)

                if layer_name not in all_data_dict:
                    all_data_dict[layer_name]   = np.zeros([img_num, now_data_shape])
                all_data_dict[layer_name][iter*bsize:(iter+1)*bsize]       = now_data_tmp.reshape([bsize, now_data_shape])

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

    if not options.save_file==None:
        file_indx_now   = 0
        for layer_name in all_data_dict:
            f = h5py.File(options.save_file + 'output_' + str(file_indx_now) + '.hdf5', "w")
            f.create_dataset(layer_name, data = all_data_dict[layer_name])
            f.close()
            file_indx_now   = file_indx_now + 1

    # remove temp file
    #os.remove(filename_temp)

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-n", "--net", dest="net")
    parser.add_option("-f", "--weights", dest="weights", default=None)
    parser.add_option("--dp-params", dest="dp_params", default='', help="Data provider params")
    parser.add_option("--preproc", dest="preproc", default='', help="Data preprocessing params")
    parser.add_option("--perm-flag", dest="perm_flag", default=1, help="The flag for whether permutating the test data", type=int)
    parser.add_option("--save-file", dest="save_file", default=None, help="The flag for whether saving the responses")
    parser.add_option("--max-iter", dest="max_iter", default=None, type=int)

    main(parser)
