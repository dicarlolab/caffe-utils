#!/bin/bash

net="/om/user/chengxuz/caffe-utils/training_wrap/train_val_alexnet.prototxt"
snapshot_prefix="/om/user/chengxuz/caffe_install/snapshot/alexnet"

dp_params="{'data_path':'/om/user/yamins/.skdata/imagenet/ChallengeSynsets2013_offline_23d6ee636ade8ad3912204410d1acc23c10357cf/cache/images_cache_e86d39462641ebc8870926aa16629eae5ca11c78_random_0_hdf5/data.raw','data_key':'data','label_path':'/om/user/chengxuz/my_meta_data/labels_for_images_cache_e86d39462641ebc8870926aa16629eae5ca11c78_random_0_hdf5/labels.hdf5','label_key':['labels'],'cache_type':'hdf5','batch_size':256,'val_len':50000}"
preproc="{'data_mean':'/om/user/hyo/caffe/imagenet_mean.npy','crop_size':227,'do_img_flip':True,'noise_level':10}"
base_lr=0.01
lr_policy="step"
gamma=0.1
stepsize=100000
display=20
max_iter=450000
momentum=0.9
weight_decay=0.0005
snapshot=10000

python train.py --net ${net} --display ${display} --base_lr ${base_lr} --lr_policy ${lr_policy} --gamma ${gamma} --stepsize ${stepsize} --max_iter ${max_iter} --momentum ${momentum}  --weight_decay ${weight_decay} --snapshot ${snapshot} --snapshot_prefix ${snapshot_prefix} --dp-params ${dp_params} --preproc ${preproc} --multi_core 4
