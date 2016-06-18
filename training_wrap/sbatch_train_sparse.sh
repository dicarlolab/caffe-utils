#sp=0.01
sp=0.02
#sp=0.05
#sp=0.1
#sp=0.2
#sp=0.5
#sp=1
#sp=2
#sp=10
#for sp in 0.01 0.02 0.05 0.1 0.2 0.5 1 2 10
for sp in 0.01 0.02 0.05 0.1 0.2 0.5 1
#for sp in 1
#for sp in 2 10
do
    for k in 1 2 3 4
    do
        #sbatch --time=6-0:0:0 --qos="use-everything" --exclude=node027 -J ca${sp} -o /om/user/chengxuz/slurm_out_all/slurm_caffe_${sp}_${k}_%j.out --gres=gpu:1 --mem 35000 -c 5 ./train_sparse.sh ${sp} ${k}
        sbatch --time=6-0:0:0 --exclude=node027 -J ca${sp} -o /om/user/chengxuz/slurm_out_all/slurm_caffe_${sp}_${k}_%j.out --gres=gpu:1 --mem 35000 -c 5 ./train_sparse.sh ${sp} ${k}
    done
done
