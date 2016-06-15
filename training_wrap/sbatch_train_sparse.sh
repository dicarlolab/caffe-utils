#sp=0.01
sp=0.02
#sp=0.05
#sp=0.1
#sp=0.2
#sp=0.5
#sp=1
#sp=2
#sp=10
#for k in 1 2 3 4
for k in 1
do
    sbatch --qos="use-everything" --exclude=node027 -J ca${sp} -o /om/user/chengxuz/slurm_out_all/slurm_caffe_${sp}_${k}_%j.out --gres=gpu:1 --mem 35000 -c 5 ./train_sparse.sh ${sp}
done
