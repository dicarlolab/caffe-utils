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
#for sp in 0
#for sp in h1
#for sp in h1 h0.1 h0.01
#for sp in h10 h3 h0.3 h0.03 h1 h0.1 h0.01
#for sp in h100 h30 13 15
#for sp in h200
#for sp in h300
#for sp in 100
#for sp in 300
#for sp in h250
#for sp in h225
#for sp in 200
#for sp in h212 h206
#for sp in 250 150
#for sp in h125 h150 h175
#for sp in 125 175
#for sp in 175
#for sp in h158 h166
#for sp in 133 141
#for sp in 144 147 h169 h172
#for sp in 30
#for sp in 0.01 0.02 0.05 0.1 0.2 0.5 1
#for sp in 0.01
#for sp in 2 10
for sp in 0
do
    for k in 1 2 3 4
    #for k in 1
    #for k in 2 3 4
    do
        #sbatch --time=6-0:0:0 --qos="use-everything" --exclude=node027 -J ca${sp} -o /om/user/chengxuz/slurm_out_all/slurm_caffe_${sp}_${k}_%j.out --gres=gpu:1 --mem 35000 -c 5 ./train_sparse.sh ${sp} ${k}

        #sbatch --exclude=node020 -J ca${sp} -o /om/user/chengxuz/slurm_out_all/slurm_caffe_${sp}_${k}_%j.out --gres=gpu:1 --mem 35000 -c 5 ./test_sparse.sh ${sp} ${k}
        #sh ./test_sparse.sh ${sp} ${k}
        #sbatch --time=6-0:0:0 -J ca${sp} -o /om/user/chengxuz/slurm_out_all/slurm_caffe_${sp}_${k}_%j.out --gres=gpu:titan-x:1 --mem 35000 -c 5 ./train_sparse.sh ${sp} ${k}
        #sbatch --time=6-0:0:0 -J ca${sp} -o /om/user/chengxuz/slurm_out_all/slurm_caffe_${sp}_${k}_%j.out --gres=gpu:1 --mem 35000 -c 5 ./train_sparse.sh ${sp} ${k}
        #sbatch --time=6-0:0:0 -J ca${sp} -o /om/user/chengxuz/slurm_out_all/slurm_caffe_${sp}_${k}_%j.out --gres=gpu:1 --mem 35000 -c 5 ./train_sparse.sh ${k}
        sbatch --time=6-0:0:0 -J ca_f${sp} -o /om/user/chengxuz/slurm_out_all/slurm_caffe_${sp}_${k}_%j.out --gres=gpu:titan-x:1 --mem 35000 -c 5 ./train_sigmoid.sh ${k}
    done
done
