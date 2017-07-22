#sp=0.01
sp=0.02
#sp=0.05
#sp=0.1
#sp=0.2
#sp=0.5
#sp=1
#sp=2
#sp=10
cache_dir=/om/user/chengxuz/RDM_results/hvm_response/pkl_result_reg_all_cache/caffe_alexnet_sp_
save_dir=/om/user/chengxuz/RDM_results/hvm_response/pkl_result_reg_all/caffe_alexnet_sp_

#for sp in 0.5 1 2 10 0 
#for sp in 0.01 0.02 0.05 0.1 0.2 0.5 1 2 10 0
#for sp in 0.02
#for sp in 0
#for sp in h1
for sp in h1 h0.1 h0.01 h10 h3 h0.3 h0.03
#for sp in h1 h0.1 h0.01
#for sp in h10 h3 h0.3 h0.03
#for sp in 0.01 0.02 0.05 0.1 0.2 0.5 1
#for sp in 0.01
#for sp in 2 10
do
    for k in 1 2 3 4
    #for k in 1
    do
        #sbatch --time=6-0:0:0 --qos="use-everything" --exclude=node027 -J ca${sp} -o /om/user/chengxuz/slurm_out_all/slurm_caffe_${sp}_${k}_%j.out --gres=gpu:1 --mem 35000 -c 5 ./train_sparse.sh ${sp} ${k}

        #sbatch --qos="use-everything" --exclude=node020 -J hvm_ca${sp} -o /om/user/chengxuz/slurm_out_all/slurm_caffe_${sp}_${k}_%j.out --gres=gpu:1 --mem 55000 -c 5 ./generate_response.sh ${sp} ${k} 200000
        #./generate_response.sh ${sp} ${k} 200000
        #sh ./test_sparse.sh ${sp} ${k}
        #sbatch --time=6-0:0:0 --exclude=node027 -J ca${sp} -o /om/user/chengxuz/slurm_out_all/slurm_caffe_${sp}_${k}_%j.out --gres=gpu:1 --mem 35000 -c 5 ./train_sparse.sh ${sp} ${k}

        for layer_indx in $(seq 1 8)
        #for layer_indx in 2
        do
            #python /om/user/chengxuz/robustness/codes_generating/script_linear_regression_check_with_noise_norm.py 8 ${layer_indx} obj ${save_dir}${sp}_${k}_200000/ ${sp} ${k} Notused 100 1 1 1 ${cache_dir}${sp}_${k}_200000/ 
            #sbatch -J li_ca${sp} --qos="use-everything" -o /om/user/chengxuz/slurm_out_all/slurm_caffe_linear_${sp}_${k}_%j.out --mem 35000 /om/user/chengxuz/robustness/scripts_analyze/sh_script_linear.sh 8 ${layer_indx} obj ${save_dir}${sp}_${k}_200000/ ${sp} ${k} Notused 100 1 1 1 ${cache_dir}${sp}_${k}_200000/ 
            for sample_num in 500 1000 1200 1400 1600 1800 2000 2500
            do
                #sbatch -J li_ca${sp} --qos="use-everything" -o /om/user/chengxuz/slurm_out_all/slurm_caffe_linear_${sp}_${k}_%j.out --mem 35000 /om/user/chengxuz/robustness/scripts_analyze/sh_script_linear.sh 8 ${layer_indx} obj ${save_dir}${sp}_${k}_200000_${sample_num}/ ${sp} ${k} Notused 100 0 1 1 ${cache_dir}${sp}_${k}_200000/ ${sample_num}
                sbatch -J li_ca${sp} -o /om/user/chengxuz/slurm_out_all/slurm_caffe_linear_${sp}_${k}_%j.out --mem 35000 /om/user/chengxuz/robustness/scripts_analyze/sh_script_linear.sh 8 ${layer_indx} obj ${save_dir}${sp}_${k}_200000_${sample_num}/ ${sp} ${k} Notused 100 0 1 1 ${cache_dir}${sp}_${k}_200000_${sample_num}/ ${sample_num}
            done
        done

        
    done
done

        :'
        save_address=/om/user/chengxuz/RDM_results/hvm_response/hdf5_file/caffe_alexnet_sp_${sp}_${k}_200000/output_
        mv ${save_address}5.hdf5 ${save_address}re_1.hdf5
        mv ${save_address}4.hdf5 ${save_address}re_2.hdf5
        mv ${save_address}3.hdf5 ${save_address}re_3.hdf5
        mv ${save_address}7.hdf5 ${save_address}re_4.hdf5
        mv ${save_address}6.hdf5 ${save_address}re_5.hdf5
        mv ${save_address}0.hdf5 ${save_address}re_6.hdf5
        mv ${save_address}1.hdf5 ${save_address}re_7.hdf5
        mv ${save_address}2.hdf5 ${save_address}re_8.hdf5

        for layer_indx in $(seq 1 8)
        #for layer_indx in 2
        do
            #python /om/user/chengxuz/robustness/codes_generating/script_linear_regression_check_with_noise_norm.py 8 ${layer_indx} obj ${save_dir}${sp}_${k}_200000/ ${sp} ${k} Notused 100 1 1 1 ${cache_dir}${sp}_${k}_200000/ 
            #sbatch -J li_ca${sp} --qos="use-everything" -o /om/user/chengxuz/slurm_out_all/slurm_caffe_linear_${sp}_${k}_%j.out --mem 35000 /om/user/chengxuz/robustness/scripts_analyze/sh_script_linear.sh 8 ${layer_indx} obj ${save_dir}${sp}_${k}_200000/ ${sp} ${k} Notused 100 1 1 1 ${cache_dir}${sp}_${k}_200000/ 
            for sample_num in 500 1000 1200 1400 1600 1800 2000 2500
            do
                #sbatch -J li_ca${sp} --qos="use-everything" -o /om/user/chengxuz/slurm_out_all/slurm_caffe_linear_${sp}_${k}_%j.out --mem 35000 /om/user/chengxuz/robustness/scripts_analyze/sh_script_linear.sh 8 ${layer_indx} obj ${save_dir}${sp}_${k}_200000_${sample_num}/ ${sp} ${k} Notused 100 0 1 1 ${cache_dir}${sp}_${k}_200000/ ${sample_num}
                sbatch -J li_ca${sp} -o /om/user/chengxuz/slurm_out_all/slurm_caffe_linear_${sp}_${k}_%j.out --mem 35000 /om/user/chengxuz/robustness/scripts_analyze/sh_script_linear.sh 8 ${layer_indx} obj ${save_dir}${sp}_${k}_200000_${sample_num}/ ${sp} ${k} Notused 100 0 1 1 ${cache_dir}${sp}_${k}_200000_${sample_num}/ ${sample_num}
            done
        done

        '
