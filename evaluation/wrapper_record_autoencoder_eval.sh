job_list="Job_list_autoencoder_eval.txt"
sample_list="../Model_Lists/sample_list_autoencoder_eval.txt"

if [ -f ${job_list} ]
then
    rm ${job_list}
fi


for sample_path in $( cat ${sample_list} )
do
    if [ $( ls ${sample_path} | wc -l ) -lt 100 ]
    then
        continue
    fi
   
    model_name=${sample_path%/*}
    model_name=${model_name##*/}

    out_name=${sample_path##*/files_sample_}

    title=${sample_path##*/}
    dset=${title%_*}
    if [[ $dset == *"_epo"* ]]
    then
        dset=${dset%_*}
    fi
    dset=${dset##*_} 

    echo $model_name

    out_path_model="../../Results/autoencoder/ModalityRecovery/${model_name}/"
    if [ ! -d ${out_path_model} ]
    then
        mkdir ${out_path_model}
    fi

    gt_path_seq="../../Data/Processed/CATH_seq/CATH_seq_${dset}.fasta"
    gt_path_structure="../../Data/Origin/CATH/pdb_all_AtomOnly/"

    out_path="${out_path_model}/${out_name}.pkl"

    if [ -f ${out_path} ]
    then
        continue
    fi

    # ./wrapper_submit_autoencoder_eval.sh \
    # --sample_path ${sample_path} \
    # --gt_path_seq ${gt_path_seq} \
    # --gt_path_structure ${gt_path_structure} \
    # --out_path ${out_path} \
    # --title ${title} \
    # --job_list ${job_list} \
    # --server Grace

    echo ${out_name}

    python autoencoder_eval.py \
    --sample_path ${sample_path} \
    --gt_path_seq ${gt_path_seq} \
    --gt_path_structure ${gt_path_structure} \
    --out_path ${out_path} 

    echo ''

done


######################### for decoding check ##########################################


# model_path="../../Results/autoencoder/decoding_check/"
# out_path_model="../../Results/autoencoder/ModalityRecovery/forDecodingCheck/"
# if [ ! -d ${out_path_model} ]
# then
#     mkdir ${out_path_model}
# fi

# for sample_path in ${model_path}/files*
# do
#     if [ $( ls ${sample_path} | wc -l ) -lt 100 ]
#     then
#         continue
#     fi
# 
#     name=${sample_path##*files_}
#     title="decoderEval_${name}"
#     dset=${name##*_}
# 
#     if [ $dset != 'train' ] && [ $dset != 'val' ] && [ $dset != 'test' ]
#     then
#         dset=${name%_*}
#         dset=${dset##*_}
#     fi
# 
#     gt_path_seq="../../Data/Processed/CATH_seq/CATH_seq_${dset}.fasta"
#     gt_path_structure="../../Data/Origin/CATH/pdb_all_AtomOnly/"
# 
#     out_path="${out_path_model}/${name}.pkl"
# 
#     if [ -f ${out_path} ]
#     then
#         continue
#     fi
#     echo $name
# 
#     ./wrapper_submit_autoencoder_eval.sh \
#     --sample_path ${sample_path} \
#     --gt_path_seq ${gt_path_seq} \
#     --gt_path_structure ${gt_path_structure} \
#     --out_path ${out_path} \
#     --title ${title} \
#     --job_list ${job_list} \
#     --server Grace
# done


