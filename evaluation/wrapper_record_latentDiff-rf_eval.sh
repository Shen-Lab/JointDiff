job_list="Job_list_latentDiff_eval.txt"

if [ -f ${job_list} ]
then
    rm ${job_list}
fi

######################### for decoding check ##########################################

version_list="../Model_Lists/version_list_reverse_decoding.txt"
result_path="../../Results/latentDiff/forward_samples/latentdiff_joint-mlp-4-512_co-MPNN/"

#for version in seq-struc_reverse_direct_step100_Epoch250_withMask # seq-struc_direct_step100 seq-struc_reverse_direct_step100_Epoch100_withMask 
#do

for version in $( cat ${version_list} )
do
    version="seq-struc_${version}"

    for file_path in ${result_path}/${version}/file_sample_gen*_prepared
    do
        kind=${file_path%_prepared*}
        kind=${kind##*file_sample_gen}

        out_path="${result_path}/${version}/Recovery${kind}/"
        if [ ! -d ${out_path} ]
        then
            mkdir ${out_path}
        fi

        echo ${version}

        for sample_path in ${file_path}/*
        do
            if [ $( ls ${sample_path} | wc -l ) -lt 20 ]
            then
                continue
            fi
        
            name=${sample_path##*/}
            title="decoderEval_${version}${kind}_${name}"
            dset='test'
        
            if [ $dset != 'train' ] && [ $dset != 'val' ] && [ $dset != 'test' ]
            then
                dset=${name%_*}
                dset=${dset##*_}
            fi
        
            gt_path_seq="../../Data/Processed/CATH_seq/CATH_seq_${dset}.fasta"
            gt_path_structure="../../Data/Origin/CATH/pdb_all_AtomOnly/"
        
            save_path="${out_path}/${name}.pkl"
            if [ -f ${save_path} ]
            then
                continue
            fi
            echo $name
        
            ./wrapper_submit_autoencoder_eval.sh \
            --sample_path ${sample_path} \
            --gt_path_seq ${gt_path_seq} \
            --gt_path_structure ${gt_path_structure} \
            --out_path ${save_path} \
            --title ${title} \
            --job_list ${job_list} \
            --server Grace
        done
    done
done
