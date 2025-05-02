
job_list='Job_list_foldseek.txt'
if [ -f ${job_list} ]
then
    rm ${job_list}
fi

server='Faster'
#server='Grace'

for database in '/scratch/user/shaowen1994/Tools/Foldseek/foldseek/database/cath/cath50' '/scratch/user/shaowen1994/Tools/Foldseek/foldseek/database/swiss_prot/swiss-prot'  # '/scratch/user/shaowen1994/Tools/Foldseek/foldseek/database/afdb_uniprot50/afdb-uniprot50'
do
    dataname=${database##*/} 

    ###########################################################
    # different modalities
    ###########################################################
    
    for version in 'stru'  # seq
    do
 
        ###### baselines ######
        
        # for model in Nature protein_generator Chroma
        # do
        #     work_path="../../Results/${model}"
        #     name=${model}
        #     if [ ${version} == 'seq' ]
        #     then
        #         input="${work_path}/seq_gen_sele.500.fa"
        #     else
        #         input="${work_path}/struc_gen_sele.500"
        #     fi
        #     output="${work_path}/uniprot_pred_foldseek-${version}-${dataname}.txt"
	#     #title="${version}-${dataname}_${name}"
        #     #output="${work_path}/uniprot_pred_foldseek-${version}-${dataname}_faster.txt"
	#     title="${version}-${dataname}_${name}_faster"
        # 
        #     if [ -f ${output} ]
        #     then
        #         continue
        #     fi

        #     ./wrapper_submit_foldseek.sh \
        #     --version ${version} \
        #     --input ${input} \
        #     --output ${output} \
        #     --database ${database} \
        #     --title ${title} \
        #     --job_list ${job_list} \
        #     --server ${server} \
        # 
        # done
        
        ###### jointDiff ######
        
        # for input in $( cat ../Model_Lists/${version}-500_list_jointdiff.sele.txt )
        for input in ../../Results/jointDiff_updated/codesign_diffab_complete_gen_share-true_dim-128-64-4_step100_lr1.e-4_wd0._posiscale50.0_sc_center_2024_10_20__23_57_14_loss-1-l1-1-1/struc_gen_last_sele.500
        do
        
            if [ ${version} == 'seq' ]
            then
                work_path=${input%/seq_gen*}
            else
                work_path=${input%/struc_gen*}
            fi

            name=${work_path##*/}
            output="${work_path}/uniprot_pred_foldseek-${version}-${dataname}.txt"
	    title="${version}-${dataname}_${name}"
            #output="${work_path}/uniprot_pred_foldseek-${version}-${dataname}_faster.txt"
	    #title="${version}-${dataname}_${name}_faster"
        
            if [ -f ${output} ]
            then
                continue
            fi

            ./wrapper_submit_foldseek.sh \
            --version ${version} \
            --input ${input} \
            --output ${output} \
            --database ${database} \
            --title ${title} \
            --job_list ${job_list} \
            --server ${server} \
        
        done
        
        ###### LaDiff ######
        
        # for input in $( cat ../Model_Lists/${version}-500_list_Ladiff.sele.txt )
        for input in ../../Results/latentDiff/latentdiff_with-ESM-IF_joint-mlp-4-512_pad-zero_dim16_vae-0.001_NoEnd_unet-2-32_withMask_gt_len-mha_Faster_100steps/struc_gen_sele.500/
        do
        
            if [ ${version} == 'seq' ]
            then
                work_path=${input%/seq_gen*}
            else
                work_path=${input%/struc_gen*}
            fi

            #work_path=${work_path%/sample*}
            name=${work_path##*/}
            output="${work_path}/uniprot_pred_foldseek-${version}-${dataname}.txt"
	    #title="${version}-${dataname}_${name}"
            #output="${work_path}/uniprot_pred_foldseek-${version}-${dataname}_faster.txt"
	    title="${version}-${dataname}_${name}_faster"

            if [ -f ${output} ]
            then
                continue
            fi
        
            ./wrapper_submit_foldseek.sh \
            --version ${version} \
            --input ${input} \
            --output ${output} \
            --database ${database} \
            --title ${title} \
            --job_list ${job_list} \
            --server ${server} \
        
        done

    done

done
