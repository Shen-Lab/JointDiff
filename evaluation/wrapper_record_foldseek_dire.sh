
job_list='Job_list_foldseek.txt'
if [ -f ${job_list} ]
then
    rm ${job_list}
fi

prostt5_model='/scratch/user/shaowen1994/Tools/Foldseek/foldseek/prostt5_out/model/'

for database in '/scratch/user/shaowen1994/Tools/Foldseek/foldseek/database/cath/cath50' '/scratch/user/shaowen1994/Tools/Foldseek/foldseek/database/swiss_prot/swiss-prot'  '/scratch/user/shaowen1994/Tools/Foldseek/foldseek/database/afdb_uniprot50/afdb-uniprot50'
do
    dataname=${database##*/} 

    ###########################################################
    # different modalities
    ###########################################################
    
    for version in 'seq' #'stru'
    do
 
        ###### baselines ######
        
        for model in Nature # protein_generator Chroma
        do
            work_path="../../Results/${model}"
            name=${model}
            if [ ${version} == 'seq' ]
            then
                input="${work_path}/seq_gen_sele.500.fa"
            else
                input="${work_path}/struc_gen_sele.500"
            fi
            output="${work_path}/uniprot_pred_foldseek-${version}-${dataname}.txt"
      
            if [ ${version} == 'seq' ]
            then 
                foldseek easy-search ${input} ${database} ${output} tmpFolder --prostt5-model ${prostt5_model}
            else
                if [ -f ${output} ]
                then                
                    rm ${output}   
                fi                 
                
                for target_pdb in ${input}/*pdb
                do
                    echo \${target_pdb}
                    foldseek easy-search \${target_pdb} ${database} result_tmp.txt tmpFolder
                    cat result_tmp.txt >> ${output}
                    rm result_tmp.txt
                done
            fi
 
        done
        
        # ###### jointDiff ######
        # 
        # for input in $( cat ../Model_Lists/${version}-500_list_jointdiff.sele.txt )
        # do
        # 
        #     if [ ${version} == 'seq' ]
        #     then
        #         work_path=${input%/seq_gen*}
        #     else
        #         work_path=${input%/struc_gen*}
        #     fi

        #     name=${work_path##*/}
        #     output="${work_path}/uniprot_pred_foldseek-${version}-${dataname}.txt"
        # 
        #     foldseek easy-search ${input} ${database} ${output} tmpFolder --prostt5-model ${prostt5_model}
        # 
        # done
        # 
        # ###### LaDiff ######
        # 
        # for input in $( cat ../Model_Lists/${version}-500_list_Ladiff.sele.txt )
        # do
        # 
        #     if [ ${version} == 'seq' ]
        #     then
        #         work_path=${input%/seq_gen*}
        #     else
        #         work_path=${input%/struc_gen*}
        #     fi

        #     work_path=${work_path%/sample*}
        #     name=${work_path##*/}
        #     output="${work_path}/uniprot_pred_foldseek-${version}-${dataname}.txt"
        # 
        #     foldseek easy-search ${input} ${database} ${output} tmpFolder --prostt5-model ${prostt5_model}
        # 
        # done

    done

done
