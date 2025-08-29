###############################################################################
# Sequence centered 
###############################################################################

echo "Sequence-centered joint-consistency:"

server='Grace'
job_list="Job_list_consistency_seq.txt"
if [ -f ${job_list} ]
then
    rm ${job_list}
fi

############################## Baselines ######################################
 
# RESULT_PATH="../../Results/"
# 
# for model in Chroma  Nature  protein_generator
# do
#     result_path="${RESULT_PATH}/Baselines/${model}/"
#     name=${model}
# 
#     if [ ${model} == 'Nature' ]
#     then
#         struc_path='../../Data/Origin/CATH/pdbs_eval/'
#     else
#         struc_path="${result_path}/samples"
#     fi
#     seq_path="${result_path}/mpnn-pred/seqs/"
#     gt_path="${result_path}/seq_gen.fa"
#     out_path="${result_path}/consistency-seq_dict.pkl"
#     title="consist-seq_${name}"
# 
#     if [ -f ${out_path} ]
#     then
#         continue
#     fi
# 
#     ./wrapper_submit_consistency-seq.sh \
#     --struc_path ${struc_path} \
#     --seq_path ${seq_path} \
#     --gt_path  ${gt_path}  \
#     --out_path ${out_path} \
#     --alignment 1 \
#     --title ${title} \
#     --server ${server} \
#     --job_list ${job_list}
# 
#     # python consistency-seq_cal.py \
#     # --seq_path ${seq_path} \
#     # --gt_path  ${gt_path}  \
#     # --out_path ${out_path} \
#     # --alignment 1
# done


############################## JointDiff (general) ######################################

result_dir="/scratch/user/shaowen1994/JointDiff_development/Results/jointDiff_development/"

for token in '_last' '_last2'  #'_last'
do

    for work_path in ${result_dir}/jointdiff*  
    do
        seq_path="${work_path}/mpnn_pred${token}/seqs/"
        if [ ! -d ${seq_path} ]
	then
	    continue
	fi

        num=$( ls ${seq_path} | wc -l )
        if [ $num -lt 100 ]
        then
            continue
        fi 
    
        name=${work_path##*/}
        title="consist-seq_${name}${token}"
    
        struc_path="${work_path}/samples${token}"
        gt_path="${work_path}/seq_gen${token}.fa"
        out_path="${work_path}/consistency-seq${token}_dict.pkl"
    
        if [ -f ${out_path} ]
        then
            continue
        fi
    
        ./wrapper_submit_consistency-seq.sh \
        --struc_path ${struc_path} \
        --seq_path ${seq_path} \
        --gt_path  ${gt_path}  \
        --out_path ${out_path} \
        --alignment 1 \
        --title ${title} \
        --server ${server} \
        --job_list ${job_list} 
    
    done
done

############################### motif-scaffolding #############################

result_dir="/scratch/user/shaowen1994/JointDiff_development/Results/jointDiff_development/"

for token in '_last' # '_last'
do

    for work_path in ${result_dir}/jointdiff*  
    do
        seq_path="${work_path}/ms_mpnn_pred${token}/seqs/"
        if [ ! -d ${seq_path} ]
	then
	    continue
	fi

        num=$( ls ${seq_path} | wc -l )
        if [ $num -lt 20 ]
        then
            continue
        fi 
    
        name=${work_path##*/}
        title="ms_consist-seq_${name}${token}"
    
        struc_path="${work_path}/motifscaffolding${token}"
        gt_path="${work_path}/ms_seq_gen${token}.fa"
        out_path="${work_path}/ms_consistency-seq${token}_dict.pkl"
    
        if [ -f ${out_path} ]
        then
            continue
        fi
    
        ./wrapper_submit_consistency-seq.sh \
        --struc_path ${struc_path} \
        --seq_path ${seq_path} \
        --gt_path  ${gt_path}  \
        --out_path ${out_path} \
        --alignment 1 \
        --title ${title} \
        --server ${server} \
        --job_list ${job_list} 
    
    done
done


###############################################################################
# Structure centered 
###############################################################################

echo "Structure-centered joint-consistency:"

server='Grace'
job_list="Job_list_consistency_struc.txt"
if [ -f ${job_list} ]
then
    rm ${job_list}
fi

############################ Baselines ########################################

# for model in Chroma  Nature  protein_generator 
# do
#    echo ${model}
# 
#    result_path="${RESULT_PATH}/Baselines/${model}/"
# 
#    if [ ${model} == 'Nature' ]
#    then
#        ref_path='../../Data/Origin/CATH/pdbs_eval/'
#    else
#        ref_path="${result_path}/samples"
#    fi
#    ref_path="${result_path}/samples/"
#    pred_path="${result_path}/struc_pred_esmfold/"
#    out_path="${result_path}/consistency-struc_dict.pkl"
# 
#    if [ -f ${out_path} ]
#    then
#        continue
#    fi
# 
#    title="consistency-struc_${model}"
# 
#    ./wrapper_submit_consistency-struc.sh \
#    --ref_path ${ref_path} \
#    --pred_path ${pred_path} \
#    --out_path ${out_path} \
#    --job_list ${job_list} \
#    --title ${title} \
#    --server ${server} 
# 
#    echo '************************************************'
# 
# done

############################## JointDiff (general) ######################################

result_dir="/scratch/user/shaowen1994/JointDiff_development/Results/jointDiff_development/"

for token in '_last' '_last2' #'_last'
do

    for work_path in ${result_dir}/jointdiff*  
    do

        pred_path="${work_path}/struc_pred_esmfold${token}"

        if [ ! -d ${pred_path} ]
	then
	    continue
	fi

        num=$( ls ${pred_path} | wc -l )
        if [ $num -lt 100 ]
        then
            continue
        fi

        name=${work_path##*/}
        title="consist-struc_${name}${token}"

        ref_path="${work_path}/samples${token}"
        out_path="${work_path}/consistency-struc${token}_dict.pkl"
   
        if [ -f ${out_path} ]
        then
            continue
        fi

        ./wrapper_submit_consistency-struc.sh \
        --ref_path ${ref_path} \
        --pred_path ${pred_path} \
        --out_path ${out_path} \
        --job_list ${job_list} \
        --title ${title} \
        --server ${server} \
        --job_list ${job_list} 

    done
done

###### motif-scaffolding ######

result_dir="/scratch/user/shaowen1994/JointDiff_development/Results/jointDiff_development/"

for token in '_last' #'_last'
do

    for work_path in ${result_dir}/jointdiff*  
    do

        pred_path="${work_path}/ms_struc_pred_esmfold${token}"

        if [ ! -d ${pred_path} ]
	then
	    continue
	fi

        num=$( ls ${pred_path} | wc -l )
        if [ $num -lt 20 ]
        then
            continue
        fi

        name=${work_path##*/}
        title="ms_consist-struc_${name}${token}"

        ref_path="${work_path}/motifscaffolding${token}"
        out_path="${work_path}/ms_consistency-struc${token}_dict.pkl"
   
        #if [ -f ${out_path} ]
        #then
        #    continue
        #fi

        ./wrapper_submit_consistency-struc.sh \
        --ref_path ${ref_path} \
        --pred_path ${pred_path} \
        --out_path ${out_path} \
        --job_list ${job_list} \
        --title ${title} \
        --server ${server} \
        --job_list ${job_list} 

    done
done


