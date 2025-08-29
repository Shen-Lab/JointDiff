###############################################################################
# foldability 
###############################################################################

echo "Foldability:"

job_list="Job_list_foldability.txt"
server='Grace'

if [ -f ${job_list} ]
then
    rm ${job_list}
fi

# ############################## Baselines ######################################
# 
# for model in Chroma  protein_generator Nature
# do
#    echo ${model}
#    seq_path="../../Results/Baselines/${model}/mpnn-pred/seqs/"
# 
#    num=$( ls ${seq_path} | wc -l )
#    if [ $num -lt 100 ]
#    then
#        continue
#    fi
# 
#    result_path=${seq_path%/mpnn-pred*}
#    name=${model}
#    title="foldability_${name}"
# 
#    struc_path="${result_path}/struc_pred_esmfold"
#    gt_path="${result_path}/seq_gen.fa"
#    out_path="${result_path}/seq_foldability.pkl"
# 
#    if [ -f ${out_path} ]
#    then
#        continue
#    fi
# 
#    ./wrapper_submit_consistency-seq.sh \
#    --struc_path ${struc_path} \
#    --seq_path ${seq_path} \
#    --gt_path  ${gt_path}  \
#    --out_path ${out_path} \
#    --alignment 1 \
#    --title ${title} \
#    --server ${server} \
#    --job_list ${job_list}
# 
#    
#    echo '************************************************'
# 
# done

############################## JointDiff (general) #####################################

result_dir="/scratch/user/shaowen1994/JointDiff_development/Results/jointDiff_development/"

for token in '_last' '_last2' #'_last'
do

    for work_path in ${result_dir}/jointdiff*
    do
    
	seq_path="${work_path}/mpnn_pred_esmfold${token}/seqs"
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
        title="foldability_${name}${token}"

        struc_path="${work_path}/struc_pred_esmfold${token}"
        gt_path="${work_path}/seq_gen${token}.fa"
        out_path="${work_path}/foldability${token}.pkl"

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


############################ motif-scaffolding ################################

result_dir="/scratch/user/shaowen1994/JointDiff_development/Results/jointDiff_development/"

for token in '_last'
do

    for work_path in ${result_dir}/jointdiff*
    do
    
	seq_path="${work_path}/ms_mpnn_pred_esmfold${token}/seqs"
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
        title="ms_foldability_${name}${token}"

        struc_path="${work_path}/ms_struc_pred_esmfold${token}"
        gt_path="${work_path}/ms_seq_gen${token}.fa"
        out_path="${work_path}/ms_foldability${token}.pkl"

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
# desinability
###############################################################################

echo "Designability:"

job_list="Job_list_designability.txt"
server='Grace'

if [ -f ${job_list} ]
then
    rm ${job_list}
fi

############################## Baselines ######################################

# for model in Chroma  protein_generator foldingdiff RFdiffusion Nature
# do
#     echo ${model}
#     pred_path="../../Results/Baselines/${model}/struc_pred_mpnn-esmfold"
# 
#     num=$( ls ${pred_path} | wc -l )
#     if [ $num -lt 100 ]
#     then
#         continue
#     fi
# 
#     result_path=${pred_path%/struc_pred*}
#     name=${model}
#     title="designability_${name}"
# 
#     if [ ${model} == 'Nature' ]
#     then
#         ref_path='../../Data/Origin/CATH/pdbs_eval/'
#     else
#         ref_path="${result_path}/samples"
#     fi
#     out_path="${result_path}/designability_dict.pkl"
# 
#     if [ -f ${out_path} ]
#     then
#         continue
#     fi
# 
#     ./wrapper_submit_consistency-struc.sh \
#     --ref_path ${ref_path} \
#     --pred_path ${pred_path} \
#     --out_path ${out_path} \
#     --job_list ${job_list} \
#     --title ${title} \
#     --server ${server} \
#     --job_list ${job_list}
# 
# done
# 
############################## JointDiff (general) ######################################

result_dir="/scratch/user/shaowen1994/JointDiff_development/Results/jointDiff_development/"

for token in '_last' '_last2' #''_last'
do

    for work_path in ${result_dir}/jointdiff*
    do
   
        pred_path="${work_path}/struc_pred_mpnn-esmfold${token}"

        if [ ! -d $pred_path ]
        then
            continue
        fi

        num=$( ls ${pred_path} | wc -l )
        if [ $num -lt 100 ]
        then
            continue
        fi

        name=${work_path##*/}
        title="designability_${name}${token}"

        ref_path="${work_path}/samples${token}"
        out_path="${work_path}/designability${token}.pkl"

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


########################## motif-scaffolding #######################################


result_dir="/scratch/user/shaowen1994/JointDiff_development/Results/jointDiff_development/"

for token in '_last'
do

    for work_path in ${result_dir}/jointdiff*
    do
   
        pred_path="${work_path}/ms_struc_pred_mpnn-esmfold${token}"

        if [ ! -d $pred_path ]
        then
            continue
        fi

        num=$( ls ${pred_path} | wc -l )
        if [ $num -lt 100 ]
        then
            continue
        fi

        name=${work_path##*/}
        title="ms_designability_${name}${token}"

        ref_path="${work_path}/motifscaffolding${token}"
        out_path="${work_path}/ms_designability${token}.pkl"

        # if [ -f ${out_path} ]
        # then
        #     continue
        # fi

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


