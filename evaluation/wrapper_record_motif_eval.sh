server='Grace'
job_list='Job_list_motif_eval.txt'
if [ -f ${job_list} ]
then
    rm ${job_list}
fi

#for pred_path in ../../Results/jointDiff_motif/*/struc_pred_mpnn
#for pred_path in ../../Results/Baseline_motif/*/struc_pred_mpnn
for pred_path in ../../Results/jointDiff_motif_case_grace/*/struc_pred_mpnn
do
    num=$( ls ${pred_path} | wc -l )
    if [ $num -lt 100 ]
    then
        continue
    fi

    result_path=${pred_path%/struc_pred*}
    name=${result_path##*/}

    title="designability_motif_${name}"

    ref_path="${result_path}/samples"
    out_path="${result_path}/designability_esmfold_TMscore_dict.pkl"

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


# for pred_path in ../../Results/jointDiff_motif/*/struc_pred_mpnn
# do
#     num=$( ls ${pred_path} | wc -l )
#     if [ $num -lt 100 ]
#     then
#         continue
#     fi
# 
#     result_path=${pred_path%/struc_pred*}
#     name=${result_path##*/}
# 
#     title="designability_motif_${name}"
# 
#     ref_path="${result_path}/samples"
#     out_path="${result_path}/designability_esmfold_TMscore_dict.pkl"
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
