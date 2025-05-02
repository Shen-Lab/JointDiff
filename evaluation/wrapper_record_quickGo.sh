
job_list='Job_list_quickGo.txt'
if [ -f ${job_list} ]
then
    rm ${job_list}
fi

threshold=30.
#server='Faster'
server='Grace'
key_token='uniprot_pred'
#key_token='go_pred'

###### baselines ######

# for model in Nature protein_generator Chroma
# do
#     work_path="../../Results/${model}"
#     name=${work_path##*/}
# 
#     for input in ${work_path}/${key_token}_*.txt
#     do
#         title=${input##*${key_token}_}
#         title=${title%.txt*}
# 
#         output="${work_path}/go_pred_${title}.pkl"
#         title="${name}_${title}"
# 
#         # if [ -f ${output} ]
# 	# then
# 	#     continue
# 	# fi
# 
#         ./wrapper_submit_quickGo.sh \
#         --in_path ${input} \
#         --out_path ${output} \
#         --threshold ${threshold} \
#         --title ${title} \
#         --job_list ${job_list} \
#         --server ${server}
#     done
# done

###### jointDiff ######

#for input in ../../Results/jointDiff/codesign_diffab_*/${key_token}_*.txt
for input in ../../Results/jointDiff_updated/codesign_diffab_complete_gen_share*/uniprot*.txt
do

    work_path=${input%/${key_token}*}
    name=${work_path##*/}

    title=${input##*${key_token}_}
    title=${title%.txt*}

    output="${work_path}/go_pred_${title}.pkl"
    title="${name}_${title}"

    # if [ -f ${output} ]
    # then
    #     continue
    # fi

    ./wrapper_submit_quickGo.sh \
    --in_path ${input} \
    --out_path ${output} \
    --threshold ${threshold} \
    --title ${title} \
    --job_list ${job_list} \
    --server ${server}

done

###### LaDiff ######

#for input in ../../Results/latentDiff/latentdiff_*/${key_token}_*.txt
for input in ../../Results/latentDiff/latentdiff_with-ESM-IF_joint-mlp-4-512_pad-zero_dim16_vae-0.001_NoEnd_unet-2-32_withMask_gt_len-mha_Faster_100steps/uniprot_*.txt
do

    work_path=${input%/${key_token}*}
    name=${work_path##*/}

    title=${input##*${key_token}_}
    title=${title%.txt*}

    output="${work_path}/go_pred_${title}.pkl"
    title="${name}_${title}"

    # if [ -f ${output} ]
    # then
    #     continue
    # fi

    ./wrapper_submit_quickGo.sh \
    --in_path ${input} \
    --out_path ${output} \
    --threshold ${threshold} \
    --title ${title} \
    --job_list ${job_list} \
    --server ${server}

done

