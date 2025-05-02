
############################################################################
# original diffusion
############################################################################

in_path="../../Results/originDiff/Uncon_seq/"
out_path="../../Results/originDiff/Fitness/"
job_list="Job_list_fitness_origin_diffusion.txt"

if [ -f ${job_list} ]
then
    rm ${job_list}
fi

for seq_path in ${in_path}/*.fasta
do
    title=${seq_path##*/}
    title=${title%.fa*}
    #title="fitness_${title}"

    result_path="${out_path}/${title}.pkl"

    ./wrapper_submit_fitness_cal.sh \
    --seq_path ${seq_path} \
    --result_path ${result_path} \
    --title ${title} \
    --batch_size 8 \
    --job_list ${job_list}

done


############################################################################
# ours (with guidance)
############################################################################

# in_path="../../Results/diffab_pre/Uncon_seq/"
# out_path="../../Results/diffab_pre/Fitness/"
# job_list="Job_list_fitness_guided_diffusion.txt"
# 
# if [ -f ${job_list} ]
# then
#     rm ${job_list}
# fi
#  
# for seq_path in ${in_path}/*.fasta 
# do
#     title=${seq_path##*/}
#     title=${title%.fa*}
#     #title="fitness_${title}"
# 
#     result_path="${out_path}/${title}.pkl"
# 
#     ./wrapper_submit_fitness_cal.sh \
#     --seq_path ${seq_path} \
#     --result_path ${result_path} \
#     --title ${title} \
#     --batch_size 8 \
#     --job_list ${job_list}
# 
# done

