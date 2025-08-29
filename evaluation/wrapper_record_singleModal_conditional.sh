job_list="Job_list_singleModal_conditional.txt"

if [ -f ${job_list} ]
then
    rm ${job_list}
fi

############### fixed-backbone sequence design ######################

# for seq_path in ../../Results/originDiff/SingleModal_seq/*
# do
#     name=${seq_path##*/}
#     echo $name
# 
#     for alignment in 0 1
#     do
#         if [ ${alignment} == 1 ]
#         then
#             out_path="../../Results/originDiff/fixbackbone_SeqIden/${name}.pkl"
#         else
#             out_path="../../Results/originDiff/fixbackbone_SeqIden/${name}_align-free.pkl"
#         fi
# 
#         if [ -f ${out_path} ]
#         then
#             continue
#         fi
# 
#         ./wrapper_submit_SequenceIden_FixBackbone.sh \
#         --seq_path ${seq_path} \
#         --out_path ${out_path} \
#         --alignment ${alignment} \
#         --job_list ${job_list}
#     done
# done

############### fixed-sequence structure design #####################

for struc_path in ../../Results/originDiff/SingleModal_struc/*
do
    name=${struc_path##*/}
    echo $name
    out_path="../../Results/originDiff/fixseq_TMscore/${name}.pkl"

    if [ -f ${out_path} ]
    then
        continue
    fi

    ./wrapper_submit_TMscore_FixSequence.sh \
    --struc_path ${struc_path} \
    --out_path ${out_path} \
    --job_list ${job_list} \
    --server Grace 
done

