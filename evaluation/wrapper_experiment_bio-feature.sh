in_path=$1

if [ -c ${in_path} ]
then
    echo 'Input is empty!'
    exit
fi

if [ ${in_path: -1} == '/' ]
then
    in_path=${in_path%/*}
fi
job_token=${in_path##*/}

####################### paths #############################

seq_in="${in_path}/ProteinMPNN_design/seqs/"
seq_out="${in_path}/ProteinMPNN_design/seqs_forAF2-multi/"
list_path="${in_path}/ProteinMPNN_design/lists_forAF2-multi/"
list_all="${in_path}/ProteinMPNN_design/list_of_path-list.txt"
out_path="${in_path}/design_AF2_pred/"

################# sequence formatting #####################

python proteinmpnn_seq_for_AF2-multimonomer.py \
--in_path ${seq_in} \
--out_path ${seq_out} \
--list_path ${list_path} \
--summary ${list_all} \
--sele_num 5

################# AF_jobs #####################

for line in $( cat ${list_all} )
do
    fasta_list_1=${line%%+*}

    if  [[ $line == *"+"* ]]
    then 
        fasta_list_2=${line##*+}
    else
        fasta_list_2='none'
    fi

    ./wrapper_submit_AF2_pred_multimonomer_2GPU.sh \
    --fasta_list_1 ${fasta_list_1} \
    --fasta_list_2 ${fasta_list_2} \
    --outpath ${out_path} \
    --job_token ${job_token}

done
