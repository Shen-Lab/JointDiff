
threshold=10.

###### baselines ######

for model in protein_generator Chroma
do
    work_path="../../Results/${model}"
    name=${model}
    seq_pred="${work_path}/go_pred_foldseek-seq.txt"
    stru_pred="${work_path}/go_pred_foldseek-stru.txt"
    out_path="${work_path}/uniprot_consist.txt"

    if [ ! -f ${seq_pred} ] || [ ! -f ${stru_pred} ]
    then
        continue
    fi

    echo 'Baseline' $name
    python uniprot_compare.py \
    --seq_pred ${seq_pred} \
    --struc_pred ${stru_pred} \
    --out_path ${out_path} \
    --threshold ${threshold}

    echo ''

done

echo '############################################################'

###### jointDiff ######

for input in ../../Results/jointDiff/codesign_diffab_*/seq_gen_sele.500.fa
do

    work_path=${input%/seq_gen*}
    name=${work_path##*/}
    seq_pred="${work_path}/go_pred_foldseek-seq.txt"
    stru_pred="${work_path}/go_pred_foldseek-stru.txt"
    out_path="${work_path}/uniprot_consist.txt"

    if [ ! -f ${seq_pred} ] || [ ! -f ${stru_pred} ]
    then
        continue
    fi

    echo 'JointDiff' $name
    python uniprot_compare.py \
    --seq_pred ${seq_pred} \
    --struc_pred ${stru_pred} \
    --out_path ${out_path} \
    --threshold ${threshold}
    
    echo ''

done

echo '############################################################'


###### LaDiff ######

for input in ../../Results/latentDiff/latentdiff_*/samples/seq_gen_sele.500.fa
do

    work_path=${input%/seq_gen*}
    work_path=${work_path%/sample*}
    name=${work_path##*/}
    seq_pred="${work_path}/go_pred_foldseek-seq.txt"
    stru_pred="${work_path}/go_pred_foldseek-stru.txt"
    out_path="${work_path}/uniprot_consist.txt"

    if [ ! -f ${seq_pred} ] || [ ! -f ${stru_pred} ]
    then
        continue
    fi

    echo 'LaDiff' $name
    python uniprot_compare.py \
    --seq_pred ${seq_pred} \
    --struc_pred ${stru_pred} \
    --out_path ${out_path} \
    --threshold ${threshold}

    echo ''

done
    
