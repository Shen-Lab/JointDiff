out_path='../../Results/latentDiff/samples/latentdiff_diagnosis/'

for model in $( cat ../Model_Lists/checkpoints_diagnosis.txt )
do
    name=${model%/*}
    name=${name##*co-MPNN_}
    save_path="${out_path}/${name}/"
    gt_seq_path="${save_path}/file_sample_gen/"

    echo $name

    ###### features ######

    # echo '**************************************************'
    # echo 'Features...'

    # result_path="${save_path}/Features/"
    # if [ ! -d ${result_path} ]
    # then
    #     mkdir ${result_path}
    # fi

    # for pdb in ${gt_seq_path}/*.pdb
    # do
    #     name=${pdb##*/}
    #     name=${name%.*}
    #     
    #     python pdb_feature_cal.py \
    #     --in_path ${pdb} \
    #     --out_path "${result_path}/${name}.pkl"

    # done

    ###### foldability (proteinMPNN) ######

    # echo '**************************************************'
    # echo 'With ProteinMPNN...'

    # seq_path="${save_path}/ProteinMPNN_design/seqs/"
    # result_path="${save_path}/foldability_proteinMPNN.pkl"

    # python foldability_cal.py \
    # --seq_path ${seq_path} \
    # --gt_path ${gt_seq_path} \
    # --out_path ${result_path} \
    # --alignment 1 \
    # --mpnn_format 1

    ###### foldability (esm-if) ######

    # echo '**************************************************'
    # echo 'With ESM-IF...'

    # seq_path="${save_path}/esm-if_pred/"
    # result_path="${save_path}/foldability_esm-IF.pkl"

    # python foldability_cal.py \
    # --seq_path ${seq_path} \
    # --gt_path ${gt_seq_path} \
    # --out_path ${result_path} \
    # --alignment 1 \
    # --mpnn_format 0
    # 
    # echo '###################################################'
    # echo ''

    ###### fitness ######

    cat_seq_path="${save_path}/seq_gen.fa"
    cat ${gt_seq_path}/*fa > ${cat_seq_path}
    fitness_path="${save_path}/fitness_dict.pkl"

    python sample_energy_fitness_cal.py \
    --seq_path ${cat_seq_path} \
    --out_path ${fitness_path} 

done

################## self conditioning ############################

for model in ../../Logs/logs_latentDiff/latentdiff_joint-mlp-4-512_co-MPNN_withMask_gt_grace/Epoch100.pt
do
    name=${model%/*}
    name=${name##*co-MPNN_}
    name="${name}_selfcondi"

    save_path="${out_path}/${name}/"
    gt_seq_path="${save_path}/file_sample_gen/"

    echo $name

    ###### features ######

    # echo '**************************************************'
    # echo 'Features...'

    # result_path="${save_path}/Features/"
    # if [ ! -d ${result_path} ]
    # then
    #     mkdir ${result_path}
    # fi

    # for pdb in ${gt_seq_path}/*.pdb
    # do
    #     name=${pdb##*/}
    #     name=${name%.*}
    #     
    #     python pdb_feature_cal.py \
    #     --in_path ${pdb} \
    #     --out_path "${result_path}/${name}.pkl"

    # done

    ###### foldability (proteinMPNN) ######

    echo '**************************************************'
    echo 'With ProteinMPNN...'

    seq_path="${save_path}/ProteinMPNN_design/seqs/"
    result_path="${save_path}/foldability_proteinMPNN.pkl"

    python foldability_cal.py \
    --seq_path ${seq_path} \
    --gt_path ${gt_seq_path} \
    --out_path ${result_path} \
    --alignment 1 \
    --mpnn_format 1

    ###### foldability (esm-if) ######

    # echo '**************************************************'
    # echo 'With ESM-IF...'

    # seq_path="${save_path}/esm-if_pred/"
    # result_path="${save_path}/foldability_esm-IF.pkl"

    # python foldability_cal.py \
    # --seq_path ${seq_path} \
    # --gt_path ${gt_seq_path} \
    # --out_path ${result_path} \
    # --alignment 1 \
    # --mpnn_format 0
    
    echo '###################################################'
    echo ''

done
