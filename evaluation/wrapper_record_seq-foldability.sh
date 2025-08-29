################ joint diffusion ###########################

echo "Joint Diffusion"
echo "#######################################################"

for work_path in ../../Results/jointDiff/codesign_diffab_*
do
    seq_path="${work_path}/esmfold-mpnn-pred/seqs/"
    if [ ! -d ${seq_path} ]
    then
        continue
    fi
  
    gt_path="${work_path}/seq_gen.fa"
    if [ ! -f ${gt_path} ]
    then
        continue
    fi

    out_path="${work_path}/seq-foldability.pkl"
    if [ -f ${out_path} ]
    then
        continue
    fi

    echo ${work_path##*/}

    python foldability_cal.py \
    --seq_path ${seq_path} \
    --gt_path  ${gt_path} \
    --out_path ${out_path} \
    --alignment 1 \
    --mpnn_format 1

    echo '' 

done 

################ laten diffusion ###########################
 
echo "Latent Diffusion"
echo "#######################################################"

for work_path in ../../Results/latentDiff/latentdiff_with-ESM-IF_joint-mlp-4-512_pad-zero_*
do
    seq_path="${work_path}/esmfold-mpnn-pred/seqs/"
    if [ ! -d ${seq_path} ]
    then
        continue
    fi
  
    gt_path="${work_path}/samples/seq_gen.fa"
    if [ ! -f ${gt_path} ]
    then
        continue
    fi

    out_path="${work_path}/seq-foldability.pkl"
    if [ -f ${out_path} ]
    then
        continue
    fi

    echo ${work_path##*/}

    python foldability_cal.py \
    --seq_path ${seq_path} \
    --gt_path  ${gt_path} \
    --out_path ${out_path} \
    --alignment 1 \
    --mpnn_format 1

    echo '' 

done 

