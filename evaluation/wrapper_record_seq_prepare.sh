### original diffusion 

# STRUC_PATH="../../Results/originDiff/Uncon_struc/"
# 
# for name in $( cat ../Model_Lists/model_stoping_Eval.txt )
# do
#     name=${name%_*}
#     struc_path="${STRUC_PATH}/${name}/"
#     out_path="../../Results/originDiff/Uncon_seq/${name}.fasta"
# 
#     if [ ! -f ${out_path} ]
#     then
#         python fasta_prepare.py --feature_path ${struc_path}/ --out_path ${out_path} --with_pdb 1
#     fi
# done

STRUC_PATH="../../Results/originDiff/Uncon_struc/"

for name in $( cat ../Model_Lists/model_stoping_Eval.txt )
do
    name=${name%_*}
    struc_path="${STRUC_PATH}/${name}/"
    out_path="../../Results/originDiff/Uncon_seq/${name}.fasta"

    if [ ! -f ${out_path} ]
    then
        python fasta_prepare.py --feature_path ${struc_path}/ --out_path ${out_path} --with_pdb 1
    fi
done

### Chroma

# feat_path="../../Results/Chroma/Features/"
# out_path="../../Results/Chroma/Uncon_seq/Chroma.fasta"
# 
# python fasta_prepare.py --feature_path ${feat_path}/ --out_path ${out_path} --with_pdb 0


### original diffusion 

# STRUC_PATH="../../Results/originDiff/Uncon_traj/struc_sele/"
# 
# for struc_path in ${STRUC_PATH}/*
# do
#     name=${struc_path##*/}
#     out_path="../../Results/originDiff/Uncon_traj/seq_sele/${name}.fasta"
#     echo $name
# 
#     if [ ! -f ${out_path} ]
#     then
#         python fasta_prepare.py --feature_path ${struc_path}/ --out_path ${out_path} --with_pdb 1
#     fi
# done
