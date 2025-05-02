##############################################################
# for AF2 
##############################################################

### original diffusion
#raw_struc_path="../../Results/originDiff/Uncon_struc/"
#raw_struc_path="../../Results/originDiff/Uncon_SingleModal_struc/"

#for name in $( cat ../Model_Lists/model_stoping_Eval.txt )
#for name in $( cat ../Model_Lists/model_stoping_sampleSele.txt )
#do
#    name=${name%_*}
#    echo $name
#
#    out_path="../../Results/originDiff/sample_sele_forAF2/${name}/"
#    if [ ! -d ${out_path} ]
#    then
#        mkdir ${out_path}
#        python sample_selection.py --sample_path ${raw_struc_path}/${name}/ --out_path ${out_path} --seq_flag 1 --AF2_flag 0
#    fi
#
#done

# ### guided diffusion
# raw_struc_path="../../Results/guidedDiff/Uncon_struc/"
# 
# for name in $( cat ../Model_Lists/model_list_feat-cal_guided.txt )
# do
#     echo $name
# 
#     out_path="../../Results/guidedDiff/sample_sele_forAF2/${name}/"
#     if [ ! -d ${out_path} ]
#     then
#         mkdir ${out_path}
#         python sample_selection.py --sample_path ${raw_struc_path}/${name}/ --out_path ${out_path} --seq_flag 1 --AF2_flag 1
#     fi
# 
# done

# ### Chroma

#sample_path="../../Results/Chroma/sampled_pdb/"
#out_path="../../Results/Chroma/sample_sele_forAF2/"
#python sample_selection.py --sample_path ${sample_path}/ --out_path ${out_path} --seq_flag 1 --AF2_flag 1

# ### Protein Generator
# 
# sample_path="../../Results/protein_generator/sampled_pdb/"
# out_path="../../Results/protein_generator/sample_sele_forAF2/"
# python sample_selection.py --sample_path ${sample_path}/ --out_path ${out_path} --seq_flag 1 --AF2_flag 0


##############################################################
# for Functional prediction
##############################################################

sele_num=500
job_list='Job_list_sample_sele.txt'
if [ -f ${job_list} ]
then
    rm ${job_list}
fi

###### baselines ######

###### jointDiff ######

for seq_path in ../../Results/jointDiff/codesign_diffab_*/seq_gen.fa
do

    work_path=${seq_path%/seq_gen*}
    name=${work_path##*/}
    struc_path="${work_path}/samples/"
    seq_out="${work_path}/seq_gen_sele.500.fa"
    struc_out="${work_path}/struc_gen_sele.500/"
    
    if [ ! -d ${struc_out} ]
    then
        mkdir ${struc_out}
    fi
    
    #python sample_sele.py \
    ./wrapper_submit_sample_sele.sh \
    --seq_path ${seq_path} \
    --struc_path ${struc_path} \
    --seq_out ${seq_out} \
    --struc_out ${struc_out} \
    --sele_num ${sele_num} \
    --token '_0_' \
    --job_list ${job_list} \
    --server Grace \
    --title "${name}"  

done

###### LaDiff ######

for seq_path in ../../Results/latentDiff/latentdiff_*/samples/seq_gen.fa
do

    work_path=${seq_path%/seq_gen*}
    name=${work_path%/samples*}
    name=${name##*/}
    struc_path="${work_path}/files_sample_gen/"
    seq_out="${work_path}/seq_gen_sele.500.fa"
    struc_out="${work_path}/struc_gen_sele.500/"

    if [ ! -d ${struc_out} ]
    then
        mkdir ${struc_out}
    fi

    #python sample_sele.py \
    ./wrapper_submit_sample_sele.sh \
    --seq_path ${seq_path} \
    --struc_path ${struc_path} \
    --seq_out ${seq_out} \
    --struc_out ${struc_out} \
    --sele_num ${sele_num} \
    --token '_step0_' \
    --job_list ${job_list} \
    --server Grace \
    --title ${name} 

done
