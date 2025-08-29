###############################################################################
# Foldability
###############################################################################

RESULT_PATH='../../Results/'
job_list='Job_list_foldability.txt'
if [ -f ${job_list} ]
then
    rm ${job_list}
fi

############################## Baselines ######################################

# for model in Chroma protein_generator
# do
#     result_path="${RESULT_PATH}/${model}/"
#     seq_path="${result_path}/seq_pred_mpnn/seqs/"
#     gt_path="${result_path}/seq_gen.fa"
#     out_path="${result_path}/foldability_dict.pkl"
# 
#     python foldability_cal.py \
#     --seq_path ${seq_path} \
#     --gt_path  ${gt_path}  \
#     --out_path ${out_path} \
#     --alignment 1
# done


# ############################## Our models #####################################
# 
# sample_sele_path='../../Results/originDiff/sample_sele_forAF2/'
# 
# for model in $( cat ../Model_Lists/model_list_foldability.txt )
# do
#     echo $model
# 
#     seq_path="${sample_sele_path}/${model}/ProteinMPNN_design/seqs/"
#     gt_path="${sample_sele_path}/${model}/sequences/"
# 
#     if [ -d ${seq_path} ]
#     then
# 
#         for align in 0 1
#         do
#             if [ ${align} == 1 ]
#             then
#                 out_path="${sample_sele_path}/${model}/foldability_SR-align_dict.pkl"
#             else
#                 out_path="${sample_sele_path}/${model}/foldability_SR_dict.pkl"
#             fi
# 
#             python foldability_cal.py \
#             --seq_path ${seq_path} \
#             --gt_path  ${gt_path}  \
#             --out_path ${out_path} \
#             --alignment ${align}
#  
#         done
#     else
#         echo "No sequence predicted."
#     fi
# 
#     echo ''
# done  #> Record/foldability_2.txt

# ###### different alignment ######
# 
# gen_struc_path='../../Results/originDiff/Uncon_traj/struc_sele/'
# gen_seq_path='../../Results/originDiff/Uncon_traj/seq_sele/'
# design_seq_path='../../Results/originDiff/Uncon_traj/ProteinMPNN_design/'
# out_dir='../../Results/originDiff/Foldability/'
# 
# for model in $( cat ../Model_Lists/model_list_foldability_align.txt )
# do
#     echo $model
# 
#     for step_1 in 0 1
#     do
#         ### sequence prepare
# 
#         gen_stuc="${gen_struc_path}/${model}_step${step_1}/"
#         gen_seq="${gen_seq_path}/${model}_step${step_1}.fasta"
# 
#         python fasta_prepare.py \
#         --feature_path ${gen_stuc} \
#         --out_path ${gen_seq} 
# 
#         ### foldability
# 
#         for step_2 in 0 1 
#         do 
# 
#             design_seq="${design_seq_path}/${model}_step${step_2}/seqs/"
#             token_match="_${step_1}_/_${step_2}_"
# 
#             echo "seq${step_1}-str${step_2}"
# 
#             for align in 1
#             do
#                 if [ ${align} == 1 ]
#                 then
#                     out_path="${out_dir}/foldability_${model}_seq${step_1}-str${step_2}_align_dict.pkl"
#                 else
#                     out_path="${out_dir}/foldability_${model}_seq${step_1}-str${step_2}_dict.pkl"
#                 fi
# 
#                 python foldability_cal.py \
#                 --seq_path ${design_seq} \
#                 --gt_path  ${gen_seq}  \
#                 --out_path ${out_path} \
#                 --alignment ${align} \
#                 --token_match ${token_match}
#  
#             done
# 
#         done
# 
#     echo ''
# 
#     done
# done  > Record/foldability_stepAlign.txt

############################ Joint diffusion #################################

# server='Faster'
server='Grace'

#for work_path in ../../Results/jointDiff/codesign_diffab_* ../../Results/jointDiff_2/codesign_diffab_*
#for work_path in ../../Results/jointDiff_3/codesign_diffab_* ../../Results/jointDiff_4/codesign_diffab_*
#for work_path in ../../Results/jointDiff/codesign_diffab_* ../../Results/jointDiff_3/codesign_diffab_*
for work_path in ../../Results/jointDiff_updated/codesign_diffab_complete_gen_share-true_dim-128-64-4_step100_lr1.e-4_wd0._posiscale10.0_sc_center_2024_10_21__23_49_44_loss-1-mse-0-0_grace/ ../../Results/jointDiff_updated/codesign_diffab_complete_gen_share-true_dim-128-64-4_step100_lr1.e-4_wd0._posiscale10.0_sc_center_2024_10_21__23_50_01_loss-1-l1-1-1_grace/ ../../Results/jointDiff_updated/codesign_diffab_complete_gen_share-true_dim-128-64-4_step100_lr1.e-4_wd0._posiscale10.0_sc_center_2024_10_21__23_50_01_loss-1-mse-1-0_grace ../../Results/jointDiff_updated/codesign_diffab_complete_gen_share-true_dim-128-64-4_step100_lr1.e-4_wd0._posiscale10.0_sc_center_2024_10_21__23_50_01_loss-1-mse-1-1_grace
do

    model=${work_path##*/}
    echo $model
    seq_gen="${work_path}/seq_gen.fa"
    struc_path="${work_path}/samples/"

    echo '############################################################'

    mpnn_path_0="${work_path}/mpnn-pred/seqs/"
    #mpnn_path_1="${work_path}/mpnn-pred/seqs_step1/"
    #mpnn_path_2="${work_path}/mpnn-pred/seqs_step2/"

    #if [ ! -d ${mpnn_path_1} ]
    #then
    #    mkdir ${mpnn_path_1}
    #    mv ${mpnn_path_0}/*_1_*.fa ${mpnn_path_1}
    #fi

    #if [ ! -d ${mpnn_path_2} ]
    #then
    #    mkdir ${mpnn_path_2}
    #    mv ${mpnn_path_0}/*_2_*.fa ${mpnn_path_2}
    #fi

    #out_path="${work_path}/consistency-seq_dict${idx}.pkl"
    out_path="${work_path}/consistency-seq_dict_step0.pkl"
    if [ -f ${out_path} ] 
    then
        continue
    fi

    for seq_path in ${mpnn_path_0} #${mpnn_path_1} ${mpnn_path_2} 
    do 
        idx=${seq_path##*seqs}
        idx=${idx%%/*}

        python foldability_cal.py \
	--seq_path ${seq_path} \
	--gt_path  ${seq_gen}  \
	--out_path ${out_path} \
	--alignment 1

        # ./wrapper_submit_foldability.sh \
        # --struc_path ${struc_path} \
        # --seq_path ${seq_path} \
        # --gt_path ${seq_gen}  \
        # --out_path ${out_path} \
        # --title "foldability-2_${model}${idx}" \
	# --server ${server} \
        # --job_list ${job_list}
    done

    echo ''

done 

 
############################## Autoencoder ####################################

# for design_path in $( cat ../Model_Lists/sample_list_autoencoder_foldability.txt )
# do
#     model=${design_path%/files_sample*}
#     model=${model##*/}
#     name=${design_path##*/files_sample_}
# 
#     mpnn_path="/scratch/user/shaowen1994/DiffCodesign_local/Results/autoencoder/samples/${model}/mpnn-pred_${name}/seqs/"
#     out_path="/scratch/user/shaowen1994/DiffCodesign_local/Results/autoencoder/samples/${model}/foldability_${name}.pkl"
# 
#     python foldability_cal.py \
#     --seq_path ${mpnn_path} \
#     --gt_path  ${design_path}  \
#     --out_path ${out_path} \
#     --alignment 1 
# 
# done


############################## LatentDiff #####################################

# for model in $( cat ../Model_Lists/model_list_foldability_latent.txt )
# do
#     echo $model
#     echo '############################################################'
#     ./wrapper_experiment_foldability.sh --version $model
#     echo ''
# done  > Record/foldability_latentdiff.txt


# for model in $( cat ../Model_Lists/model_list_foldability_latent.txt )
# do
#     echo $model
#     echo '############################################################'
#     ./wrapper_experiment_foldability.sh --version $model --method 'esm-if'
#     echo ''
# done  > Record/foldability_latentdiff_esm-if_v2.txt

