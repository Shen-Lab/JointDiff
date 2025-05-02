###### guided diffusion ######

# #for name in $( cat ../Model_Lists/model_list_feat-cal_guided.txt )
# for name in codesign_multinomial_step100_posiscale17.6_6-512-256_2024_02_01__00_55_13 
# do
#     echo $name
# 
#     #in_path="../../Results/guidedDiff/sample_sele_forAF2/${name}/structures/"
#     #out_path="../../Results/guidedDiff/sample_sele_forAF2/${name}/structures_forProteinMPNN/"
#     in_path="../../Results/originDiff/sample_sele_forAF2/${name}/structures/"
#     out_path="../../Results/originDiff/sample_sele_forAF2/${name}/structures_forProteinMPNN/"
# 
#     if [ ! -d ${in_path} ]
#     then
#         echo "${in_path} does not exist!"
#     else
#         mkdir ${out_path}
# 
#         for pdb in ${in_path}/*.pdb
#         do
#             out_file=${pdb##*/}
#             out_file="${out_path}/${out_file}"
# 
#             python pdb_formatting.py \
#             --pdb_path ${pdb} \
#             --out_path ${out_file} \
#             --print_status 1
# 
#         done
# 
#     fi
# 
# done

###### step align ######

for ori_path in ../../Results/originDiff/Uncon_traj/struc_sele/codesign_diffab_complete_gen_share-true_step100_lr1.e-4_wd0.0_posiscale100.0_2024_01_20__16_37_11_step0_ori
do
    echo $ori_path

    tar_path=${ori_path%_ori*}

    for pdb in ${ori_path}/*pdb
    do
        tar_pdb="${tar_path}/${pdb##*/}"

        python pdb_formatting.py \
        --pdb_path ${pdb} \
        --out_path ${tar_pdb} \
        --print_status 1

    done

done


