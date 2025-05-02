
############################################################################
# foldingdiff
############################################################################

# pdb_path="../../Results/foldingdiff/sampled_pdb/"
# out_path="../../Results/foldingdiff/features/"
# 
# for pdb in ${pdb_path}*
# do
#     name=${pdb##*/}
#     name=${name%.*}
#     out_file="${out_path}${name}.pkl"
# 
#     echo $name
#     python pdb_feature_cal.py --in_path ${pdb} --out_path ${out_file}
# done > Record/foldingdiff_feat.txt

############################################################################
# RFdiffusion
############################################################################

# pdb_path="../../Results/RFdiffusion/sampled_pdb_mixed/"
# out_path="../../Results/RFdiffusion/features/"
# 
# for pdb in ${pdb_path}*
# do
#     name=${pdb##*/}
#     name=${name%.*}
#     out_file="${out_path}${name}.pkl"
# 
#     echo $name
#     python pdb_feature_cal.py --in_path ${pdb} --out_path ${out_file}
# done > Record/RFdiffusion_feat.txt

############################################################################
# Chroma
############################################################################

#pdb_path="../../Results/Chroma/sampled_pdb/"
#out_path="../../Results/Chroma/Features/"

#for pdb in ${pdb_path}/*.pdb
#do
#    name=${pdb##*/}
#    name=${name%.*}
#    out_file="${out_path}${name}.pkl"

#    echo $name
#    python pdb_feature_cal.py --in_path ${pdb} --out_path ${out_file}
#done > Record/Chroma_feat.txt

############################################################################
# ProteinGenerator
############################################################################

# pdb_path="../../Results/protein_generator/sampled_pdb/"
# out_path="../../Results/protein_generator/Features/"
# 
# for pdb in ${pdb_path}/*.pdb
# do
#     name=${pdb##*/}
#     name=${name%.*}
#     out_file="${out_path}${name}.pkl"
# 
#     echo $name
#     python pdb_feature_cal.py --in_path ${pdb} --out_path ${out_file}
# done > Record/proteingenerator_feat.txt

############################################################################
# Nature
############################################################################

# pdb_path="../../Data/Origin/CATH/pdbs_eval/"
# out_path="../../Results/Nature/features/"
# 
# for pdb in ${pdb_path}*
# do
#     name=${pdb##*/}
#     name=${name%.*}
#     out_file="${out_path}${name}.pkl"
# 
#     echo $name
#     python pdb_feature_cal.py --in_path ${pdb} --out_path ${out_file}
# done > Record/nature.txt

############################################################################
# ours
############################################################################

# STRUC_PATH='../../Results/originDiff/Uncon_struc/'
# OUT_PATH='../../Results/originDiff/Features/'
# 
# for model in $( cat ../Model_Lists/model_list_feat-cal.txt )
# do 
#     pdb_path="${STRUC_PATH}/${model}/"
#     out_path="${OUT_PATH}/${model}/"
#  
#     if [ -d ${out_path} ] && [ $( ls ${out_path} | wc -l ) -gt 500 ]
#     then
#         echo "Completed for ${model}."
#     else
#         if [ ! -d ${out_path} ]
#         then
#             mkdir ${out_path}
#         fi
# 
#         echo $model >> Record/feat_ours.txt
# 
#         for pdb in ${pdb_path}/*.pdb
#         do
#             name=${pdb##*/}
#             name=${name%.*}
#             echo $name
# 
#             out_file="${out_path}/${name}.pkl"
#             #echo ${pdb}
#             #echo $out_file
#  
#             python pdb_feature_cal.py --in_path ${pdb} --out_path ${out_file}
# 
#         done >> Record/feat_ours.txt
# 
#         echo '' >> Record/feat_ours.txt
#     fi
# 
#     if [ $( ls ${out_path} | wc -l ) -lt 500 ]
#     then
#         echo "Failed for ${model}!"
#     fi
# done

############################################################################
# ours (with guidance)
############################################################################

# pdb_path="../../Results/diffab/Uncon_struc/codesign_diffab_complete_gen_share-true_step100_lr1.e-4_wd0.0_2023_05_01__17_05_48/"
# out_path="../../Results/diffab/Features/codesign_diffab_complete_gen_share-true_step100_lr1.e-4_wd0.0_2023_05_01__17_05_48/"
# 
# for pdb_path in ../../Results/diffab/Uncon_struc/codesign_SingleChain_*
# do
#     version=${pdb_path##*/}
#     out_path="../../Results/diffab/Features/${version}/"
#     mkdir $out_path
# 
#     echo $pdb_path
#     echo $out_path
# 
#     for pdb in ${pdb_path}/*
#     do
#         name=${pdb##*/}
#         name=${name%.*}
#         out_file="${out_path}${name}.pkl"
#     
#         echo $name
#         python pdb_feature_cal.py --in_path ${pdb} --out_path ${out_file}
#     done
# done

#for pdb_path in ../../Results/diffab/Uncon_struc/codesign_SingleChain_*max*
#for pdb_path in ../../Results/diffab/Uncon_struc/codesign_SingleChain_*max_fitness-none*
#for pdb_path in ../../Results/diffab/Uncon_struc/*step1_* ../../Results/diffab/Uncon_struc/*step5_*
#for pdb_path in ../../Results/originDiff/Uncon_struc/*
# for pdb_path in $( cat model_list.txt )
#for pdb_path in ../../Results/originDiff/Uncon_SingleModal_struc/*modal*   
#do
#    version=${pdb_path##*/}
#    out_path="../../Results/originDiff/Features/${version}/"
#    mkdir $out_path
#
#    echo $pdb_path
#    echo $out_path
#
#    for pdb in ${pdb_path}/*
#    do
#        name=${pdb##*/}
#        name=${name%.*}
#        out_file="${out_path}${name}.pkl"
#    
#        echo $name
#        python pdb_feature_cal.py --in_path ${pdb} --out_path ${out_file}
#    done
#done

# job_list='Job_list_feat-cal.txt'
# 
# if [ -f ${job_list} ]
# then
#     rm ${job_list}
# fi
# 
# STRUC_PATH='../../Results/guidedDiff/Uncon_struc/'
# OUT_PATH='../../Results/guidedDiff/Features/'
# 
# for model in $( cat ../Model_Lists/model_list_feat-cal_guided.txt )
# do
#     pdb_path="${STRUC_PATH}/${model}/"
#     out_path="${OUT_PATH}/${model}/"
# 
#     if [ -d ${out_path} ] && [ $( ls ${out_path} | wc -l ) -gt 500 ]
#     then
#         echo "Completed for ${model}."
#     else
#         if [ ! -d ${out_path} ]
#         then
#             mkdir ${out_path}
#         fi
# 
#         ./wrapper_submit_feature_cal.sh \
#         --in_path  ${pdb_path} \
#         --out_path ${out_path} \
#         --title ${model%/*} \
#         --job_list ${job_list}
# 
#         # echo $model >> Record/feat_ours_guided.txt
# 
#         # for pdb in ${pdb_path}/*.pdb
#         # do
#         #     name=${pdb##*/}
#         #     name=${name%.*}
#         #     echo $name
# 
#         #     out_file="${out_path}/${name}.pkl"
#         #     #echo ${pdb}
#         #     #echo $out_file
# 
#         #     python pdb_feature_cal.py --in_path ${pdb} --out_path ${out_file}
# 
#         # done >> Record/feat_ours_guided.txt
# 
#         # echo '' >> Record/feat_ours_guided.txt
#     fi
# 
#     #if [ $( ls ${out_path} | wc -l ) -lt 500 ]
#     #then
#     #    echo "Failed for ${model}!"
#     #fi
# done

############################################################################
# jointDiff
############################################################################

job_list='Job_list_feat-cal.txt'

if [ -f ${job_list} ]
then
    rm ${job_list}
fi

server='Grace'

#for pdb_path in ../../Results/jointDiff*/codesign_diffab_*/samples
#for pdb_path in ../../Results/latentDiff/latentdiff_with-ESM-IF_*/samples/files_sample_gen
#for pdb_path in ../../Results/jointDiff/codesign_diffab_*/samples
#for pdb_path in ../../Results/jointDiff_updated/codesign_diffab_*loss*/samples  
# for pdb_path in ../../Results/Baselines/foldingdiff/samples  
# do
#    work_path=${pdb_path%/samples*}
#    #job_idx=${pdb_path##*jointDiff}
#    job_idx=${job_idx%%/sample*}
# 
#    model=${work_path##*/}
#    title="feat-cal_${model}${job_idx}"
# 
#    echo ${title}
# 
#    out_path="${work_path}/features/"
#    summary_path="${work_path}/feat_summary_dict.pkl"
# 
#    if [ ! -d ${out_path} ]
#    then
#        mkdir ${out_path}
#    fi
# 
#    ./wrapper_submit_feature_cal.sh \
#    --in_path ${pdb_path} \
#    --out_path ${out_path} \
#    --summary_path ${summary_path} \
#    --job_list ${job_list} \
#    --title ${title}
# 
#    echo '************************************************'
# 
# done

# #for work_path in ../../Results/jointDiff_updated/codesign_diffab_*loss*
# for work_path in ../../Results/jointDiff_updated_eval/codesign_diffab_*
# do
#     for token in '' # '_last'
#     do
#         pdb_path="${work_path}/samples${token}"
# 
#         model=${work_path##*/}
#         title="feat-cal_${model}${job_idx}"
# 
#         echo ${title}
# 
#         out_path="${work_path}/features_last/"
#         summary_path="${work_path}/feat_summary_dict${token}.pkl"
# 
#         if [ ! -d ${out_path} ]
#         then
#             mkdir ${out_path}
#         fi
#         if [ -f ${summary_path} ]
#         then
#             continue
#         fi
# 
#         ./wrapper_submit_feature_cal.sh \
#         --in_path ${pdb_path} \
#         --out_path ${out_path} \
#         --summary_path ${summary_path} \
#         --job_list ${job_list} \
#         --server ${server} \
#         --title ${title}
# 
#         echo '************************************************'
#     done
# done


#for work_path in ../../Results/latentDiff_updated_eval/latentdiff_*
#for work_path in ../../Results/latentDiff_updated/latentdiff_*
for work_path in ../../Results/latentDiff/latentdiff_*DiT*
do
    model=${work_path##*/}
    echo $model

    for token in '' #'_last'
    do
        pdb_path="${work_path}/samples${token}/files_sample_gen"
        if [ ! -d ${pdb_path} ]
        then 
            continue
        fi

        num=$( ls ${pdb_path} | wc -l )
        if [ $num -lt 100 ]
        then
            continue
        fi

        title="feat-cal_${model}${token}"

        echo ${title}

        out_path="${work_path}/features${token}/"
        summary_path="${work_path}/feat_summary_dict${token}.pkl"

        if [ ! -d ${out_path} ]
        then
            mkdir ${out_path}
        fi

        ./wrapper_submit_feature_cal.sh \
        --in_path ${pdb_path} \
        --out_path ${out_path} \
        --summary_path ${summary_path} \
        --job_list ${job_list} \
        --server ${server} \
        --token '_step0' \
        --title ${title}

        echo '************************************************'
    done
done

#for pdb_path in ../../Results/latentDiff_DiT/*/samples*
for pdb_path in ../../Results/latentDiff_DiT/*/samples*
do
    if [ ! -d ${pdb_path} ]
    then 
        continue
    fi

    num=$( ls ${pdb_path} | wc -l )
    if [ $num -lt 100 ]
    then
        continue
    fi

    work_path=${pdb_path%/samples*}
    model=${work_path##*/}

    version=${pdb_path##*/samples}
    version=${version%/*}
    title="feat-cal_${model}${version}"

    echo ${title}

    out_path="${work_path}/features${version}/"
    summary_path="${work_path}/feat_summary_dict${version}.pkl"

    if [ -f ${summary_path} ]
    then
        continue
    fi

    if [ ! -d ${out_path} ]
    then
        mkdir ${out_path}
    fi

    ./wrapper_submit_feature_cal.sh \
    --in_path ${pdb_path} \
    --out_path ${out_path} \
    --summary_path ${summary_path} \
    --job_list ${job_list} \
    --server ${server} \
    --token '_step0' \
    --title ${title}

    echo '************************************************'
done
