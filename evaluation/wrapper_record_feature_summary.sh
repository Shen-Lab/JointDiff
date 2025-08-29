nature_path='../../Results/Nature/features_summary.pkl'

# #################### Nature #####################
# 
# python pdb_feature_summarize.py \
# --feat_path ../../Results/Nature/features/ \
# --out_path ${nature_path} \
# --nature_path 'None' > Record/nature_feat.txt
# 
# ################### Baselines ###################
# 
# for model in foldingdiff Chroma RFdiffusion
# do
#     echo $model
# 
#     feat_path="../../Results/${model}/features/"
#     out_path="../../Results/${model}/features_summary.pkl"
# 
#     python pdb_feature_summarize.py \
#     --feat_path ${feat_path} \
#     --out_path ${out_path} \
#     --nature_path ${nature_path}
# 
#     echo ''
# done > Record/baseline_feat.txt

################### Ours ###################

for model in $( cat ../Model_Lists/model_list_feat-summary.txt )
do
    echo $model

    feat_path="../../Results/originDiff/Features/${model}/"
    out_path="../../Results/originDiff/Features_Summaries/${model}_features_summary.pkl"

    python pdb_feature_summarize.py \
    --feat_path ${feat_path} \
    --out_path ${out_path} \
    --nature_path ${nature_path}

    echo ''
done > Record/origindiff_feat_3.txt


################### Ours (guided) ###################

#for model in $( cat ../Model_Lists/model_list_feat-cal_guided.txt )
#do
#    echo $model
#
#    feat_path="../../Results/guidedDiff/Features/${model}/"
#    out_path="../../Results/guidedDiff/Features_Summaries/${model}_features_summary.pkl"
#
#    python pdb_feature_summarize.py \
#    --feat_path ${feat_path} \
#    --out_path ${out_path} \
#    --nature_path ${nature_path}
#
#    echo ''
#done > Record/guideddiff_feat.txt
