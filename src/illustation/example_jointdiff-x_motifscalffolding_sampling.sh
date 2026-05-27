### target description
data_path="../data/motif-scaffolding_benchmark/benchmark.csv"
### processed data; if not found, it will be automatically processed and saved at this path
info_dict_path="../data/motif-scaffolding_benchmark/benchmark_processed.pkl"
### path of the motif structcures
pdb_path="../data/motif-scaffolding_benchmark/pdbs_processed/"

ckpt_path="../checkpoints/JointDiff-x_rm_fape-dist-disto.pt"

############################################################
# original length
############################################################

# sample_length=0
# out_path="../samples/motifscaffolding_jointdiff-x_samelength/"
# mkdir -p -v ${out_path}
# 
# python infer_motifscaffolding_jointdiff.py \
# --data_path ${data_path} \
# --info_dict_path ${info_dict_path} \
# --pdb_path ${pdb_path}  \
# --model_path ${ckpt_path} \
# --result_path ${out_path} \
# --attempt 5 \
# --sample_length ${sample_length} \
# --save_type 'last'

############################################################
# sampling length
############################################################

sample_length=1
out_path="../samples/motifscaffolding_jointdiff-x_sampledlength/"
mkdir -p -v ${out_path}

python infer_motifscaffolding_jointdiff.py \
--data_path ${data_path} \
--info_dict_path ${info_dict_path} \
--pdb_path ${pdb_path}  \
--model_path ${ckpt_path} \
--result_path ${out_path} \
--attempt 5 \
--sample_length ${sample_length} \
--save_type 'last'

