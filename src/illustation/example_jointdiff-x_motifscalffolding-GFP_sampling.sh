### target description
data_path="../data/motif-scaffolding_design/motif.csv"
### processed data; if not found, it will be automatically processed and saved at this path
info_dict_path="../data/motif-scaffolding_design/motif_processed.pkl"
### path of the motif structcures
pdb_path="../data/motif-scaffolding_design/motif_pdbs/"

ckpt_path="../checkpoints/JointDiff-x_rm_fape-dist-disto.pt"
out_path="../samples/motifscaffolding-GFP_jointdiff-x/"
mkdir -p -v ${out_path}

python infer_motifscaffolding_jointdiff.py \
--data_path ${data_path} \
--info_dict_path ${info_dict_path} \
--pdb_path ${pdb_path}  \
--model_path ${ckpt_path} \
--result_path ${out_path} \
--attempt 5 \
--sample_length 0 \
--save_type 'last'
