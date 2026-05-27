ckpt_path="../checkpoints/JointDiff_model.pt"
out_path="../samples/unconditional_jointdiff/"
mkdir -p -v ${out_path}

python infer_jointdiff.py \
--model_path ${ckpt_path} \
--result_path ${out_path} \
--size_range 100 200 20 \
--num 5 \
--save_type 'last'
