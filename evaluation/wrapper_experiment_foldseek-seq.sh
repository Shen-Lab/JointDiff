database='../../../Tools/Foldseek/foldseek/database/pdb/pdb'
model='../../../Tools/Foldseek/foldseek/prostt5_out/'

in_seq=$1
out_path=$2

foldseek easy-search ${in_seq} ${database} ${out_path} tmp --prostt5-model ${model}
