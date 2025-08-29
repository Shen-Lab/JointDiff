version='latentdiff_with-ESM-IF_joint-mlp-4-512_pad-zero_dim16_noEnd_withMask_gt_Faster'
work_dir='/scratch/user/shaowen1994/DiffCodesign_local/Results/latentDiff/'
method='proteinmpnn'

ARGUMENT_LIST=(
    "version"
    "work_dir"
    "method"
)

# read arguments
opts=$(getopt \
    --longoptions "$(printf "%s:," "${ARGUMENT_LIST[@]}")" \
    --name "$(basename "$0")" \
    --options "" \
    -- "$@"
)

eval set --$opts

while [[ $# -gt 0 ]]; do
    case "$1" in
        --version)
            version=$2
            shift 2
            ;;

        --work_dir)
            work_dir=$2
            shift 2
            ;;

        --method)
            method=$2
            shift 2
            ;;

        *)
            break
            ;;
    esac
done

##################### path #################################

work_dir="${work_dir}/${version}/"

### gt sequence
in_path="${work_dir}/samples/files_sample_gen/"
if [ -z ${in_path} ]
then
    echo 'Input is empty!'
    exit
fi

### predicted sequence

if [ ${method} == 'proteinmpnn' ]
then
    tar_path="${work_dir}/seq_pred_mpnn/seqs/"
    out_path="${work_dir}/foldability_dict.pkl"
    mpnn_format=1
else
    tar_path="${work_dir}/seq_pred_emsif/"
    out_path="${work_dir}/foldability_esm-if_dict.pkl"
    mpnn_format=0
fi 

if [ -z ${tar_path} ]
then
    echo 'No predicted sequence!'
    exit
fi

##################### calculation ##########################

echo "python foldability_cal.py \
--seq_path ${tar_path} \
--gt_path ${in_path} \
--out_path ${out_path} \
--mpnn_format ${mpnn_format}
"
echo "****************************************************"

python foldability_cal.py \
--seq_path ${tar_path} \
--gt_path ${in_path} \
--out_path ${out_path} \
--mpnn_format ${mpnn_format}
