### prepare evaluation jobs for the autoencoder

sample_path='../../Results/autoencoder/samples/autoencoder_joint-mlp-4-512/files_sample_autoencoder_joint-mlp-4-512_in-seq_test/'
gt_path_seq='../../Data/Processed/CATH_seq/CATH_seq_test.fasta'
gt_path_structure='../../Data/Origin/CATH/pdb_all_AtomOnly/'
out_path='../../Results/autoencoder/ModalityRecovery/autoencoder_joint-mlp-4-512/results_in-seq_test.pkl'
title='autoencoderEval_autoencoder_joint-mlp-4-512_in-seq_test'

environment='codesign'
job_time=02
server='Grace'
job_list='Job_list_autoencoder_eval.txt'

ARGUMENT_LIST=(
    "sample_path"
    "gt_path_seq"
    "gt_path_structure"
    "out_path"
    "title"
    "environment"
    "job_time"
    "server"
    "job_list"
)

# read arguments
opts=$(getopt     --longoptions "$(printf "%s:," "${ARGUMENT_LIST[@]}")" \
    --name "$(basename "$0")" \
    --options "" \
    -- "$@"
)

eval set --$opts

while [[ $# -gt 0 ]]; do
    case "$1" in
        --sample_path)
            sample_path=$2
            shift 2
            ;;

        --gt_path_seq)
            gt_path_seq=$2
            shift 2
            ;;

        --gt_path_structure)
            gt_path_structure=$2
            shift 2
            ;;

        --out_path)
            out_path=$2
            shift 2
            ;;

        --title)
            title=$2
            shift 2
            ;;

        --environment)
            environment=$2
            shift 2
            ;;

        --job_time)
            job_time=$2
            shift 2
            ;;

        --server)
            server=$2
            shift 2
            ;;

        --job_list)
            job_list=$2
            shift 2
            ;;

        *)
            break
            ;;
    esac
done

if [ ! -d "${server}_jobs" ]
then
    mkdir ${server}_jobs
fi

if [ ! -d "Output" ]
then
    mkdir Output
fi

if [ ${server} == 'Grace' ]  # Grace
then
    account='132821644222'
    conda='Anaconda3/2023.07-2'
    cuDNN='module load cuDNN/8.0.5.39-CUDA-11.1.1'
elif [ ${server} == 'Terra' ]  # Terra
then
    account='122821642941'
    conda='Anaconda/3-5.0.0.1'
    cuDNN='module load cuDNN/8.2.1.32-CUDA-11.3.1'
else                           # Faster
    account='142788516569'
    conda='Anaconda3/2021.11'
    cuDNN='module load cuDNN/8.0.4.30-CUDA-11.1.1'
fi

title=${title}
run_command="python autoencoder_eval.py \
--sample_path ${sample_path} \
--gt_path_seq ${gt_path_seq} \
--gt_path_structure ${gt_path_structure} \
--out_path ${out_path} \
"

current=$( pwd )

echo "Job Title:" ${title}
echo "Command:" ${run_command}

echo "#!/bin/bash
##ENVIRONMENT SETTINGS; CHANGE WITH CAUTION
#SBATCH --export=NONE                #Do not propagate environment
#SBATCH --get-user-env=L             #Replicate login environment

##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=${title}
#SBATCH --time=${job_time}:00:00              
#SBATCH --ntasks=1
#SBATCH --mem=50G                  
#SBATCH --output=Output/output_${title}

##OPTIONAL JOB SPECIFICATIONS
#SBATCH --account=${account}       #Set billing account to 

#First Executable Line
source ~/.bashrc
module load ${conda}
source activate ${environment}
${cuDNN}

mkdir ../Temp_${title}
cp *.py ../Temp_${title} 
cp TMalign ../Temp_${title} 
cd ../Temp_${title}

${run_command}

cd ${current}
rm -r ../Temp_${title}
source deactivate" > ${server}_jobs/${server}_${title}.job

echo "${server}_jobs/${server}_${title}.job" >> ${job_list}
#sbatch ${server}_jobs/${server}_${title}.job
