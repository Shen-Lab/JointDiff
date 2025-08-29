### Prepare jobs for fitness score cal

seq_path="../../Results/diffab_pre/Uncon_seq/codesign_SingleChain_fitness_2023_10_03__01_30_24_diffab_fitness-none.fasta"
result_path="../../Results/diffab_pre/Fitness/codesign_SingleChain_fitness_2023_10_03__01_30_24_diffab_fitness-none.pkl"
batch_size=8
title="none"
environment='diffab'
job_time=10
gpu_num=1
server='Faster'
job_list='Job_list_fitness_cal.txt'

ARGUMENT_LIST=(
    "seq_path"
    "result_path"
    "title"
    "batch_size"
    "environment"
    "job_time"
    "gpu_num"
    "server"
    "job_list"
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
        --seq_path)
            seq_path=$2
            shift 2
            ;;

        --result_path)
            result_path=$2
            shift 2
            ;;

        --title)
            title=$2
            shift 2
            ;;

        --batch_size)
            batch_size=$2
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

        --gpu_num)
            gpu_num=$2
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
    account='132821644222'  #'132821649270'
    conda='Anaconda3/2020.07'
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

if [ $title == 'none' ]
then
    title=${seq_path##*/}
    title=${title%.fa*}
fi

title="fitness_${title}"

run_command="python sample_fitness_cal.py \
--in_path ${seq_path} \
--out_path ${result_path} \
--batch_size ${batch_size}" 

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
#SBATCH --ntasks=28
#SBATCH --mem=50G                  
#SBATCH --output=Output/output_${title}

##OPTIONAL JOB SPECIFICATIONS
#SBATCH --account=${account}       #Set billing account to
#SBATCH --gres=gpu:${gpu_num}                 #Request 1 GPU per node can be 1 or 2
#SBATCH --partition=gpu              #Request the GPU partition/queue

#First Executable Line
source ~/.bashrc
module load ${conda}
source activate ${environment}
${cuDNN}

cd ../
cp -r Evaluation  Temp_${title}
cd Temp_${title}

${run_command}

cd ${current}
rm -r ../Temp_${title}

source deactivate" > ${server}_jobs/${server}_${title}.job

echo "${server}_jobs/${server}_${title}.job" >> ${job_list}
#sbatch ${server}_jobs/${server}_${title}.job 
