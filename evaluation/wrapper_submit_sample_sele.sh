### <Function of the scripts>

seq_path='../../Results/Chroma/seq_gen.fa'
struc_path='../../Results/Chroma/samples/'
seq_out='../../Results/Chroma/seq_gen_sele.500.fa'
struc_out='../../Results/Chroma/struc_gen_sele.500/'
sele_num=500
token='none'
title=''

environment='python3.8'
job_time=01
server='Faster'
job_list='Job_list_sample_sele.txt'

ARGUMENT_LIST=(
    "seq_path"
    "struc_path"
    "seq_out"
    "struc_out"
    "sele_num"
    "token"
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
        --seq_path)
            seq_path=$2
            shift 2
            ;;

        --struc_path)
            struc_path=$2
            shift 2
            ;;

        --seq_out)
            seq_out=$2
            shift 2
            ;;

        --struc_out)
            struc_out=$2
            shift 2
            ;;

        --sele_num)
            sele_num=$2
            shift 2
            ;;

        --token)
            token=$2
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
    account='132821649552'
    #conda='module load Anaconda3/2020.07'
    conda=''
    cuDNN='module load cuDNN/8.0.5.39-CUDA-11.1.1'
else                           # Faster
    account='142788516569'
    conda='module load Anaconda3/2021.11'
    cuDNN='module load cuDNN/8.0.4.30-CUDA-11.1.1'
fi

title="sample_sele_${title}"

run_command="python sample_sele.py \
--seq_path ${seq_path} \
--struc_path ${struc_path} \
--seq_out ${seq_out} \
--struc_out ${struc_out} \
--sele_num ${sele_num} \
--token ${token} \
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
#source ~/.bashrc
${conda}
source activate ${environment}
${cuDNN}

mkdir ../Temp_${title}
cp *.py ../Temp_${title} 
cd ../Temp_${title}

${run_command}

cd ${current}
rm -r ../Temp_${title}
source deactivate" > ${server}_jobs/${server}_${title}.job

echo "${server}_jobs/${server}_${title}.job" >> ${job_list}
#sbatch ${server}_jobs/${server}_${title}.job
