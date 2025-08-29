### <Function of the scripts>

in_path='../../Results/Chroma/go_pred_foldseek-seq.txt'
out_path='../../Results/Chroma/go_foldseek-seq.pkl'
threshold=30.
title='Chroma_seq'

environment='python3.8'
job_time=01
server='Faster'
job_list='Job_list_quickGo.txt'

ARGUMENT_LIST=(
    "in_path"
    "out_path"
    "threshold"
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
        --in_path)
            in_path=$2
            shift 2
            ;;

        --out_path)
            out_path=$2
            shift 2
            ;;

        --threshold)
            threshold=$2
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
    conda=''
    cuDNN='module load cuDNN/8.0.5.39-CUDA-11.1.1'
else                           # Faster
    account='142788516569'
    conda='module load Anaconda3/2021.11'
    cuDNN='module load cuDNN/8.0.4.30-CUDA-11.1.1'
fi

title="quickGo_${title}"
run_command="python quickGO_map.py \
--in_path ${in_path} \
--out_path ${out_path} \
--threshold ${threshold} \
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
${conda}
source activate ${environment}
${cuDNN}
module load WebProxy

mkdir ../Temp_${title}
cp *.py ../Temp_${title} 
cd ../Temp_${title}

${run_command}

cd ${current}
rm -r ../Temp_${title}
source deactivate" > ${server}_jobs/${server}_${title}.job

echo "${server}_jobs/${server}_${title}.job" >> ${job_list}
#sbatch ${server}_jobs/${server}_${title}.job
