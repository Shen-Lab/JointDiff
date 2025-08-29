### <Function of the scripts>

title='consistency-struc'

environment='diffab'
job_time='06'
server='Grace'
job_list='Job_list_consistency-struc.txt'

ARGUMENT_LIST=(
    "ref_path"
    "pred_path"
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
        --ref_path)
            ref_path=$2
            shift 2
            ;;

        --pred_path)
            pred_path=$2
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
    account='132821649552'
    conda=''
    cuDNN='module load cuDNN/8.9.2.26-CUDA-12.1.1'
else                           # Faster
    account='142788516569'
    conda='module load Anaconda3/2021.11'
    cuDNN='module load cuDNN/8.9.2.26-CUDA-12.1.1'
fi

title=${title}
run_command="python consistency-struc_cal.py \
--ref_path ${ref_path} \
--pred_path ${pred_path} \
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
${conda}
source activate ${environment}
${cuDNN}

mkdir ../Temp_${title}
cp *.py ../Temp_${title} 
cp TMalign_cpp ../Temp_${title} 
cd ../Temp_${title}

${run_command}

cd ${current}
rm -r ../Temp_${title}
source deactivate" > ${server}_jobs/${server}_${title}.job

echo "${server}_jobs/${server}_${title}.job" >> ${job_list}
#sbatch ${server}_jobs/${server}_${title}.job
