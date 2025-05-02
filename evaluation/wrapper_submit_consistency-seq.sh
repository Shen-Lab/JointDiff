### <Function of the scripts>

alignment=1
mpnn_format=1
token_match='none'
title='consistency-seq_debug'

environment='diffab'
job_time=10
server='Faster'
job_list='Job_list_consistency-seq.txt'

ARGUMENT_LIST=(
    "struc_path"
    "seq_path"
    "gt_path"
    "out_path"
    "alignment"
    "mpnn_format"
    "token_match"
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
        --struc_path)
            struc_path=$2
            shift 2
            ;;

        --seq_path)
            seq_path=$2
            shift 2
            ;;

        --gt_path)
            gt_path=$2
            shift 2
            ;;

        --out_path)
            out_path=$2
            shift 2
            ;;

        --alignment)
            alignment=$2
            shift 2
            ;;

        --mpnn_format)
            mpnn_format=$2
            shift 2
            ;;

        --token_match)
            token_match=$2
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
run_command="python consistency-seq_cal.py \
--seq_path ${seq_path} \
--gt_path ${gt_path} \
--out_path ${out_path} \
--alignment ${alignment} \
--mpnn_format ${mpnn_format} \
--token_match ${token_match} \
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
cd ../Temp_${title}

if [ ! -f ${gt_path} ]
then
    python seq_extract.py \
    --pdb_path ${struc_path} \
    --out_path ${gt_path}
fi

${run_command}

cd ${current}
rm -r ../Temp_${title}
source deactivate" > ${server}_jobs/${server}_${title}.job

echo "${server}_jobs/${server}_${title}.job" >> ${job_list}
#sbatch ${server}_jobs/${server}_${title}.job
