### <Function of the scripts>

struc_path='../../Results/originDiff/forward-diff_struc/struc/Step100_posiscale10.0/'
contact_path='../../Data/Processed/CATH_forDiffAb/ContactMap_test_rearanged/'
out_path='../../Results/originDiff/sbmopenmm_check/forward_Step100_posiscale10.0_100-1.pkl'
with_seq=1
CA_only=1
torsion_energy=1.0
contact_energy=1.0
temperature=1.0
job_num=1
job_idx=1

environment='diffab'
job_time=10
server='Faster'
job_list='Job_list.txt'

ARGUMENT_LIST=(
    "struc_path"
    "contact_path"
    "out_path"
    "with_seq"
    "CA_only"
    "torsion_energy"
    "contact_energy"
    "temperature"
    "job_num"
    "job_idx"
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

        --contact_path)
            contact_path=$2
            shift 2
            ;;

        --out_path)
            out_path=$2
            shift 2
            ;;

        --with_seq)
            with_seq=$2
            shift 2
            ;;

        --CA_only)
            CA_only=$2
            shift 2
            ;;

        --torsion_energy)
            torsion_energy=$2
            shift 2
            ;;

        --contact_energy)
            contact_energy=$2
            shift 2
            ;;

        --temperature)
            temperature=$2
            shift 2
            ;;

        --job_num)
            job_num=$2
            shift 2
            ;;

        --job_idx)
            job_idx=$2
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
    conda='Anaconda3/2020.07'
    cuDNN='module load cuDNN/8.0.5.39-CUDA-11.1.1'
elif [ ${server} == 'Terra' ]  # Terra
then
    account='122821642941'
    conda='Anaconda/3-5.0.0.1'
    cuDNN='module load cuDNN/8.2.1.32-CUDA-11.3.1'
else                           # Faster
    account='142788516569'
    conda='module load Anaconda3/2021.11'
    cuDNN='module load cuDNN/8.0.4.30-CUDA-11.1.1'
fi

title=${out_path##*/}
title=${title%.*}

run_command="python sbm_energy_test.py \
--struc_path ${struc_path} \
--contact_path ${contact_path} \
--out_path ${out_path} \
--with_seq ${with_seq} \
--CA_only ${CA_only} \
--torsion_energy ${torsion_energy} \
--contact_energy ${contact_energy} \
--temperature ${temperature} \
--job_num ${job_num} \
--job_idx ${job_idx} \
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
cd ../Temp_${title}

${run_command}

cd ${current}
rm -r ../Temp_${title}
source deactivate" > ${server}_jobs/${server}_${title}.job

echo "${server}_jobs/${server}_${title}.job" >> ${job_list}
#sbatch ${server}_jobs/${server}_${title}.job
