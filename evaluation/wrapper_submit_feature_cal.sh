### <Function of the scripts>

nature_path='../../Results/Baselines/Nature/features_summary.pkl'
token='_0_'

environment='diffab'
job_time=10
server='Faster'
job_list='Job_list_feat-cal.txt'

ARGUMENT_LIST=(
    "in_path"
    "out_path"
    "summary_path"
    "nature_path"
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
        --in_path)
            in_path=$2
            shift 2
            ;;

        --out_path)
            out_path=$2
            shift 2
            ;;

        --summary_path)
            summary_path=$2
            shift 2
            ;;

        --nature_path)
            nature_path=$2
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
    conda=''
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

record_path="${summary_path%.*}_record.txt"

current=$( pwd )

title="feat_${title}"
echo "Job Title:" ${title}

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

### feature dict

for pdb in ${in_path}/*${token}*.pdb
do
    name=\${pdb##*/}
    name=\${name%.*}
    echo \$name

    out_file=\"${out_path}/\${name}.pkl\"

    if [ -f \${out_file} ]
    then
        continue
    fi

    python pdb_feature_cal.py --in_path \${pdb} --out_path \${out_file}

done

### feature summary

python pdb_feature_summarize.py \
--feat_path ${out_path} \
--out_path ${summary_path} \
--token ${token} \
--nature_path ${nature_path} > ${record_path} 

cd ${current}
rm -r ../Temp_${title}
source deactivate" > ${server}_jobs/${server}_${title}.job

echo "${server}_jobs/${server}_${title}.job" >> ${job_list}
#sbatch ${server}_jobs/${server}_${title}.job
