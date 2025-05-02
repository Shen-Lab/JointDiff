### <Function of the scripts>

fasta_file='none'
nature_file='../../Data/Processed/CATH_seq/CATH_seq_all.fasta'
cluster_method='diamond'
threshold=0.3

environment='diffab'
job_time=10
server='Faster'
job_list='Job_list_MSI.txt'

ARGUMENT_LIST=(
    "fasta_file"
    "nature_file"
    "merged_file"
    "cluster_file"
    "outpath"
    "cluster_method"
    "threshold"
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
        --fasta_file)
            fasta_file=$2
            shift 2
            ;;

        --nature_file)
            nature_file=$2
            shift 2
            ;;

        --merged_file)
            merged_file=$2
            shift 2
            ;;

        --cluster_file)
            cluster_file=$2
            shift 2
            ;;

        --outpath)
            outpath=$2
            shift 2
            ;;

        --cluster_method)
            cluster_method=$2
            shift 2
            ;;

        --threshold)
            threshold=$2
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
    account='132821649270'
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

###################### preprocess #############################################

if [ ! -f ${merged_file} ]
then
    cat ${fasta_file} ${nature_file} > ${merged_file}
fi

###################### title and commands #####################################

title=${title}

if [ ${cluster_method} == 'diamond' ]
then
    ### diamond clustering
    title="diamond_${title}"
    thre_percent=$(echo "scale=4; $threshold*100" | bc)
    cluster_command="diamond cluster -d ${merged_file} -o ${cluster_file} --approx-id ${thre_percent} -M 64G"
    move_command=""

else
    ### mmseqs2 clustering
    title="mmseqs2_${title}"
    cluster_command="mmseqs easy-cluster ${merged_file} ${title} tmp --min-seq-id $threshold -c 0.8 --cov-mode 1"
    move_command="mv ${title}_cluster.tsv ${cluster_file}"
fi

run_command="python MSI_cal.py \
--seq_file ${merged_file} \
--cluster_file ${cluster_file} \
--outpath ${outpath} \
--cluster_method ${cluster_method} \
--threshold ${threshold} \
"

current=$( pwd )

echo "Job Title:" ${title}
echo "Commands:" 
echo ${cluster_command}
echo ${move_command}
echo ${run_command}

###################### output #################################################

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
cp diamond ../Temp_${title}
cp *.py ../Temp_${title} 
cd ../Temp_${title}

${cluster_command}
${move_command}
${run_command}

cd ${current}
rm -r ../Temp_${title}
source deactivate" > ${server}_jobs/${server}_${title}.job

echo "${server}_jobs/${server}_${title}.job" >> ${job_list}
#sbatch ${server}_jobs/${server}_${title}.job
