### <Function of the scripts>

version='seq'
input='../../Results/Chroma/seq_gen_sele.500.fa'
output='../../Results/Chroma/go_pred_foldseek-seq.txt'
database='/scratch/user/shaowen1994/Tools/Foldseek/foldseek/database/swiss_prot/swiss-prot'
prostt5_model='/scratch/user/shaowen1994/Tools/Foldseek/foldseek/prostt5_out/model/'
title='chroma_swiss-pro_seq'

environment='python3.8'
job_time=20
server='Faster'
job_list='Job_list_foldseek.txt'

ARGUMENT_LIST=(
    "version"
    "input"
    "output"
    "database"
    "prostt5_model"
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
        --version)
            version=$2
            shift 2
            ;;

        --input)
            input=$2
            shift 2
            ;;

        --output)
            output=$2
            shift 2
            ;;

        --database)
            database=$2
            shift 2
            ;;

        --prostt5_model)
            prostt5_model=$2
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

title="foldseek_${title}"
current=$( pwd )

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
module load ${conda}
source activate ${environment}
${cuDNN}

mkdir ../Temp_${title}
cp *.py ../Temp_${title} 
cd ../Temp_${title}

" > ${server}_jobs/${server}_${title}.job

if [ ${version} == 'seq' ]
then
    echo "foldseek easy-search ${input} ${database} ${output} tmpFolder --prostt5-model ${prostt5_model}" >> ${server}_jobs/${server}_${title}.job

else
 
echo "if [ -f ${output} ]
then                
    rm ${output}   
fi                 

for target_pdb in ${input}/*pdb
do
    echo \${target_pdb}
    foldseek easy-search \${target_pdb} ${database} result.txt tmpFolder
    cat result.txt >> ${output}
done" >> ${server}_jobs/${server}_${title}.job

fi

echo "
cd ${current}
rm -r ../Temp_${title}
source deactivate" >> ${server}_jobs/${server}_${title}.job

echo "${server}_jobs/${server}_${title}.job" >> ${job_list}
#sbatch ${server}_jobs/${server}_${title}.job
