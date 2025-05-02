### Check the generated samples

model_list="../Model_Lists/model_stoping_Eval.txt"

struc_path="../../Results/originDiff/Uncon_struc/"
struc_gen="../Model_Lists/StruGen_stat.txt"
model_sele_list="../Model_Lists/model_stoping_Eval-sele.txt"

seq_path="../../Results/originDiff/Uncon_seq/"
seq_gen="../Model_Lists/SeqGen_stat.txt"

sample_sele_path="../../Results/originDiff/sample_sele_forAF2/"
sample_sele="../Model_Lists/SampleSele_stat.txt"
single_seq="../Model_Lists/SingleSeq_stat.txt"

AF2_pred="../Model_Lists/AF2_self-con_stat.txt"
ProteinMPNN_design="../Model_Lists/ProteinMPNN_stat.txt"
ProteinMPNN_seqAF2="../Model_Lists/ProteinMPNN_seqAF2_stat.txt"
ProteinMPNN_AF2="../Model_Lists/AF2_design_stat.txt"

ARGUMENT_LIST=(
    "model_list"
    "struc_path"
    "struc_gen"
    "model_sele_list"
    "seq_path"
    "seq_gen"
    "sample_sele_path"
    "sample_sele"
    "single_seq"
    "AF2_pred"
    "ProteinMPNN_design"
    "ProteinMPNN_seqAF2"
    "ProteinMPNN_AF2"
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
        --model_list)
            model_list=$2
            shift 2
            ;;

        --struc_path)
            struc_path=$2
            shift 2
            ;;

        --struc_gen)
            struc_gen=$2
            shift 2
            ;;

        --model_sele_list)
            model_sele_list=$2
            shift 2
            ;;

        --seq_path)
            seq_path=$2
            shift 2
            ;;

        --seq_gen)
            seq_gen=$2
            shift 2
            ;;
 
        --sample_sele_path)
            sample_sele_path=$2
            shift 2
            ;;

        --sample_sele)
            sample_sele=$2
            shift 2
            ;;

        --single_seq)
            single_seq=$2
            shift 2
            ;;

        --AF2_pred)
            AF2_pred=$2
            shift 2
            ;;

        --ProteinMPNN_design)
            ProteinMPNN_design=$2
            shift 2
            ;;

        --ProteinMPNN_seqAF2)
            ProteinMPNN_seqAF2=$2
            shift 2
            ;;

        --ProteinMPNN_AF2)
            ProteinMPNN_AF2=$2
            shift 2
            ;;
 
        *)
            break
            ;;

    esac
done

### structure check
echo "Check structure generation..."

for model in $( cat ${model_list} )
do
    name=${model%_*}

    echo $name
    if [ -d ${struc_path}/${name} ]
    then
        ls ${struc_path}/${name} | wc -l
    else
        echo "None"
    fi
done > ${struc_gen}

for model in $( cat ${model_list} )
do
    name=${model%_*}

    if [ ! -d ${struc_path}/${name} ]
    then
        echo $model
    fi
done > ${model_sele_list}

echo "Statistics recorded in ${struc_gen}."
echo "**********************************************************"

### sequence check
echo "Check sequence generation..."

for model in $( cat ${model_list} )
do
    model=${model%_*}

    echo $model
    if [ -f ${seq_path}/${model}.fasta ]
    then
        cat ${seq_path}/${model}.fasta | wc -l
    else
        echo "None"
    fi
done > ${seq_gen}

echo "Statistics recorded in ${seq_gen}."
echo "**********************************************************"

### sample sele check
echo "Check sample selection..."

for model in $( cat ${model_list} )
do
    model=${model%_*}

    echo $model
    if [ -d ${sample_sele_path}/${model}/structures/ ]
    then
        ls ${sample_sele_path}/${model}/structures/ | wc -l
    else
        echo "None"
    fi
done > ${sample_sele}

echo "Statistics recorded in ${sample_sele}."
echo "**********************************************************"

### single seq for AF2 (self-consistency)
echo "Check single sequence for AF2..."

for model in $( cat ${model_list} )
do
    model=${model%_*}

    if [[ ${model} == *"sequenceOnly"* ]]
    then
        break
    fi

    echo $model
    if [ -d ${sample_sele_path}/${model}/SingleSeq_forAF2/ ]
    then
        ls ${sample_sele_path}/${model}/SingleSeq_forAF2/ | wc -l

    elif [ -d ${sample_sele_path}/${model}/for_AF2/ ]
    then
        mkdir ${sample_sele_path}/${model}/SingleSeq_forAF2/
        
        for file in ${sample_sele_path}/${model}/for_AF2/*.fasta
        do
            name=${file##*/} 
            name=${name%.fasta*}

            head -2 ${file} > ${sample_sele_path}/${model}/SingleSeq_forAF2/${name}_1.fasta
            tail -2 ${file} > ${sample_sele_path}/${model}/SingleSeq_forAF2/${name}_2.fasta

        done

        ls ${sample_sele_path}/${model}/SingleSeq_forAF2/ | wc -l

    else
        echo "None"
    fi
done > ${single_seq}

echo "Statistics recorded in ${sample_sele}."
echo "**********************************************************"


### AF2 pred (self-consistency)
echo "Check AF2 pred (self-consistency)..."

if [ -f ${AF2_pred} ]
then
   rm ${AF2_pred}
fi

for model in $( cat ${model_list} )
do
    model=${model%_*}

    if [[ ${model} == *"sequenceOnly"* ]]
    then
        break
    fi

    echo $model >> ${AF2_pred}

    if [ -f ${sample_sele_path}/${model}/AF2_pred_failed_list.txt ]
    then
        rm ${sample_sele_path}/${model}/AF2_pred_failed_list.txt
    fi

    if [ -d ${sample_sele_path}/${model}/AF2_pred ]
    then

        python AF2_result_check.py \
        --seq_path ${sample_sele_path}/${model}/SingleSeq_forAF2/ \
        --pred_path ${sample_sele_path}/${model}/AF2_pred/ \
        --fail_path ${sample_sele_path}/${model}/AF2_pred_failed \
        --out_list ${sample_sele_path}/${model}/AF2_pred_failed_list.txt  

        ls ${sample_sele_path}/${model}/AF2_pred/ | wc -l >> ${AF2_pred}

    else
        mkdir ${sample_sele_path}/${model}/AF2_pred/

        python AF2_result_check.py \
        --seq_path ${sample_sele_path}/${model}/SingleSeq_forAF2/ \
        --pred_path ${sample_sele_path}/${model}/AF2_pred/ \
        --fail_path ${sample_sele_path}/${model}/AF2_pred_failed \
        --out_list ${sample_sele_path}/${model}/AF2_pred_failed_list.txt

        echo "None" >> ${AF2_pred}
    fi

    if [ $( cat ${sample_sele_path}/${model}/AF2_pred_failed_list.txt | wc -l ) == 0 ]
    then
        rm ${sample_sele_path}/${model}/AF2_pred_failed_list.txt
    fi

done 

echo "Statistics recorded in ${AF2_pred}."
echo "**********************************************************"

### ProteinMPNN
echo "Check ProteinMPNN design..."

for model in $( cat ${model_list} )
do
    model=${model%_*}

    if [[ ${model} == *"sequenceOnly"* ]]
    then
        break
    fi

    echo $model

    if [ -d ${sample_sele_path}/${model}/ProteinMPNN_design/seqs/ ]
    then
        ls ${sample_sele_path}/${model}/ProteinMPNN_design/seqs/ | wc -l

    elif [ -d ${sample_sele_path}/${model}/structures/ ]  # pdb formatting
    then
        if [ ! -d ${sample_sele_path}/${model}/structures_forProteinMPNN/ ]
        then
            mkdir ${sample_sele_path}/${model}/structures_forProteinMPNN/
        fi

        for pdb in ${sample_sele_path}/${model}/structures/*.pdb
        do
            out_file=${pdb##*/}
            out_file="${sample_sele_path}/${model}/structures_forProteinMPNN/${out_file}"

            python pdb_formatting.py \
            --pdb_path ${pdb} \
            --out_path ${out_file} \
            --print_status 0

        done
 
    else
        echo "None"
    fi
done > ${ProteinMPNN_design}

echo "Statistics recorded in ${ProteinMPNN_design}."
echo "**********************************************************"

### ProteinMPNN)
echo "Check ProteinMPNN design for AF2..."

for model in $( cat ${model_list} )
do
    model=${model%_*}

    if [[ ${model} == *"sequenceOnly"* ]]
    then
        break
    fi

    echo $model

    if [ -d ${sample_sele_path}/${model}/ProteinMPNN_design/seqs_forAF2/ ]
    then
        ls ${sample_sele_path}/${model}/ProteinMPNN_design/seqs_forAF2/ | wc -l

    elif [ -d ${sample_sele_path}/${model}/ProteinMPNN_design/seqs/ ]
    then
        mkdir ${sample_sele_path}/${model}/ProteinMPNN_design/seqs_forAF2/

        python proteinmpnn_seq_for_AF2.py \
        --in_path ${sample_sele_path}/${model}/ProteinMPNN_design/seqs/ \
        --out_path ${sample_sele_path}/${model}/ProteinMPNN_design/seqs_forAF2/ \
        --sele_num 1

        ls ${sample_sele_path}/${model}/ProteinMPNN_design/seqs_forAF2/ | wc -l

    else
        echo "None"
    fi
done > ${ProteinMPNN_seqAF2}

echo "Statistics recorded in ${ProteinMPNN_seqAF2}."
echo "**********************************************************"

### AF2 pred (designability)
echo "Check AF2 pred (designability)..."

if [ -f ${ProteinMPNN_AF2} ]
then
   rm ${ProteinMPNN_AF2}
fi

for model in $( cat ${model_list} )
do
    model=${model%_*}

    if [[ ${model} == *"sequenceOnly"* ]]
    then
        break
    fi

    echo $model >> ${ProteinMPNN_AF2}

    if [ -f ${sample_sele_path}/${model}/design_AF2_pred_failed_list.txt ]
    then
        rm ${sample_sele_path}/${model}/design_AF2_pred_failed_list.txt
    fi

    if [ -d ${sample_sele_path}/${model}/design_AF2_pred ] && [ -d ${sample_sele_path}/${model}/ProteinMPNN_design/seqs_forAF2/ ]
    then

        python AF2_result_check.py \
        --seq_path ${sample_sele_path}/${model}/ProteinMPNN_design/seqs_forAF2/ \
        --pred_path ${sample_sele_path}/${model}/design_AF2_pred/ \
        --fail_path ${sample_sele_path}/${model}/design_AF2_pred_failed \
        --out_list ${sample_sele_path}/${model}/design_AF2_pred_failed_list.txt

        ls ${sample_sele_path}/${model}/design_AF2_pred/ | wc -l >> ${ProteinMPNN_AF2}

    elif [ -d ${sample_sele_path}/${model}/ProteinMPNN_design/seqs_forAF2/ ]
    then
        mkdir ${sample_sele_path}/${model}/design_AF2_pred

        python AF2_result_check.py \
        --seq_path ${sample_sele_path}/${model}/ProteinMPNN_design/seqs_forAF2/ \
        --pred_path ${sample_sele_path}/${model}/design_AF2_pred/ \
        --fail_path ${sample_sele_path}/${model}/design_AF2_pred_failed \
        --out_list ${sample_sele_path}/${model}/design_AF2_pred_failed_list.txt

        echo "None"  >> ${ProteinMPNN_AF2}

    else
        echo "None"  >> ${ProteinMPNN_AF2}
    fi
done 

echo "Statistics recorded in ${ProteinMPNN_AF2}."
echo "**********************************************************"
