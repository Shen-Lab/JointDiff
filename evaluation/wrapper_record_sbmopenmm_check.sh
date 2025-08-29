struc_path='../../Results/originDiff/forward-diff_struc/struc/Step100_posiscale10.0/'
contact_path='../../Data/Processed/CATH_forDiffAb/ContactMap_test_rearanged/'
#result_path='../../Results/originDiff/sbmopenmm_check/'
result_path='../../Results/originDiff/sbmopenmm_check_grace/'
job_list='Job_list_sbmopenmm_check.txt'
#server='Faster'
server='Grace'
job_num=10

if [ -f ${job_list} ]
then
    rm ${job_list}
fi

for with_seq in 0 1
do
    for CA_only in 0 1 
    do
        if [ CA_only != 0 ] || [ with_seq == 1 ]
	then
            out_path="${result_path}/Step100_posiscale10.0_seq-${with_seq}_CA-${CA_only}_withCont_temp-300.0_${job_num}-${job_idx}.pkl"

            for (( job_idx=1; job_idx<=${job_num}; job_idx++ ))
            do

            ./wrapper_submit_sbmopenmm_check.sh \
            --struc_path ${struc_path} \
            --contact_path ${contact_path} \
            --out_path ${out_path} \
            --with_seq ${with_seq} \
            --CA_only ${CA_only} \
            --temperature 300.0 \
            --server ${server} \
            --job_list ${job_list} \
            --job_num ${job_num} \
            --job_idx ${job_idx}
            
            done
        fi
    done
done

### temperature
for (( job_idx=1; job_idx<=${job_num}; job_idx++ ))
do

./wrapper_submit_sbmopenmm_check.sh \
--struc_path ${struc_path} \
--contact_path ${contact_path} \
--out_path "${result_path}/Step100_posiscale10.0_seq-1_CA-0_withCont_temp-1.0_${job_num}-${job_idx}.pkl" \
--with_seq 1 \
--CA_only 1 \
--temperature 1.0 \
--server ${server} \
--job_list ${job_list} \
--job_num ${job_num} \
--job_idx ${job_idx}

done

### no contact
for (( job_idx=1; job_idx<=${job_num}; job_idx++ ))
do

./wrapper_submit_sbmopenmm_check.sh \
--struc_path ${struc_path} \
--contact_path 'None' \
--out_path "${result_path}/Step100_posiscale10.0_seq-1_CA-0_noCont_temp-300.0_${job_num}-${job_idx}.pkl" \
--with_seq 1 \
--CA_only 1 \
--temperature 300.0 \
--server ${server} \
--job_list ${job_list} \
--job_num ${job_num} \
--job_idx ${job_idx}

done
