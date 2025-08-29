job_list=$1

while [ $( cat $job_list | wc -l ) -ge 1 ]
do
    echo 'Scanning...'
    job_num=$( squeue -u shaowen1994 | wc -l )
    while [ $job_num -le 390 ]
    do 
        i=1
        for file in $( cat $job_list )
        do 
            echo $file
            sbatch $file
            sed -i '1d' $job_list
            if [ $i -ge 100 ]
            then
                break
            fi
            i=$((i+1)) 
        done
        job_num=$( squeue -u shaowen1994 | wc -l )
    done
    sleep 10m
done
