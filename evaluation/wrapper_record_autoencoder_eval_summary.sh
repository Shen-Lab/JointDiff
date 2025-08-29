version_list='../Model_Lists/version_list_autoencoder_eval.txt'
out_path='../../Results/autoencoder/ModalityRecovery/summary.txt'

archi_flag=0
version_flag=0

for line in $( cat ${version_list} )
do
   if [[ $line == *"######"* ]] 
   then
       if [ ${archi_flag} == 1 ]
       then
           echo ''
       fi

       archi_flag=1
       version_flag=0
       echo '##############################################'
       echo "# ${line##*#}"
       echo '##############################################'
       echo ''

   elif [[ $line == *"###"* ]] 
   then
       # if [ ${version_flag} == 1 ]
       # then
       #     echo '..............................................'
       #     echo ''
       # fi

       echo '..............................................'
       version_flag=1
       echo ${line##*#}
       echo '..............................................'
       echo ''

   else
       echo $line
       echo '**********************************************'
       path="Output/output_autoencoderEval_${line}"

       length=$( cat ${path} | wc -l )
       if [ ${length} -gt 16 ]
       then
           tail -15 $path | head -14
       else
           tail -9 $path | head -8
       fi

       echo '**********************************************'
       echo ''
   fi 

done #> ${out_path}
