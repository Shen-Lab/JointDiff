database='../../../Tools/Foldseek/foldseek/database/pdb/pdb'

in_path=$1
out_path=$2

if [ -f ${out_path} ]
then
    echo "Warning! ${out_path} exists!"
    rm ${out_path}
fi

if [ -d foldseek_tmp ]
then
    echo "Warning! foldseek_tmp exists!"
else
    mkdir foldseek_tmp
fi

for pdb in ${in_path}/*.pdb
do
    name=${pdb##*/}
    foldseek easy-search ${pdb} ${database} foldseek_tmp/${name}.txt tmp
    cat foldseek_tmp/${name}.txt >> ${out_path}
    rm foldseek_tmp/${name}.txt
done

#rm -r foldseek_tmp
