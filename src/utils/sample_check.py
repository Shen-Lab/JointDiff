import sys
import os
import shutil

in_dir = sys.argv[1]
del_flag = (len(sys.argv) > 2 and sys.argv[2] == '1')

if not os.path.isdir(in_dir):
    raise ValueError(f"{in_dir} is not a directory!")

def condition_check(d, token, posi = 'any'):
    if posi == 'any' and token in d:
        return True
    elif posi == 'start' and d.startswith(token):
        return True
    elif posi == 'end' and d.endswith(token):
        return True
    else:
        return False

def dir_stat_check(in_dir, token, posi = 'any', mpnn_flag = False, del_flag = False):
    print(f'******************* {token} ************************')
    dir_list = [
        d for d in os.listdir(in_dir) if 
        os.path.isdir(os.path.join(in_dir, d)) and condition_check(d, token, posi)
    ]
    if dir_list:
        dir_list = sorted(dir_list)

        for d in dir_list:
            tar_path = os.path.join(in_dir, d)

            if mpnn_flag and os.path.join(tar_path, 'seqs'):
                f_num = len(os.listdir(os.path.join(tar_path, 'seqs')))
            elif mpnn_flag:
                f_num = 0
            else:
                f_num = len(os.listdir(tar_path))  

            print(f'{tar_path}: {f_num}')

            if f_num == 0 and del_flag:
                shutil.rmtree(tar_path)


if in_dir.startswith('/'):
    in_dir = in_dir.strip('/')
    in_dir = '/' + in_dir
else:
    in_dir = in_dir.strip('/')
print(in_dir.split('/')[-1])

###### monomer design ######
dir_stat_check(in_dir, token = 'sample', posi = 'start', mpnn_flag = False, del_flag = del_flag)

###### motif-scaffolding ######
dir_stat_check(in_dir, token = 'motif', posi = 'start', mpnn_flag = False, del_flag = del_flag)

###### mpnn pred ######
dir_stat_check(in_dir, token = 'mpnn_pred', posi = 'any', mpnn_flag = True, del_flag = del_flag)

###### esmfold pred ######
dir_stat_check(in_dir, token = 'struc_pred', posi = 'any', mpnn_flag = False, del_flag = del_flag)

print()
