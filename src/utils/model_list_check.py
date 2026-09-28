import sys
import os

tar = sys.argv[1]

ref_dir = 'ModelLists/'
ref_list = [f for f in os.listdir(ref_dir)
    if f.startswith('jointdiff')
]

ref_set = set()
for f in ref_list:
    with open(os.path.join(ref_dir, f), 'r') as rf:
        ref_set = ref_set.union(set(rf.readlines()))

with open(tar, 'r') as rf:
    for line in rf:
        if line not in ref_set:
            print(line.strip('\n'))
