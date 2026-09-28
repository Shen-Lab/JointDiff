import os
import sys 

in_path = sys.argv[1]
model = in_path.split('/checkpoints')[0].split('/')[-1]

ckpt_list = [int(p.split('.')[0]) 
    for p in os.listdir(in_path) if p.endswith('.pt') and (not p.startswith('point'))
]

if not ckpt_list:
    print('%s: empty' % (model))
else:
    ckpt_sele = max(ckpt_list)
    print('%s_%d' % (model, ckpt_sele))
