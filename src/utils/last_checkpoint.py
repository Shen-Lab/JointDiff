import os
import argparse

parser = argparse.ArgumentParser()

parser.add_argument(
    '--checkpoint_path', 
    type=str, 
    default='../../Logs/logs_autoencoder/autoencoder_mlp-2/'
)
parser.add_argument(
    '--token',
    type=str,
    default=None
)

args = parser.parse_args()

if args.token is None or args.token.upper() == 'NONE':
    args.token = None
    start_token = ''
else:
    start_token = args.token

epoch_list = [ 
    int(e.split('.pt')[0].split(args.token)[-1])
    for e in os.listdir(args.checkpoint_path)
    if e.endswith('.pt') and e.startswith(start_token)
]

if len(epoch_list) == 0:
    print('none')

else:
    max_epoch = max(epoch_list)
    last_cp = '%s%d.pt' % (start_token, max_epoch)
    
    print(os.path.join(args.checkpoint_path, last_cp))

