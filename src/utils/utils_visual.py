import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import math

from utils.utils_train import dict_save, dict_load

#################### Loss Visualization #######################################

def loss_read(file_path):
    """Read the loss from the training log."""
    loss_dict = {}
    
    with open(file_path, 'r') as rf:
        lines = rf.readlines()
        
        for line in lines:
            if line[39:46] == '[train]':
                flag = True
                sk = 'train'
            elif line[39:44] == '[val]':
                flag = True
                sk = 'val'
            else:
                flag = False
                
            if flag:
                if not sk in loss_dict.keys():
                    loss_dict[sk] = {'iter':[], 'all':[]}
                
                line = line.strip('\n').split(' | ')
                idx = int(line[0].split(' ')[-1])
                loss_dict[sk]['iter'].append(idx)
                
                for loss_term in line[1:]:
                    if 'loss' in loss_term:
                        loss = float(loss_term.split(' ')[-1])
                        
                        if '(' in loss_term:
                            loss_name = loss_term.split('(')[1].split(')')[0]
                        else:
                            loss_name = 'all'
                            
                        if loss_name not in loss_dict[sk]:
                            loss_dict[sk][loss_name] = [loss]
                        else:
                            loss_dict[sk][loss_name].append(loss)
                
    return loss_dict


def loss_ave(idx_list, loss_list, inter = 1000):
    '''For each <inter> iterations, take the average of the losses.'''
    loss_temp = []
    loss_out = []
    idx_out = []
    for i,l in enumerate(loss_list):
        loss_temp.append(l)
        if len(loss_temp) == inter:
            idx_out.append(idx_list[i])
            loss_out.append(np.mean(loss_temp))
            loss_temp = []
    if len(loss_temp) > 0:
        idx_out.append(idx_list[-1])
        loss_out.append(np.mean(loss_temp))
    return idx_out, loss_out


def loss_plot(loss_dict, inter = 1000, loss_type_list = ['rot', 'pos', 'seq'],
              color_list = ['red', 'blue', 'orange', 'green', 'purple']):
    """Plot out the loss curve."""
    plt.figure(figsize = (15, 4))
    ### overall
    plt.subplot(1,3,1)
    for i,sk in enumerate(['train', 'val']):
        if sk not in loss_dict:
            continue
        
        if sk == 'train':
            idx_list, loss_list = loss_ave(loss_dict[sk]['iter'], loss_dict[sk]['all'], inter = inter)
        else:
            idx_list, loss_list = loss_dict[sk]['iter'], loss_dict[sk]['all']
        plt.plot(idx_list, loss_list, label = sk, color = color_list[i])
    plt.xlabel('Iterations', fontsize = 12)
    plt.ylabel('Loss', fontsize = 12)
    plt.legend(fontsize = 10)
    
    ### separate loss
    for i,sk in enumerate(['train', 'val']):
        if sk not in loss_dict:
            continue
        
        plt.subplot(1,3,i+2)
        for j,version in enumerate(loss_type_list):
            if (version not in loss_dict[sk]) or (len(loss_dict[sk][version]) == 0):
                continue
            
            if sk == 'train':
                idx_list, loss_list = loss_ave(loss_dict[sk]['iter'], loss_dict[sk][version], inter = inter)
            else:
                idx_list, loss_list = loss_dict[sk]['iter'], loss_dict[sk][version]
            plt.plot(idx_list, loss_list, label = version, color = color_list[2 + j])
            
        plt.xlabel('Iterations', fontsize = 12)
        plt.ylabel('Loss', fontsize = 12)
        plt.title(sk, fontsize = 15)
        plt.legend(fontsize = 10)
    plt.show()


def convergent_check(loss_list, iter_list = None):
    """Check whether the training has converged."""
    if len(loss_list) == 0:
        print('Unconverged')
        return None
    
    num = 0
    idx = np.argmin(loss_list)
    loss_min = loss_list[idx]
    if idx == len(loss_list) - 1:
        print('Unconverged')
        return None
    else:
        if iter_list is not None:
            idx = iter_list[idx]
        print('Coverged at Iter %d (%f).'%(idx, loss_min))
        return idx


def loss_wrapper(loss_list, sk, idx, interval):
    """For training set take the mean; otherwise take the point."""
    if sk ==' train':
        return  np.mean(loss_list[idx - interval : idx])
    else:
        return  loss_list[idx]


def model_stat(model_list, version_list, title, log_path, 
               loss_type_list = ['rot', 'pos', 'seq'], iter_sele_dict = None, interval = 1000,
               color_list = ['red', 'blue', 'orange', 'green', 'purple']):
    
    if iter_sele_dict is None:
        iter_sele_dict = {}
    
    for i, ver in enumerate(version_list):
        print('%s = %s'%(title, ver))
        
        model = model_list[i]
        path = os.path.join(log_path, model, 'log.txt')
        
        ###### read the losses ######
        loss_dict = loss_read(path)

        ###### stopping point ######
        if 'val' not in loss_dict: # no validation
            iter_sele = None
        else:
            iter_sele = convergent_check(loss_dict['val']['all'], iter_list = loss_dict['val']['iter'])
        
        if iter_sele is None: # unconverge
            loss_plot(loss_dict, inter = interval)
            continue
            
        ###### save the losses of this model ###### 
        iter_sele_dict[model_list[i]] = {'iter': iter_sele}
        
        for sk in ['train', 'val']:
            ### different datasets
            
            if sk not in loss_dict:
                continue
            
            iter_sele_dict[model_list[i]][sk] = {}
            idx = loss_dict[sk]['iter'].index(iter_sele)
               
            ### different losses
            for loss_type in ['all'] + loss_type_list:
                if loss_type not in loss_dict[sk]:
                    continue
                
                iter_sele_dict[model_list[i]][sk]['loss_%s' % loss_type] = loss_wrapper(
                    loss_dict[sk][loss_type], sk, idx, interval
                )
            
            ### print out the losses
            text = '%s: loss=%.4f;' % (sk, 
                                       iter_sele_dict[model_list[i]][sk]['loss_all'])
            text_overleaf = 'for overleaf: %.3f' % iter_sele_dict[model_list[i]][sk]['loss_all']
                
            for loss_type in loss_type_list:
                if ('loss_%s' % loss_type) not in iter_sele_dict[model_list[i]][sk]:
                    continue
                
                text += ' %s=%.4f;' % (loss_type,
                                       iter_sele_dict[model_list[i]][sk]['loss_%s' % loss_type])
                text_overleaf += ' & %.3f' % iter_sele_dict[model_list[i]][sk]['loss_%s' % loss_type]
                
            text = text.strip(';')
            print(text)
            print(text_overleaf)

        ###### plot the loss ######
        loss_plot(loss_dict, loss_type_list = loss_type_list, inter = interval, color_list = color_list)

    return iter_sele_dict
