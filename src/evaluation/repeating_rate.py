import numpy as np
import pickle
import argparse

def repeat_score(seq, k):
    """ratio of the repeating regions (>=k) in the sequence."""
    if not seq:
        return 0

    char_pre = None
    count = 0
    count_max = 0
    rep_region = 0
    for char in seq + '!':
        if char == char_pre:
            count += 1
        else:
            if count >= k:
                rep_region += count
            count = 1
      
        char_pre = char
        count_max = max(count_max, count)

    return rep_region / len(seq), count_max / len(seq)


####################################### main function #######################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in_path', type=str, default='seqs/Nature_seq.500.fa')
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--threshold', type=int, default=0.)

    args = parser.parse_args()

    score_list = []
    mr_list = []
    num = 0

    with open(args.in_path, 'r') as rf:
        for line in rf:
            if line.startswith('>'):
                continue
            seq = line.strip('\n')
            score, max_repeat = repeat_score(seq, args.k)
            score_list.append(score)
            mr_list.append(max_repeat)
            if score > args.threshold:
                num += 1

    if score_list:
        print('Repeating rate: %f; %.2f%% > %f' % (
            np.mean(score_list), float(num) / len(score_list) * 100, args.threshold
        ))
        print('Longest Repeating rate:', np.mean(mr_list))
    else:
        print('Empty input!')



