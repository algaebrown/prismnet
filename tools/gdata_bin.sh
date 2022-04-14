#!/bin/bash
d=clip_data
for p in `cat data/${d}/all.list`
do 
    python -u tools/generate_dataset.py $p 1 5 data/$d
    # 1: name, 2: is_bin, 3: in_ver, 4:data_path
    # is_bin = 1, binarize, anyscore < 0: 0, else 1
    # is_bin = 0, rescale somehow
    # there is a max_len of 101
done
