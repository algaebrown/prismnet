rbp=441_PUM_K562
folder=rsr_data
datapath=data/$folder
outdir=exp/prismnet/
python -u tools/generate_dataset.py $rbp 1 5 $datapath

exp/prismnet/train.sh $rbp $folder

#exp/prismnet/infer.sh $rbp $datapath/$rbp.tsv

folder=evan_data
datapath=data/$folder

for rbp in 441_PUM2_K562_subset 441_PUM2_K562_subset_2193 441_PUM2_K562_subset_4386 441_PUM2_K562_subset_8772; 
do 
python -u tools/generate_dataset.py $rbp 1 5 $datapath; 
#exp/prismnet/train.sh $rbp(p) $folder(d); 

# https://stackoverflow.com/questions/57021620/how-to-calculate-unbalanced-weights-for-bcewithlogitsloss-in-pytorch
# recomment the weight to be the ratio pos/neg
python -u tools/main.py \
    --train \
    --eval \
    --lr 0.001 \
    --data_dir data/$folder \
    --p_name $rbp\
    --out_dir $outdir \
    --exp_name $exp \
    --pos_weight 2 \
    ${@:5}
done

