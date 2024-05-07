rbp=K562_rep6.PUM2
folder=ABC_data/tsv/
datapath=data/$folder
outdir=exp/prismnet/
python -u tools/generate_dataset.py $rbp 1 5 $datapath



python -u tools/main.py \
    --train \
    --eval \
    --lr 0.001 \
    --data_dir $datapath \
    --p_name $rbp\
    --out_dir $outdir \
    --exp_name $rbp \
    --pos_weight 2 \
    --mode seq \
    ${@:5}

# infer
infer_file=$datapath$rbp.tsv
python -u tools/main.py \
    --load_best \
    --infer \
    --infer_file $infer_file \
    --p_name $rbp\
    --out_dir $outdir \
    --exp_name $rbp\
    --mode seq \
    ${@:6}| tee $outdir/out/log.txt


inferred_prob=$outdir/out/infer/${rbp}_PrismNet_seq_$rbp.tsv.probs

# # combind
paste -d $'\t' $infer_file $inferred_prob > $inferred_prob.full
# # HAR: this step is super slow, do only high prob ones


awk -F"\t" '$7>0.99 {print}' $inferred_prob.full | cut -f 1,2,3,4,5,6 > $outdir/$rbp.highprob

# only with this "infer" it does not break, else it predicts odd probabilities
python -u tools/main.py \
    --load_best \
    --har \
    --infer \
    --infer_file $outdir/$rbp.highprob \
    --p_name $rbp\
    --out_dir $outdir \
    --exp_name $rbp \
    --saliency \
    --mode seq \
    ${@:6}| tee $outdir/out/log.txt

# outputs at $outdir/out/har/${rbp}_PrismNet_seq_${rbp}.highprob.har
har=$outdir/out/har/${rbp}_PrismNet_seq_${rbp}.highprob.har
rsync $har hsher@tscc-login2.sdsc.edu:projects/oligo_results/ABC_skipper/Password:
rsync $outdir/$rbp.highprob hsher@tscc-login2.sdsc.edu:projects/oligo_results/ABC_skipper/