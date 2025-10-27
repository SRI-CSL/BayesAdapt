modelwrapper=mle
model=Qwen/Qwen3-14B-Base
#model=Qwen/Qwen3-32B
lora_r=8
lora_alpha=$(echo $lora_r*2 | bc)

#for dataset in winogrande_s; do
#for dataset in winogrande_s; do
#for dataset in winogrande_s ARC-Challenge ARC-Easy winogrande_m obqa boolq; do
for dataset in winogrande_s; do
    for gpuid_seed in "2 0"; do
        read gpuid seed <<< "$gpuid_seed"
        name=$modelwrapper-$dataset-seed$seed
        CUDA_VISIBLE_DEVICES=$gpuid python run/main.py --dataset-type mcdataset --dataset $dataset \
            --model-type causallm --model $model --modelwrapper $modelwrapper \
            --lr 1e-4 --batch-size 4 \
            --opt adamw --warmup-ratio 0.06 \
            --max-seq-len 300 \
            --seed $seed \
            --evaluate \
            --wandb-name $name --wandb-project "zeroshot-Qwen7B-id" \
            --apply-classhead-lora --lora-r $lora_r --lora-alpha $lora_alpha --lora-dropout 0.0 \
            --log-path $name \
            --max-train-steps 0 \
            --eval-per-steps 6000
    done 
done
