modelwrapper=mle
model=meta-llama/Llama-2-7b-hf
ori_dataset=obqa

for seed in 0 1 3; do
    name=map-$ori_dataset-seed$seed-ood
    CUDA_VISIBLE_DEVICES=$seed python run/main.py --dataset-type mcdataset --dataset $ori_dataset \
        --model-type causallm --model $model --modelwrapper $modelwrapper \
        --lr 1e-4 --batch-size 4 \
        --opt adamw --warmup-ratio 0.06 \
        --max-seq-len 300 \
        --seed $seed \
        --wandb-name $name --wandb-project "MAP-llama7B-all-ood" \
        --apply-classhead-lora --lora-r 8 --lora-alpha 16 --lora-dropout 0 \
        --log-path $name \
        --max-train-steps 5000 \
        --eval-per-steps 6000 \
        --opt-wd 0.01 \
        --checkpoint --checkpoint-name $name \
        &
done
wait

for dataset in ARC-Challenge ARC-Easy MMLU-chem MMLU-phy; do
    for seed in 0 1 3; do
        name=map-$dataset-seed$seed-ood
        CUDA_VISIBLE_DEVICES=$seed python run/main.py --dataset-type mcdataset --dataset $dataset \
            --model-type causallm --model $model --modelwrapper $modelwrapper \
            --lr 1e-4 --batch-size 4 \
            --opt adamw --warmup-ratio 0.06 \
            --max-seq-len 300 \
            --seed $seed \
            --wandb-name $name --wandb-project "MAP-llama7B-all-ood" \
            --apply-classhead-lora --lora-r 8 --lora-alpha 16 --lora-dropout 0 \
            --log-path $name \
            --max-train-steps 0 \
            --eval-per-steps 6000 \
            --opt-wd 0.01 \
            --load-lora-path checkpoints/$modelwrapper/$model/$ori_dataset/map-$ori_dataset-seed$seed-ood \
            &
    done
    wait
done
