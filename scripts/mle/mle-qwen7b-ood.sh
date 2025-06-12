modelwrapper=mle
#model=meta-llama/Llama-2-7b-hf
model=Qwen/Qwen2.5-7B
ori_dataset=obqa

#for seed in 0 1 3; do
#for gpuid_seed in "0 0" "1 1" "2 2" "3 3"; do
for gpuid_seed in "0 4" "1 5" "2 6" "3 7"; do
    read gpuid seed <<< "$gpuid_seed"
    name=$modelwrapper-$ori_dataset-seed$seed-ood
    CUDA_VISIBLE_DEVICES=$gpuid python run/main.py --dataset-type mcdataset --dataset $ori_dataset \
        --model-type causallm --model $model --modelwrapper $modelwrapper \
        --lr 1e-4 --batch-size 4 \
        --opt adamw --warmup-ratio 0.06 \
        --max-seq-len 300 \
        --seed $seed \
        --wandb-name $name --wandb-project "MLE-qwen7B-all-ood" \
        --apply-classhead-lora --lora-r 8 --lora-alpha 16 --lora-dropout 0 \
        --log-path $name \
        --max-train-steps 5000 \
        --eval-per-steps 6000 \
        --checkpoint --checkpoint-name $name \
        &
done
wait


for dataset in ARC-Challenge ARC-Easy MMLU-chem MMLU-phy; do
    for gpuid_seed in "0 4" "1 5" "2 6" "3 7"; do
        read gpuid seed <<< "$gpuid_seed"
        name=$modelwrapper-$dataset-seed$seed-ood
        CUDA_VISIBLE_DEVICES=$gpuid python run/main.py --dataset-type mcdataset --dataset $dataset \
            --model-type causallm --model $model --modelwrapper $modelwrapper \
            --lr 1e-4 --batch-size 4 \
            --opt adamw --warmup-ratio 0.06 \
            --max-seq-len 300 \
            --seed $seed \
            --wandb-name $name --wandb-project "MLE-qwen7B-all-ood" \
            --apply-classhead-lora --lora-r 8 --lora-alpha 16 --lora-dropout 0 \
            --log-path $name \
            --max-train-steps 0 \
            --eval-per-steps 6000 \
            --load-lora-path checkpoints/$modelwrapper/$model/$ori_dataset/$modelwrapper-$ori_dataset-seed$seed-ood \
            &
    done
    wait
done
