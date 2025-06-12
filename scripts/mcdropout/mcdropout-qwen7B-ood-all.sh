modelwrapper=mcdropout
model=Qwen/Qwen2.5-7B
ori_dataset=obqa

for sample in 10; do
    for gpuid_seed in "0 0"; do
        read gpuid seed <<< "$gpuid_seed"
        name=$modelwrapper-$ori_dataset-seed$seed-ood
        CUDA_VISIBLE_DEVICES=$gpuid python run/main.py --dataset-type mcdataset --dataset $ori_dataset \
            --model-type causallm --model $model --modelwrapper $modelwrapper \
            --lr 1e-4 --batch-size 4 \
            --opt adamw --warmup-ratio 0.06 \
            --max-seq-len 300 \
            --seed $seed \
            --wandb-name $name --wandb-project "MCD-qwen7B-ood" \
            --apply-classhead-lora --lora-r 8 --lora-alpha 16 --lora-dropout 0.1 \
            --log-path $name \
            --max-train-steps 5000 \
            --eval-per-steps 6000 \
            --bayes-eval-n-samples $sample --bayes-eval-n-samples-final $sample \
            --checkpoint --checkpoint-name $name \
        & 
    done
    wait
done

for dataset in ARC-Challenge ARC-Easy MMLU-chem MMLU-phy; do
    for sample in 10; do
        for gpuid_seed in "0 0"; do
            read gpuid seed <<< "$gpuid_seed"
            name=$modelwrapper-$dataset-seed$seed-ood
            CUDA_VISIBLE_DEVICES=$gpuid python run/main.py --dataset-type mcdataset --dataset $dataset \
                --model-type causallm --model $model --modelwrapper $modelwrapper \
                --lr 1e-4 --batch-size 4 \
                --opt adamw --warmup-ratio 0.06 \
                --max-seq-len 300 \
                --seed $seed \
                --wandb-name $name --wandb-project "MCD-qwen7B-ood" \
                --apply-classhead-lora --lora-r 8 --lora-alpha 16 --lora-dropout 0.1 \
                --log-path $name \
                --max-train-steps 0 \
                --eval-per-steps 6000 \
                --bayes-eval-n-samples $sample --bayes-eval-n-samples-final $sample \
                --load-lora-path checkpoints/$modelwrapper/$model/$ori_dataset/$modelwrapper-$ori_dataset-seed$seed-ood \
                &
        done
        wait
    done
done
