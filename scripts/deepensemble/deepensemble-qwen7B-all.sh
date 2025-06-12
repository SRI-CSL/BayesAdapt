modelwrapper=deepensemble
model=Qwen/Qwen2.5-7B

for dataset in winogrande_s ARC-Challenge ARC-Easy winogrande_m obqa boolq; do
    #for gpuid_seed in "0 0" "1 1" "2 2" "3 3"; do
    for gpuid_seed in "0 4" "1 5" "2 6" "3 7"; do
        read gpuid seed <<< "$gpuid_seed"
        name=$modelwrapper-$dataset-seed$seed
        CUDA_VISIBLE_DEVICES=$gpuid python run/main.py --dataset-type mcdataset --dataset $dataset \
            --model-type causallm --model $model --modelwrapper $modelwrapper \
            --lr 1e-4 --batch-size 4 \
            --opt adamw --warmup-ratio 0.06 \
            --max-seq-len 300 \
            --seed $seed \
            --wandb-name $name --wandb-project "ENS-Qwen-2.5-7B-all" \
            --apply-classhead-lora --lora-r 8 --lora-alpha 16 --lora-dropout 0 \
            --log-path $name \
            --max-train-steps 5000 \
            --eval-per-steps 6000 \
            --ensemble-n 3 \
            &
    done
    wait
done
