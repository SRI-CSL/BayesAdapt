modelwrapper=mcdropout
#model=meta-llama/Llama-2-7b-hf
model=Qwen/Qwen2.5-32B

for dataset in winogrande_s ARC-Challenge ARC-Easy winogrande_m obqa boolq; do
    for sample in 10; do
        for seed in 0 1 3; do
            name=$modelwrapper-$dataset-sample$sample-seed$seed
            CUDA_VISIBLE_DEVICES=2 python run/main.py --dataset-type mcdataset --dataset $dataset \
                --model-type causallm --model $model --modelwrapper $modelwrapper \
                --lr 1e-4 --batch-size 2 \
                --opt adamw --warmup-ratio 0.06 \
                --max-seq-len 300 \
                --seed $seed \
                --wandb-name $name --wandb-project "MCD-qwen32B-all" \
                --apply-classhead-lora --lora-r 8 --lora-alpha 16 --lora-dropout 0.1 \
                --log-path $name \
                --max-train-steps 5000 \
                --eval-per-steps 6000 \
                --bayes-eval-n-samples $sample --bayes-eval-n-samples-final $sample 
        done
    done
done
