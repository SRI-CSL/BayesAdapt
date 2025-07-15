modelwrapper=scalabl
model=meta-llama/Llama-2-7b-hf
eps=0.05
beta=0.2
kllr=0.1
gamma=8
lora_r=8
lora_alpha=$(echo $lora_r*2 | bc)

for dataset in winogrande_s ARC-Challenge ARC-Easy winogrande_m obqa boolq; do
    for sample in 10; do
        for gpuid_seed in "0 0"; do
            read gpuid seed <<< "$gpuid_seed"
            name=$modelwrapper-$dataset-sample$sample-eps$eps-kllr$kllr-beta$beta-gamma$gamma-seed$seed-id
            CUDA_VISIBLE_DEVICES=$gpuid python run/main.py --dataset-type mcdataset --dataset $dataset \
                --model-type causallm --model $model --modelwrapper $modelwrapper \
                --lr 1e-4 --batch-size 4 \
                --opt adamw --warmup-ratio 0.06 \
                --max-seq-len 300 \
                --seed $seed \
                --evaluate \
                --wandb-name $name --wandb-project "ScalaBL-llama7B-id" \
                --apply-classhead-lora --lora-r $lora_r --lora-alpha $lora_alpha --lora-dropout 0 \
                --log-path $name \
                --max-train-steps 5000 \
                --eval-per-steps 6000 \
                --bayes-klreweighting \
                --bayes-eps $eps --bayes-beta $beta --bayes-gamma $gamma --bayes-kllr $kllr --bayes-datasetrescaling \
                --bayes-train-n-samples 1 --bayes-eval-n-samples 1 --bayes-eval-n-samples-final $sample  
        done 
    done
done
