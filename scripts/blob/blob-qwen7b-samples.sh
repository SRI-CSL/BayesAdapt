modelwrapper=blob
model=Qwen/Qwen2.5-7B
eps=0.05
beta=0.2
kllr=0.01
gamma=8

ori_dataset=winogrande_s

#for dataset in winogrande_s; do
#for dataset in winogrande_s; do
#for dataset in winogrande_s ARC-Challenge ARC-Easy winogrande_m obqa boolq; do
for dataset in winogrande_s; do
    for sample in 10; do
        for seed in 0 1 3; do
            name=$modelwrapper-$dataset-sample$sample-eps$eps-kllr$kllr-beta$beta-gamma$gamma-seed$seed-samples 
            CUDA_VISIBLE_DEVICES=$seed python run/main.py --dataset-type mcdataset --dataset $dataset \
                --model-type causallm --model $model --modelwrapper $modelwrapper \
                --lr 1e-4 --batch-size 4 \
                --opt adamw --warmup-ratio 0.06 \
                --max-seq-len 300 \
                --seed $seed \
                --evaluate \
                --wandb-name $name --wandb-project "BLoB-Qwen7B-samples" \
                --apply-classhead-lora --lora-r 8 --lora-alpha 16 --lora-dropout 0 \
                --log-path $name \
                --max-train-steps 5000 \
                --eval-per-steps 6000 \
                --bayes-klreweighting \
                --bayes-eps $eps --bayes-beta $beta --bayes-gamma $gamma --bayes-kllr $kllr --bayes-datasetrescaling \
                --bayes-train-n-samples 1 --bayes-eval-n-samples 1 --bayes-eval-n-samples-final $sample  \
                --checkpoint --checkpoint-name $name \
                &
        done 
        wait
    done
done

for dataset in winogrande_s; do
    for sample in 1 3 5 7 9 11 13 15 17 20; do
        for seed in 0 1 3; do
            name=$modelwrapper-$dataset-sample$sample-eps$eps-kllr$kllr-beta$beta-gamma$gamma-seed$seed-samples$sample
            CUDA_VISIBLE_DEVICES=$seed python run/main.py --dataset-type mcdataset --dataset $dataset \
                --model-type causallm --model $model --modelwrapper $modelwrapper \
                --lr 1e-4 --batch-size 4 \
                --opt adamw --warmup-ratio 0.06 \
                --max-seq-len 300 \
                --seed $seed \
                --wandb-name $name --wandb-project "BLoB-Qwen7B-samples" \
                --apply-classhead-lora --lora-r 8 --lora-alpha 16 --lora-dropout 0 \
                --log-path $name \
                --max-train-steps 0 \
                --eval-per-steps 6000 \
                --bayes-klreweighting \
                --bayes-eps $eps --bayes-beta $beta --bayes-gamma $gamma --bayes-kllr $kllr --bayes-datasetrescaling \
                --bayes-train-n-samples 1 --bayes-eval-n-samples 1 --bayes-eval-n-samples-final $sample  \
                --load-lora-path checkpoints/$modelwrapper/$model/$ori_dataset/$modelwrapper-$ori_dataset-sample10-eps$eps-kllr$kllr-beta$beta-gamma$gamma-seed$seed-samples
        done 
    done
done

