modelwrapper=svdblob
model=meta-llama/Llama-2-7b-hf
eps=0.05
beta=0.2
kllr=0.1
gamma=8

ori_dataset=obqa

for sample in 10; do
    for seed in 0 1 3; do
        name=$modelwrapper-$ori_dataset-eps$eps-kllr$kllr-beta$beta-gamma$gamma-seed$seed-ood-noU
        CUDA_VISIBLE_DEVICES=$seed python run/main.py --dataset-type mcdataset --dataset $ori_dataset \
            --model-type causallm --model $model --modelwrapper $modelwrapper \
            --lr 1e-4 --batch-size 4 \
            --opt adamw --warmup-ratio 0.06 \
            --max-seq-len 300 \
            --seed $seed \
            --wandb-name $name --wandb-project "SVDBLoB-llama-ood-all-noU" \
            --apply-classhead-lora --lora-r 8 --lora-alpha 16 --lora-dropout 0 \
            --log-path $name \
            --max-train-steps 5000 \
            --eval-per-steps 6000 \
            --bayes-klreweighting \
            --bayes-eps $eps --bayes-beta $beta --bayes-gamma $gamma --bayes-kllr $kllr \
            --bayes-train-n-samples 1 --bayes-eval-n-samples $sample --bayes-eval-n-samples-final $sample \
            --checkpoint --checkpoint-name $name \
            & 
    done
    wait
done

for dataset in ARC-Challenge ARC-Easy MMLU-chem MMLU-phy; do
    for sample in 10; do
        for seed in 0 1 3; do
            name=$modelwrapper-$dataset-sample$sample-eps$eps-kllr$kllr-beta$beta-gamma$gamma-seed$seed-ood-noU
            CUDA_VISIBLE_DEVICES=$seed python run/main.py --dataset-type mcdataset --dataset $dataset \
                --model-type causallm --model $model --modelwrapper $modelwrapper \
                --lr 1e-4 --batch-size 4 \
                --opt adamw --warmup-ratio 0.06 \
                --max-seq-len 300 \
                --seed $seed \
                --wandb-name $name --wandb-project "SVDBLoB-llama-ood-all-noU" \
                --apply-classhead-lora --lora-r 8 --lora-alpha 16 --lora-dropout 0 \
                --log-path $name \
                --max-train-steps 0 \
                --eval-per-steps 6000 \
                --bayes-klreweighting \
                --bayes-eps $eps --bayes-beta $beta --bayes-gamma $gamma --bayes-kllr $kllr --bayes-datasetrescaling \
                --bayes-train-n-samples 1 --bayes-eval-n-samples $sample --bayes-eval-n-samples-final $sample \
                --load-lora-path checkpoints/$modelwrapper/$model/$ori_dataset/$modelwrapper-$ori_dataset-eps$eps-kllr$kllr-beta$beta-gamma$gamma-seed$seed-ood-noU \
                &
        done
        wait
    done
done
