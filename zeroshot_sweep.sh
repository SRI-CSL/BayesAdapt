GPU_ID=2
MODELFAMILY="Qwen"
CONFIGNAME="default"

#for dataset in boolq; do #ARC-Challenge ARC-Easy winogrande_m obqa boolq MMLU-chem MMLU-phy; do
for dataset in winogrande_s ARC-Challenge ARC-Easy winogrande_m obqa boolq MMLU-chem MMLU-phy; do
    for model in "Qwen2.5-0.5B" "Qwen2.5-1.5B" "Qwen2.5-3B" "Qwen2.5-7B" "Qwen2.5-14B"; do
    #for model in "Qwen2.5-7B"; do
        HFMODEL="${MODELFAMILY}/${model}"
        LOGDIR="logs/${MODELFAMILY}/${model}/zeroshot/${dataset}/seed0"
        echo $LOGDIR
        python run/evaluate.py --config-name $CONFIGNAME\
            hf_model=$HFMODEL\
            gpu_id=$GPU_ID\
            logdir=$LOGDIR\
            dataset.name=$dataset
    done
done
