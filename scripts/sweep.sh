GPU_ID=2
MODELFAMILY="Qwen"
WRAPPER=$1
CONFIGNAME="train_${WRAPPER}"

for dataset in winogrande_s ARC-Challenge ARC-Easy winogrande_m obqa boolq; do
    for model in "Qwen2.5-0.5B" "Qwen2.5-1.5B" "Qwen2.5-3B" "Qwen2.5-7B" "Qwen2.5-14B"; do
        HFMODEL="${MODELFAMILY}/${model}"
        WANDBPROJECT="${model}-${WRAPPER}-${dataset}"
        for seed in 0 3; do
            LOGDIR="logs/${MODELFAMILY}/${model}/${WRAPPER}/${dataset}/seed${seed}"
            WANDBNAME="${model}-${WRAPPER}-${dataset}-seed${seed}"
            echo $LOGDIR
            
            python run/train.py --config-name $CONFIGNAME\
                hf_model=$HFMODEL\
                gpu_id=$seed\
                logdir=$LOGDIR\
                dataset.name=$dataset\
                wandb.project=$WANDBPROJECT\
                wandb.name=$WANDBNAME\
                seed=$seed &
        done
        wait
        for seed in 0 3; do
            LOGDIR="logs/${MODELFAMILY}/${model}/${WRAPPER}/${dataset}/seed${seed}"
            echo $LOGDIR

            python run/evaluate.py --config-name $CONFIGNAME\
                hf_model=$HFMODEL\
                gpu_id=$seed\
                logdir=$LOGDIR\
                dataset.name=$dataset\
                seed=$seed &
        done
        wait
    done
done
