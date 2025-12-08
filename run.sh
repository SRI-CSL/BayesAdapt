#export CUDA_VISIBLE_DEVICES=1 
#SEED=0

#MODELFAMILY="google"
#HFMODEL="$MODELFAMILY/gemma-3-4b-it"

MODELFAMILY="Qwen"
#MODEL="Qwen3-VL-4B-Instruct"
#HFMODEL="$MODELFAMILY/$MODEL"

#R=8
#LORA_ALPHA=$(($R*2))
#QUANT=16

#CONFIGNAME="train_mle"
CONFIGNAME="train_scalabl"
#CONFIGNAME="train_scalabl"
#WANDBPROJECT="BLoB-Qwen2.5-7B"
#WANDBPROJECT="$CONFIGNAME-Qwen2.5-7B_laplace"
#DATASET="winogrande_s"
#LOGDIR="logs/$WANDBPROJECT-${DATASET}-seed${SEED}-td"
#LOGDIR="logs/Qwen/Qwen2.5-0.5B/scalabl/winogrande_s/seed0/"
#LOGDIR="./gemma-3-test_scalabl/"
#echo $LOGDIR

#for model in "Qwen2.5-0.5B" "Qwen2.5-7B" "Qwen2.5-14B" "Qwen2.5-32B"; do
    #HFMODEL="$MODELFAMILY/$model"
#done

#python train.py --config-name $CONFIGNAME hf_model=$HFMODEL seed=$SEED gpu_id=$GPU_ID\
    #wandb.project=$WANDBPROJECT \
    #wandb.name="$WANDBPROJECT-seed${SEED}-hydra" \
    #logdir=$LOGDIR \
    #dataset.name=$DATASET 

WANDBPROJECT="blora_debug"
GPU_ID=7
SEED=0
#for dataset in winogrande_s ARC-Challenge ARC-Easy winogrande_m obqa boolq MMLU-chem MMLU-phy; do
#


MODELFAMILY="Qwen"
#for dataset in ARC-Challenge ARC-Easy winogrande_m obqa; do
for dataset in winogrande_s; do
    for model_size in "7B"; do
    #for model_size in "8B"; do
    #for model_size in "7B"; do
        #MODEL="Qwen3-$model_size"
        MODEL="Qwen2.5-$model_size"
        #MODEL="Qwen3-VL-8B-Instruct"
        HFMODEL="$MODELFAMILY/$MODEL"
        for instruct in "False"; do
            for quant in 16; do
                LOGDIR="logs/$MODELFAMILY/$MODEL/quant$quant/scalabl/${dataset}/instruct${instruct}/seed${SEED}/"
                echo $LOGDIR

                python train.py --config-name $CONFIGNAME\
                    hf_model=$HFMODEL\
                    seed=$SEED\
                    gpu_id=$GPU_ID\
                    wandb.name=$LOGDIR\
                    logdir=$LOGDIR \
                    quantize_bits=$quant\
                    dataset.name=$dataset \
                    dataset.instruct=$instruct 

                python evaluate.py --config-name $CONFIGNAME\
                    hf_model=$HFMODEL\
                    gpu_id=$GPU_ID \
                    logdir=$LOGDIR \
                    quantize_bits=$quant \
                    dataset.name=$dataset \
                    dataset.instruct=$instruct
            done
        done
    done
done
    


#python evaluate.py --config-name $CONFIGNAME hf_model=$HFMODEL gpu_id=$GPU_ID logdir=$LOGDIR dataset.name=$DATASET quantize_bits=$QUANT

#python convert_dataset.py --config-name $CONFIGNAME hf_model=$HFMODEL seed=$SEED gpu_id=$GPU_ID\
    #wandb.project=$WANDBPROJECT \
    #wandb.name="$WANDBPROJECT-seed${SEED}-hydra" \
    #logdir=$LOGDIR \
    #dataset.name=$DATASET 

