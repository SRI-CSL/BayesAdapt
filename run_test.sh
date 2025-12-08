export BNB_CUDA_VERSION=125
SEED=0
GPU_ID=0
MODELFAMILY="Qwen"
HFMODEL="$MODELFAMILY/Qwen2.5-0.5B"
#HFMODEL="$MODELFAMILY/gemma-3-12b-it"
#CONFIGNAME="train_mle"
CONFIGNAME="train_mle"
#WANDBPROJECT="BLoB-Qwen2.5-7B"
WANDBPROJECT="$CONFIGNAME-Qwen2.5-7B_laplace"
DATASET="winogrande_s"
#LOGDIR="logs/$WANDBPROJECT-${DATASET}-seed${SEED}-td"
#LOGDIR="logs/Qwen/Qwen2.5-0.5B/scalabl/winogrande_s/seed0/"
#LOGDIR="./mle_log_test/"
LOGDIR="./gemma-3-test/"
echo $LOGDIR

#for model in "Qwen2.5-0.5B" "Qwen2.5-7B" "Qwen2.5-14B" "Qwen2.5-32B"; do
    #HFMODEL="$MODELFAMILY/$model"
#done

#python train.py --config-name $CONFIGNAME hf_model=$HFMODEL seed=$SEED gpu_id=$GPU_ID\
    #wandb.name="$WANDBPROJECT-seed${SEED}-hydra" \
    #logdir=$LOGDIR \
    #dataset.name=$DATASET 

#python evaluate.py --config-name $CONFIGNAME hf_model=$HFMODEL seed=$SEED gpu_id=$GPU_ID logdir=$LOGDIR dataset.name=$DATASET 

python train.py \
    hf_model="Qwen/Qwen2.5-0.5B" \
    dataset.name="winogrande_s" \
    +lora=default \
    +lora/wrapper=scalabl \
    optim=vi \
    seed=0 \
    logdir=$LOGDIR 

#optim.kl_optimizer.lr=0.2 \
#python convert_dataset.py --config-name $CONFIGNAME hf_model=$HFMODEL seed=$SEED gpu_id=$GPU_ID\
    #wandb.project=$WANDBPROJECT \
    #wandb.name="$WANDBPROJECT-seed${SEED}-hydra" \
    #logdir=$LOGDIR \
    #dataset.name=$DATASET 

