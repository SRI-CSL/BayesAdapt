#export CUDA_VISIBLE_DEVICES=1 
SEED=0
GPU_ID=2
MODELFAMILY="google"
#HFMODEL="$MODELFAMILY/Qwen2.5-7B"
HFMODEL="$MODELFAMILY/gemma-3-12b-it"
#CONFIGNAME="train_mle"
CONFIGNAME="default"
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
    #wandb.project=$WANDBPROJECT \
    #wandb.name="$WANDBPROJECT-seed${SEED}-hydra" \
    #logdir=$LOGDIR \
    #dataset.name=$DATASET 

python evaluate.py --config-name $CONFIGNAME hf_model=$HFMODEL seed=$SEED gpu_id=$GPU_ID logdir=$LOGDIR dataset.name=$DATASET 

#python convert_dataset.py --config-name $CONFIGNAME hf_model=$HFMODEL seed=$SEED gpu_id=$GPU_ID\
    #wandb.project=$WANDBPROJECT \
    #wandb.name="$WANDBPROJECT-seed${SEED}-hydra" \
    #logdir=$LOGDIR \
    #dataset.name=$DATASET 

