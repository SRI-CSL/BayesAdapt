export CUDA_VISIBLE_DEVICES=1 
SEED=0
GPU_ID=0
MODEL="Qwen/Qwen2.5-7B"
CONFIGNAME="train_blob"
WANDBPROJECT="BLoB-Qwen2.5-7B"
LOGDIR="logs/$WANDBPROJECT-seed${SEED}-id-hydra"
echo $LOGDIR

python run/train.py --config-name $CONFIGNAME model=$MODEL seed=$SEED gpu_id=$GPU_ID\
    wandb.project=$WANDBPROJECT \
    wandb.name="$WANDBPROJECT-seed${SEED}-hydra" \
    logdir=$LOGDIR \
    optim.max_train_steps=5000 \
    dataset.name=boolq 

python run/eval.py --config-name $CONFIGNAME model=$MODEL seed=$SEED gpu_id=$GPU_ID logdir=$LOGDIR dataset.name=boolq
