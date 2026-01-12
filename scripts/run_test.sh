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

export CUDA_VISIBLE_DEVICES=1,2,3

python train_and_evaluate.py --multirun \
    hydra/launcher=ray \
    +hydra.launcher.ray.init.num_gpus=3 \
    +hydra.launcher.ray.remote.num_gpus=1 \
    hf_model="Qwen/Qwen2.5-7B" \
    dataset.name="winogrande_s","ARC-Easy","ARC-Challenge","obqa" \
    +lora=default \
    +lora/wrapper=scalabl \
    optim=vi \
    samples.test.backbone=10 \
    n_eval_trials=5 \
    seed=0,1,2 \
    gpu_id=0 #ray will handle CUDA_VISIBLE_DEVICES so we just set gpu_id=0 here

#python evaluate.py \
    #hf_model="Qwen/Qwen2.5-7B" \
    #dataset.name="winogrande_s" \
    #+lora=default \
    #+lora/wrapper=scalabl \
    #optim=vi \
    #seed=0 \
    #samples.test.backbone=10 \
    #n_eval_trials=5 \
    #gpu_id=3 


#python evaluate.py \
    #hf_model="Qwen/Qwen2.5-7B" \
    #dataset.name="winogrande_s" \
    #+lora=default \
    #+lora/wrapper=scalabl \
    #optim=vi \
    #seed=0 \
    #gpu_id=3 \
    #samples.test.backbone=10 \
    #n_eval_trials=5 \
    #logdir=$LOGDIR 


#optim.kl_optimizer.lr=0.2 \
#python convert_dataset.py --config-name $CONFIGNAME hf_model=$HFMODEL seed=$SEED gpu_id=$GPU_ID\
    #wandb.project=$WANDBPROJECT \
    #wandb.name="$WANDBPROJECT-seed${SEED}-hydra" \
    #logdir=$LOGDIR \
    #dataset.name=$DATASET 

