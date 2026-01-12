#export CUDA_VISIBLE_DEVICES=1,2,3

#python train.py \
    #+lora=default \
    #+lora/wrapper=mcdropout \
    #lora.config.lora_dropout=0.1 \
    #samples.test.backbone=10 \
    #n_eval_trials=5 \
    #hf_model=Qwen/Qwen3-4B \
    #dataset.name=winogrande_s \
    #dataset.instruct=True \
    #seed=0 \
    #gpu_id=0 #ray will handle CUDA_VISIBLE_DEVICES so we just set gpu_id=0 here

#python evaluate.py \
    #+lora/wrapper=mcdropout \
    #+lora=default \
    #lora.config.lora_dropout=0.1 \
    #samples.test.backbone=10 \
    #n_eval_trials=5 \
    #hf_model=Qwen/Qwen3-4B \
    #dataset.name=winogrande_s \
    #dataset.instruct=True \
    #seed=0 \
    #gpu_id=0 #ray will handle CUDA_VISIBLE_DEVICES so we just set gpu_id=0 here

    #checkpoint=\${logdir}/state_dict.pt \

python train_and_evaluate.py --multirun \
    hydra/launcher=ray \
    +hydra.launcher.ray.init.num_gpus=8 \
    +hydra.launcher.ray.remote.num_gpus=1 \
    +lora=default \
    +lora/wrapper=mcdropout \
    lora.config.lora_dropout=0.1 \
    samples.test.backbone=10 \
    n_eval_trials=5 \
    hf_model=Qwen/Qwen2.5-0.5B,Qwen/Qwen2.5-1.5B,Qwen/Qwen2.5-3B,Qwen/Qwen2.5-7B \
    dataset.name=winogrande_s,winogrande_m,ARC-Easy,ARC-Challenge,obqa \
    dataset.instruct=False \
    seed=0,1,2 \
    pbar=False \
    gpu_id=0 #ray will handle CUDA_VISIBLE_DEVICES so we just set gpu_id=0 here

#python train_and_evaluate.py --multirun \
    #hydra/launcher=ray \
    #+hydra.launcher.ray.init.num_gpus=8 \
    #+hydra.launcher.ray.remote.num_gpus=1 \
    #+lora=default \
    #+lora/wrapper=mcdropout \
    #lora.config.lora_dropout=0.1 \
    #samples.test.backbone=10 \
    #n_eval_trials=5 \
    #hf_model=Qwen/Qwen3-0.6B,Qwen/Qwen3-1.7B,Qwen/Qwen3-4B,Qwen/Qwen3-8B \
    #dataset.name=winogrande_s,winogrande_m,ARC-Easy,ARC-Challenge,obqa \
    #dataset.instruct=True \
    #seed=0,1,2 \
    #pbar=False \
    #gpu_id=0 #ray will handle CUDA_VISIBLE_DEVICES so we just set gpu_id=0 here
