#export CUDA_VISIBLE_DEVICES=1,2,3

#optim=vi \
python evaluate.py --multirun \
    hydra/launcher=ray \
    +hydra.launcher.ray.init.num_gpus=8 \
    +hydra.launcher.ray.remote.num_gpus=1 \
    +lora=default \
    +lora/wrapper=tfb \
    optim=binary_search \
    trainer=binary_search \
    optim.max_train_steps=5 \
    samples.test.backbone=10 \
    lora.load_mle_checkpoint=True \
    n_eval_trials=1 \
    hf_model=Qwen/Qwen3-0.6B,Qwen/Qwen3-1.7B,Qwen/Qwen3-4B,Qwen/Qwen3-8B \
    dataset@train_dataset=winogrande_s\
    collate_fn=instruct \
    pbar=True  \
    seed=0,1,2,3 \
    gpu_id=0 #ray will handle CUDA_VISIBLE_DEVICES so we just set gpu_id=0 here

#python evaluate.py \
    #+lora=default \
    #+lora/wrapper=scalabl \
    #optim=vi \
    #samples.test.backbone=10 \
    #n_eval_trials=5 \
    #hf_model=Qwen/Qwen2.5-7B \
    #dataset.name=winogrande_s \
    #dataset.instruct=False \
    #checkpoint=\${logdir}/state_dict.pt \
    #seed=0 \
    #gpu_id=0 #ray will handle CUDA_VISIBLE_DEVICES so we just set gpu_id=0 here

#qwen2.5
#python train_and_evaluate.py --multirun \
    #hydra/launcher=ray \
    #+hydra.launcher.ray.init.num_gpus=8 \
    #+hydra.launcher.ray.remote.num_gpus=1 \
    #+lora=default \
    #+lora/wrapper=scalabl \
    #optim=vi \
    #samples.test.backbone=10 \
    #n_eval_trials=5 \
    #hf_model=Qwen/Qwen2.5-0.5B,Qwen/Qwen2.5-1.5B,Qwen/Qwen2.5-3B,Qwen/Qwen2.5-7B \
    #dataset.name=winogrande_s,winogrande_m,ARC-Easy,ARC-Challenge,obqa \
    #dataset.instruct=False \
    #seed=0,1,2 \
    #gpu_id=0 #ray will handle CUDA_VISIBLE_DEVICES so we just set gpu_id=0 here

#qwen3
#python train_and_evaluate.py --multirun \
    #hydra/launcher=ray \
    #+hydra.launcher.ray.init.num_gpus=8 \
    #+hydra.launcher.ray.remote.num_gpus=1 \
    #+lora=default \
    #+lora/wrapper=scalabl \
    #optim=vi \
    #samples.test.backbone=10 \
    #n_eval_trials=5 \
    #hf_model=Qwen/Qwen3-0.6B,Qwen/Qwen3-1.7B,Qwen/Qwen3-4B,Qwen/Qwen3-8B \
    #dataset.name=winogrande_s,winogrande_m,ARC-Easy,ARC-Challenge,obqa \
    #dataset.instruct=True \
    #seed=0,1,2 \
    #gpu_id=0 #ray will handle CUDA_VISIBLE_DEVICES so we just set gpu_id=0 here
