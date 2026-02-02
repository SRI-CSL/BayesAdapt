
python train_and_evaluate.py --multirun \
    hydra/launcher=ray \
    +hydra.launcher.ray.init.num_gpus=8 \
    +hydra.launcher.ray.remote.num_gpus=1 \
    +lora=default \
    +lora/wrapper=tfb \
    optim=binary_search \
    trainer=binary_search \
    lora.config.r=8 \
    optim.max_train_steps=5 \
    samples.test.backbone=10 \
    lora.load_mle_checkpoint=True \
    n_eval_trials=1 \
    hf_model=Qwen/Qwen3-0.6B,Qwen/Qwen3-1.7B,Qwen/Qwen3-4B,Qwen/Qwen3-8B,Qwen/Qwen3-14B\
    dataset@train_dataset=winogrande_xs,winogrande_s,winogrande_m,winogrande_l \
    collate_fn=instruct \
    pbar=True  \
    seed=0,1,2,3\
    gpu_id=0 #ray will handle CUDA_VISIBLE_DEVICES so we just set gpu_id=0 here
