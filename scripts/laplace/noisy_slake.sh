python evaluate.py --multirun \
    hydra/launcher=ray \
    +hydra.launcher.ray.init.num_gpus=8 \
    +hydra.launcher.ray.remote.num_gpus=1 \
    +lora=default \
    hf_model=Qwen/Qwen3-VL-8B-Instruct \
    optim=laplace \
    trainer=laplace \
    n_eval_trials=10 \
    lora.load_mle_checkpoint=True \
    dataset@train_dataset=slake \
    dataset@test_dataset=noisy_slake1,noisy_slake2,noisy_slake4,noisy_slake8,noisy_slake16,noisy_slake32,noisy_slake64,noisy_slake128 \
    collate_fn=vlm\
    seed=0,1,2,3\
    pbar=True \
    gpu_id=0 
