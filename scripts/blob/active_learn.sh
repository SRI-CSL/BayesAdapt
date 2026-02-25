python active_learn.py --multirun \
    hydra/launcher=ray \
    +hydra.launcher.ray.init.num_gpus=8 \
    +hydra.launcher.ray.remote.num_gpus=1 \
    +lora=default \
    +lora/wrapper=blob \
    optim=vi \
    trainer=vi \
    optim.kl_optimizer.lr=0.01 \
    samples.test.backbone=10 \
    hf_model=Qwen/Qwen3-VL-8B-Instruct \
    optim.max_train_steps=1000 \
    dataset@train_dataset=srqa \
    collate_fn=vlm\
    seed=0,1,2,3\
    pbar=True \
    gpu_id=0 
