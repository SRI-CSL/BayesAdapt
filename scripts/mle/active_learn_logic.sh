python active_learn.py --multirun \
    hydra/launcher=ray \
    +hydra.launcher.ray.init.num_gpus=8 \
    +hydra.launcher.ray.remote.num_gpus=1 \
    +lora=default \
    lora.config.r=8 \
    quantize_bits=16  \
    hf_model=Qwen/Qwen3-0.6B \
    optim.max_train_steps=1000 \
    dataset@train_dataset=expression_logic \
    collate_fn=instruct \
    seed=0,1,2,3\
    pbar=True \
    overwrite=True \
    gpu_id=0 #ray will handle CUDA_VISIBLE_DEVICES so we just set gpu_id=0 here
