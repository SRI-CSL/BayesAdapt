python active_learn.py --multirun \
    +hydra.launcher.ray.init.num_gpus=8 \
    +hydra.launcher.ray.remote.num_gpus=1 \
    +lora=default \
    +lora/wrapper=tempscale \
    trainer=two_stage \
    lora.config.r=8 \
    quantize_bits=16  \
    hf_model=Qwen/Qwen3-VL-8B-Instruct \
    optim.max_train_steps=1000 \
    dataset@train_dataset=srqa \
    collate_fn=vlm\
    seed=0,1,2,3\
    pbar=True \
    gpu_id=0 
