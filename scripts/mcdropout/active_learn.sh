python active_learn.py --multirun \
    hydra/launcher=ray \
    +hydra.launcher.ray.init.num_gpus=8 \
    +hydra.launcher.ray.remote.num_gpus=1 \
    +lora=default \
    +lora/wrapper=mcdropout \
    lora.config.r=8 \
    lora.config.lora_dropout=0.1 \
    quantize_bits=16  \
    samples.test.backbone=10 \
    n_eval_trials=1 \
    hf_model=Qwen/Qwen3-VL-8B-Instruct \
    optim.max_train_steps=1000 \
    dataset@train_dataset=srqa \
    collate_fn=vlm\
    seed=0,1,2,3\
    pbar=True \
    overwrite=True \
    gpu_id=0 #ray will handle CUDA_VISIBLE_DEVICES so we just set gpu_id=0 here
