python train_and_evaluate.py --multirun \
    hydra/launcher=ray \
    +hydra.launcher.ray.init.num_gpus=8 \
    +hydra.launcher.ray.remote.num_gpus=1 \
    +lora=default \
    +lora/wrapper=deepensemble \
    lora.config.r=8 \
    quantize_bits=16  \
    lora.wrapper.ensemble_size=5 \
    samples.train.backbone=5 \
    samples.test.backbone=5 \
    hf_model=Qwen/Qwen3-0.6B,Qwen/Qwen3-1.7B,Qwen/Qwen3-4B \
    optim.max_train_steps=5000 \
    dataset@train_dataset=circuit_logic,expression_logic \
    collate_fn=instruct\
    seed=0,1,2,3 \
    pbar=True \
    overwrite=False \
    use_wandb=False \
    gpu_id=0


python evaluate.py --multirun \
    hydra/launcher=ray \
    +hydra.launcher.ray.init.num_gpus=8 \
    +hydra.launcher.ray.remote.num_gpus=1 \
    +lora=default \
    +lora/wrapper=deepensemble \
    lora.config.r=8 \
    quantize_bits=16  \
    hf_model=Qwen/Qwen3-0.6B,Qwen/Qwen3-1.7B,Qwen/Qwen3-4B \
    load_pretrained_checkpoint=True \
    optim.max_train_steps=5000 \
    lora.wrapper.ensemble_size=5 \
    samples.train.backbone=5 \
    samples.test.backbone=5 \
    dataset@train_dataset=circuit_logic,expression_logic \
    dataset@test_dataset=expression_logic,circuit_logic \
    collate_fn=instruct\
    seed=0,1,2,3 \
    pbar=True \
    overwrite=False \
    gpu_id=0
