python train_and_evaluate.py --multirun \
    hydra/launcher=ray \
    +hydra.launcher.ray.init.num_gpus=8 \
    +hydra.launcher.ray.remote.num_gpus=1 \
    +lora=default \
    +lora/wrapper=deepensemble \
    lora.wrapper.ensemble_size=10 \
    samples.train.backbone=10 \
    samples.test.backbone=10 \
    hf_model=Qwen/Qwen3-0.6B,Qwen/Qwen3-1.7B,Qwen/Qwen3-4B,Qwen/Qwen3-8B \
    optim.max_train_steps=5000 \
    dataset@train_dataset=circuit_logic,expression_logic \
    collate_fn=instruct\
    seed=0,1,2,3 \
    pbar=True \
    gpu_id=0


python evaluate.py --multirun \
    hydra/launcher=ray \
    +hydra.launcher.ray.init.num_gpus=8 \
    +hydra.launcher.ray.remote.num_gpus=1 \
    +lora=default \
    +lora/wrapper=deepensemble \
    hf_model=Qwen/Qwen3-0.6B,Qwen/Qwen3-1.7B,Qwen/Qwen3-4B,Qwen/Qwen3-8B \
    load_pretrained_checkpoint=True \
    lora.wrapper.ensemble_size=10 \
    samples.train.backbone=10\
    samples.test.backbone=10\
    dataset@train_dataset=circuit_logic,expression_logic \
    dataset@test_dataset=expression_logic,circuit_logic \
    collate_fn=instruct\
    seed=0,1,2,3 \
    pbar=True \
    gpu_id=0
