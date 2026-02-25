python train_and_evaluate.py --multirun \
    hydra/launcher=ray \
    +hydra.launcher.ray.init.num_gpus=8 \
    +hydra.launcher.ray.remote.num_gpus=1 \
    +lora=default \
    +lora/wrapper=tempscale \
    hf_model=Qwen/Qwen3-0.6B,Qwen/Qwen3-1.7B,Qwen/Qwen3-4B,Qwen/Qwen3-8B \
    lora.load_mle_checkpoint=True \
    lora.requires_grad=False \
    optim.train_split=validation \
    optim.nll_optimizer.lr=1e-3 \
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
    +lora/wrapper=tempscale \
    hf_model=Qwen/Qwen3-0.6B,Qwen/Qwen3-1.7B,Qwen/Qwen3-4B,Qwen/Qwen3-8B \
    lora.load_mle_checkpoint=True \
    load_pretrained_checkpoint=True \
    dataset@train_dataset=circuit_logic,expression_logic \
    dataset@test_dataset=expression_logic,circuit_logic \
    collate_fn=instruct\
    seed=0,1,2,3 \
    pbar=True \
    gpu_id=0
