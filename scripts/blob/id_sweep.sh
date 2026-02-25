python train_and_evaluate.py --multirun \
    hydra/launcher=ray \
    +hydra.launcher.ray.init.num_gpus=8 \
    +hydra.launcher.ray.remote.num_gpus=1 \
    +lora=default \
    +lora/wrapper=blob \
    optim=vi \
    trainer=vi \
    optim.kl_optimizer.lr=0.01 \
    samples.test.backbone=10 \
    hf_model=Qwen/Qwen3-0.6B,Qwen/Qwen3-1.7B,Qwen/Qwen3-4B,Qwen/Qwen3-8B,Qwen/Qwen3-14B \
    dataset@train_dataset=winogrande_xs,winogrande_s,winogrande_m,winogrande_l,ARC-Easy,ARC-Challenge,obqa \
    collate_fn=instruct\
    seed=0,1,2,3 \
    pbar=True \
    gpu_id=0 
