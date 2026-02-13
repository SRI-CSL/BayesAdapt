python train_and_evaluate.py --multirun \
    hydra/launcher=ray \
    +hydra.launcher.ray.init.num_gpus=8 \
    +hydra.launcher.ray.remote.num_gpus=1 \
    +lora=default \
    +lora/wrapper=tempscale \
    lora.load_mle_checkpoint=True \
    lora.requires_grad=False \
    optim.train_split=validation \
    optim.nll_optimizer.lr=1e-3 \
    lora.config.r=8 \
    hf_model=Qwen/Qwen3-14B \
    dataset@train_dataset=winogrande_xs,winogrande_s,winogrande_m,winogrande_l,ARC-Easy,ARC-Challenge,obqa \
    collate_fn=instruct\
    seed=0,1,2,3 \
    overwrite=True \
    pbar=False \
    gpu_id=0 #ray will handle CUDA_VISIBLE_DEVICES so we just set gpu_id=0 here
