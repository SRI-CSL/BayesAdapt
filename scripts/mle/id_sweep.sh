#dataset@train_dataset=winogrande_xs,winogrande_s,winogrande_m,winogrande_l,ARC-Easy,ARC-Challenge,obqa \
python train_and_evaluate.py --multirun \
    hydra/launcher=ray \
    +hydra.launcher.ray.init.num_gpus=8 \
    +hydra.launcher.ray.remote.num_gpus=1 \
    +lora=default \
    lora.config.r=8 \
    quantize_bits=16 \
    hf_model=Qwen/Qwen3-0.6B,Qwen/Qwen3-1.7B,Qwen/Qwen3-4B,Qwen/Qwen3-8B \
    dataset@train_dataset=winogrande_xs,winogrande_s,winogrande_m,winogrande_l \
    collate_fn=instruct\
    seed=0,1,2,3\
    pbar=False \
    gpu_id=0 #ray will handle CUDA_VISIBLE_DEVICES so we just set gpu_id=0 here
