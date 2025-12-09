export CUDA_VISIBLE_DEVICES=1,2,3

python train_and_evaluate.py --multirun \
    hydra/launcher=ray \
    +hydra.launcher.ray.init.num_gpus=3 \
    +hydra.launcher.ray.remote.num_gpus=1 \
    +lora=default \
    hf_model="Qwen/Qwen3-0.6B","Qwen/Qwen3-1.7B","Qwen/Qwen3-4B","Qwen/Qwen3-8B" \
    dataset.name="winogrande_s","winogrande_m","ARC-Easy","ARC-Challenge","obqa" \
    dataset.instruct="True","False" \
    seed=0,1,2 \
    gpu_id=0 #ray will handle CUDA_VISIBLE_DEVICES so we just set gpu_id=0 here
