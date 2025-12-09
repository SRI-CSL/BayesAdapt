export CUDA_VISIBLE_DEVICES=1,2,3

python evaluate.py --multirun \
    hydra/launcher=ray \
    +hydra.launcher.ray.init.num_gpus=3 \
    +hydra.launcher.ray.remote.num_gpus=1 \
    hf_model="Qwen/Qwen3-0.6B","Qwen/Qwen3-1.7B","Qwen/Qwen3-4B","Qwen/Qwen3-8B" \
    dataset.name="winogrande_s","winogrande_m","ARC-Easy","ARC-Challenge","obqa","boolq" \
    dataset.instruct="True","False" \
    gpu_id=0 #ray will handle CUDA_VISIBLE_DEVICES so we just set gpu_id=0 here
