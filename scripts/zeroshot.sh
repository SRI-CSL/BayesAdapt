#export CUDA_VISIBLE_DEVICES=1,2,3

python evaluate.py \
    hf_model=Qwen/Qwen3-VL-8B-Instruct \
    collate_fn=vlm\
    dataset@train_dataset=MNIST\
    gpu_id=1 #ray will handle CUDA_VISIBLE_DEVICES so we just set gpu_id=0 here


#python evaluate.py --multirun \
    #hydra/launcher=ray \
    #+hydra.launcher.ray.init.num_gpus=8 \
    #+hydra.launcher.ray.remote.num_gpus=1 \
    #hf_model=Qwen/Qwen2.5-0.5B,Qwen/Qwen2.5-1.5B,Qwen/Qwen2.5-3B,Qwen/Qwen2.5-7B \
    #dataset.name=winogrande_s,winogrande_m,ARC-Easy,ARC-Challenge,obqa \
    #dataset.instruct=False \
    #gpu_id=0 #ray will handle CUDA_VISIBLE_DEVICES so we just set gpu_id=0 here


#python evaluate.py --multirun \
    #hydra/launcher=ray \
    #+hydra.launcher.ray.init.num_gpus=8 \
    #+hydra.launcher.ray.remote.num_gpus=1 \
    #hf_model=Qwen/Qwen3-0.6B,Qwen/Qwen3-1.7B,Qwen/Qwen3-4B,Qwen/Qwen3-8B \
    #dataset.name=winogrande_s,winogrande_m,ARC-Easy,ARC-Challenge,obqa \
    #dataset.instruct=True \
    #gpu_id=0 #ray will handle CUDA_VISIBLE_DEVICES so we just set gpu_id=0 here
