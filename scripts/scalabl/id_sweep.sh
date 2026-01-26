#export CUDA_VISIBLE_DEVICES=1,2,3
export CUDA_VISIBLE_DEVICES=4,5,6,7

#python evaluate.py \
    #hf_model=Qwen/Qwen2.5-7B \
    #dataset.name=winogrande_s \
    #dataset.instruct=False \
    #seed=0 \
    #pbar=True \
    #gpu_id=0 #ray will handle CUDA_VISIBLE_DEVICES so we just set gpu_id=0 here

#python train_and_evaluate.py \
    #+lora=default \
    #hf_model=Qwen/Qwen2.5-7B \
    #dataset@train_dataset=winogrande_s\
    #collate_fn=base\
    #seed=0 \
    #pbar=True \
    #gpu_id=0 #ray will handle CUDA_VISIBLE_DEVICES so we just set gpu_id=0 here

python train_and_evaluate.py --multirun \
    hydra/launcher=ray \
    +hydra.launcher.ray.init.num_gpus=4 \
    +hydra.launcher.ray.remote.num_gpus=1 \
    +lora=default \
    +lora/wrapper=scalabl \
    optim=vi \
    trainer=vi \
    samples.test.backbone=10 \
    n_eval_trials=5 \
    quantize_bits=4,8,16 \
    lora.config.r=2,4,8,16,32\
    hf_model=Qwen/Qwen3-0.6B,Qwen/Qwen3-1.7B,Qwen/Qwen3-4B,Qwen/Qwen3-8B \
    dataset@train_dataset=winogrande_xs,winogrande_s,winogrande_m,winogrande_l,ARC-Easy,ARC-Challenge,obqa \
    collate_fn=instruct\
    seed=0,1,2,3\
    pbar=False \
    gpu_id=0 #ray will handle CUDA_VISIBLE_DEVICES so we just set gpu_id=0 here


#python evaluate.py \
    #+lora=default \
    #hf_model=Qwen/Qwen3-8B \
    #dataset@train_dataset=obqa \
    #dataset@test_dataset=MMLU-Chem \
    #seed=0 \
    #pbar=True \
    #gpu_id=0 #ray will handle CUDA_VISIBLE_DEVICES so we just set gpu_id=0 here


#python evaluate.py \
    #+lora=default \
    #hf_model=Qwen/Qwen2.5-7B \
    #dataset.name=winogrande_s \
    #dataset.instruct=False \
    #seed=0 \
    #pbar=True \
    #checkpoint=\${logdir}/state_dict.pt \
    #gpu_id=0 #ray will handle CUDA_VISIBLE_DEVICES so we just set gpu_id=0 here

#python train_and_evaluate.py --multirun \
    #hydra/launcher=ray \
    #+hydra.launcher.ray.init.num_gpus=8 \
    #+hydra.launcher.ray.remote.num_gpus=1 \
    #+lora=default \
    #hf_model=Qwen/Qwen2.5-0.5B,Qwen/Qwen2.5-1.5B,Qwen/Qwen2.5-3B,Qwen/Qwen2.5-7B \
    #dataset.name=winogrande_s,winogrande_m,ARC-Easy,ARC-Challenge,obqa \
    #dataset.instruct=False \
    #seed=0,1,2 \
    #pbar=False \
    #gpu_id=0 #ray will handle CUDA_VISIBLE_DEVICES so we just set gpu_id=0 here

#python train_and_evaluate.py --multirun \
    #hydra/launcher=ray \
    #+hydra.launcher.ray.init.num_gpus=8 \
    #+hydra.launcher.ray.remote.num_gpus=1 \
    #+lora=default \
    #hf_model=Qwen/Qwen3-0.6B,Qwen/Qwen3-1.7B,Qwen/Qwen3-4B,Qwen/Qwen3-8B \
    #dataset.name=winogrande_s,winogrande_m,ARC-Easy,ARC-Challenge,obqa \
    #dataset.instruct=True \
    #seed=0,1,2 \
    #gpu_id=0 #ray will handle CUDA_VISIBLE_DEVICES so we just set gpu_id=0 here
