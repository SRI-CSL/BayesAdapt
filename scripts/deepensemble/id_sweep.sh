for dataset in winogrande_xs winogrande_s winogrande_m winogrande_l; do
    for gpu_id in 4 5 6 7; do
        seed=$((gpu_id - 4))
        python train_and_evaluate.py \
            +lora=default \
            +lora/wrapper=deepensemble \
            lora.config.r=8 \
            hf_model=Qwen/Qwen3-14B \
            dataset@train_dataset=$dataset \
            collate_fn=instruct\
            lora.wrapper.ensemble_size=10 \
            samples.train.backbone=10 \
            samples.test.backbone=10 \
            optim.batch_size=1 \
            seed=$seed\
            pbar=True \
            gpu_id=$gpu_id &
    done
    wait
done

#python train_and_evaluate.py \
    #+lora=default \
    #+lora/wrapper=deepensemble \
    #lora.config.r=8 \
    #hf_model=Qwen/Qwen3-14B \
    #dataset@train_dataset=obqa \
    #collate_fn=instruct\
    #lora.wrapper.ensemble_size=10 \
    #samples.train.backbone=10 \
    #samples.test.backbone=10 \
    #optim.batch_size=1 \
    #seed=0\
    #pbar=True \
    #gpu_id=7

#python train_and_evaluate.py --multirun \
    #hydra/launcher=ray \
    #+hydra.launcher.ray.init.num_gpus=8 \
    #+hydra.launcher.ray.remote.num_gpus=1 \
    #+lora=default \
    #+lora/wrapper=deepensemble \
    #lora.config.r=8 \
    #hf_model=Qwen/Qwen3-0.6B,Qwen/Qwen3-1.7B,Qwen/Qwen3-4B,Qwen/Qwen3-8B \
    #dataset@train_dataset=ARC-Easy,ARC-Challenge,obqa \
    #collate_fn=instruct\
    #lora.wrapper.ensemble_size=10 \
    #samples.train.backbone=10 \
    #samples.test.backbone=10 \
    #seed=0,1,2,3\
    #pbar=False \
    #gpu_id=0 #ray will handle CUDA_VISIBLE_DEVICES so we just set gpu_id=0 here
