python train_and_evaluate.py --multirun \
    hydra/launcher=ray \
    +hydra.launcher.ray.init.num_gpus=8 \
    +hydra.launcher.ray.remote.num_gpus=1 \
    +lora=default \
    +lora/wrapper=mcdropout \
    lora.config.lora_dropout=0.1 \
    samples.test.backbone=10 \
    hf_model=Qwen/Qwen3-VL-2B-Instruct,Qwen/Qwen3-VL-4B-Instruct,Qwen/Qwen3-VL-8B-Instruct \
    dataset@train_dataset=slake,mmstar,MathVerse,srqa \
    collate_fn=vlm \
    pbar=True \
    seed=0,1,2,3\
    gpu_id=0
