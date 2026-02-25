python train_and_evaluate.py --multirun \
    hydra/launcher=ray \
    +hydra.launcher.ray.init.num_gpus=8 \
    +hydra.launcher.ray.remote.num_gpus=1 \
    +lora=default \
    +lora/wrapper=tfb \
    optim=binary_search \
    trainer=binary_search \
    optim.max_train_steps=5 \
    samples.test.backbone=10 \
    lora.load_mle_checkpoint=True \
    hf_model=Qwen/Qwen3-VL-2B-Instruct,Qwen/Qwen3-VL-4B-Instruct,Qwen/Qwen3-VL-8B-Instruct \
    dataset@train_dataset=slake,mmstar,MathVerse,srqa \
    collate_fn=vlm\
    seed=0,1,2,3\
    pbar=True \
    gpu_id=0
