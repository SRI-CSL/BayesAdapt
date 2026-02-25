python evaluate.py --multirun \
    hydra/launcher=ray \
    +hydra.launcher.ray.init.num_gpus=8 \
    +hydra.launcher.ray.remote.num_gpus=1 \
    +lora=default \
    +lora/wrapper=blob \
    optim=vi \
    trainer=vi \
    samples.test.backbone=10 \
    hf_model=Qwen/Qwen3-0.6B,Qwen/Qwen3-1.7B,Qwen/Qwen3-4B,Qwen/Qwen3-8B,Qwen/Qwen3-14B \
    dataset@train_dataset=obqa \
    dataset@test_dataset=MMLU-Chem,MMLU-Physics,MMLU-Math,MMLU-Bio,MMLU-CS \
    load_pretrained_checkpoint=True \
    collate_fn=instruct\
    seed=0,1,2,3\
    pbar=True \
    gpu_id=0 
