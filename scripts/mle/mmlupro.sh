python active_learn.py \
    +lora=default \
    lora.config.r=8 \
    quantize_bits=16  \
    hf_model=Qwen/Qwen3-8B \
    optim.max_train_steps=1000 \
    dataset@train_dataset=MMLU-Pro \
    collate_fn=instruct\
    seed=0\
    pbar=True \
    overwrite=True \
    gpu_id=0 #ray will handle CUDA_VISIBLE_DEVICES so we just set gpu_id=0 here
