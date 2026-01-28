python train_and_evaluate.py \
    +lora=default \
    lora.config.r=8 \
    quantize_bits=16  \
    hf_model=Qwen/Qwen3-8B \
    dataset@train_dataset=MMLU-Pro \
    optim.train_split=test \
    collate_fn=instruct\
    seed=0\
    pbar=True \
    gpu_id=0 #ray will handle CUDA_VISIBLE_DEVICES so we just set gpu_id=0 here
