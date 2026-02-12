python active_learn.py \
    +lora=default \
    +lora/wrapper=tfb \
    optim=binary_search \
    trainer=binary_search \
    lora.config.r=8 \
    quantize_bits=16  \
    hf_model=Qwen/Qwen3-VL-8B-Instruct \
    optim.max_train_steps=1000 \
    samples.test.backbone=10 \
    dataset@train_dataset=srqa \
    collate_fn=vlm\
    seed=0\
    pbar=True \
    overwrite=True \
    gpu_id=0 #ray will handle CUDA_VISIBLE_DEVICES so we just set gpu_id=0 here
