python train_and_evaluate.py \
    +lora=default \
    +lora/wrapper=scalabl \
    optim=vi \
    trainer=vi \
    lora.config.r=8 \
    quantize_bits=16  \
    hf_model=Qwen/Qwen3-8B \
    optim.max_train_steps=5000 \
    dataset@train_dataset=circuit_logic \
    samples.test.backbone=10 \
    collate_fn=instruct\
    seed=0\
    pbar=True \
    overwrite=True \
    gpu_id=7


python evaluate.py \
    +lora=default \
    +lora/wrapper=scalabl \
    optim=vi \
    trainer=vi \
    lora.config.r=8 \
    quantize_bits=16  \
    hf_model=Qwen/Qwen3-8B \
    samples.test.backbone=10 \
    load_pretrained_checkpoint=True \
    optim.max_train_steps=5000 \
    dataset@train_dataset=circuit_logic \
    dataset@test_dataset=expression_logic \
    collate_fn=instruct\
    seed=0\
    pbar=True \
    overwrite=True \
    gpu_id=7
