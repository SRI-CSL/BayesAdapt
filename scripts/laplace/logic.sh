#python evaluate.py \
    #+lora=default \
    #optim=laplace \
    #trainer=laplace \
    #lora.load_mle_checkpoint=True \
    #optim.batch_size=1 \
    #lora.config.r=8 \
    #quantize_bits=16  \
    #hf_model=Qwen/Qwen3-8B \
    #dataset@train_dataset=circuit_logic \
    #dataset@test_dataset=circuit_logic \
    #collate_fn=instruct\
    #seed=0 \
    #pbar=True \
    #overwrite=False \
    #gpu_id=0

python evaluate.py --multirun \
    hydra/launcher=ray \
    +hydra.launcher.ray.init.num_gpus=8 \
    +hydra.launcher.ray.remote.num_gpus=1 \
    +lora=default \
    optim=laplace \
    trainer=laplace \
    lora.load_mle_checkpoint=True \
    lora.config.r=8 \
    quantize_bits=16  \
    hf_model=Qwen/Qwen3-0.6B,Qwen/Qwen3-1.7B,Qwen/Qwen3-4B \
    optim.max_train_steps=5000 \
    dataset@train_dataset=circuit_logic,expression_logic \
    dataset@test_dataset=expression_logic,circuit_logic \
    collate_fn=instruct\
    seed=0,1,2,3 \
    pbar=True \
    overwrite=False \
    gpu_id=0
