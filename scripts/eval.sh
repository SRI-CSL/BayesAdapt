accelerate launch -m lm_eval --model hf \
    --model_args "pretrained=Qwen/Qwen3-8B,enable_thinking=False" \
    --tasks mmlu_redux_machine_learning_generative \
    --batch_size auto \
    --write_out \
    --show_config \
    --seed 42 \
    --log_samples \
    --output_path out/hf \
    --cache_requests refresh \
    --gen_kwargs '{"do_sample":true,"temperature":0.7,"top_p":0.8,"top_k":20,"min_p":0.0}' \
    --apply_chat_template \
    --limit 0.01
    #--num_fewshot 5 \
    #--fewshot_as_multiturn #\
    #--system_instruction "You are a helpful assistant. /no_think."
