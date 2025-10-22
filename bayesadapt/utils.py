import os
import torch
from peft import get_peft_model
from transformers import AutoModelForCausalLM, AutoConfig
from lorawrappers.utils import wrap_lora_layers 
from hydra.utils import instantiate

def load_model(cfg, device):
    model_config = AutoConfig.from_pretrained(cfg.hf_model)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.hf_model, 
        quantization_config=None,
        device_map=device,
        torch_dtype=torch.bfloat16,
        tie_word_embeddings=False,
    )

    #some models share the weights between the embedding layer and lm_head
    #this is typically done to save memory for small models (i.e. Qwen2.5-0.5B)
    #here we explicitly untie the weights and copy the embedding weights to the lm_head
    #this ensures that only the last layer is stochastic during VI approaches
    if model_config.tie_word_embeddings:
        embed_weights = model.model.embed_tokens.weight.detach().clone()
        sd = {'weight': embed_weights}
        model.lm_head.load_state_dict(sd)
        model.config.tie_word_embeddings = False
        print("copied embedding weights to lm_head weights")

    if 'lora' in cfg:
        peft_config = instantiate(cfg.lora.config)
        peft_config.target_modules = list(peft_config.target_modules) #make sure it's a list, otherwise save_pretrained fails
        model = get_peft_model(model, peft_config)
        if 'wrapper' in cfg.lora:
            wrapper_fn = instantiate(cfg.lora.wrapper)
            wrap_lora_layers(model, wrapper_fn, cfg.lora.wrapper.target_modules)
            model = model.to(device) #make sure modified layers are on the right device

    if os.path.exists(cfg.checkpoint):
        sd = torch.load(cfg.checkpoint, map_location='cpu')
        model.load_state_dict(sd, strict=False)
        print('model loaded from', cfg.checkpoint)
    return model
