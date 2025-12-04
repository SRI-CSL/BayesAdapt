import os
import torch
from peft import get_peft_model
from transformers import AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from bayesadapt.lorawrappers.utils import wrap_lora_layers 
from hydra.utils import instantiate

def load_model(cfg, device, class_ids=None):
    model_config = AutoConfig.from_pretrained(cfg.hf_model)
    if cfg.quantize_bits == 4:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        torch_dtype = torch.bfloat16
    elif cfg.quantize_bits == 8:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        torch_dtype = torch.float16
    elif cfg.quantize_bits == 16:
        bnb_config = None
        torch_dtype = torch.bfloat16
    else:
        raise ValueError(f"Unsupported quantization bits: {cfg.quantize_bits}")

    if 'Qwen3-VL' in cfg.hf_model:
        from transformers import Qwen3VLForConditionalGeneration as model_class
    elif 'gemma-3' in cfg.hf_model:
        from transformers import Gemma3ForConditionalGeneration as model_class
    else:
        model_class = AutoModelForCausalLM

    model = model_class.from_pretrained(
        cfg.hf_model, 
        quantization_config=bnb_config,
        device_map=device,
        dtype=torch_dtype,
        tie_word_embeddings=False,
    )

    #some models share the weights between the embedding layer and lm_head
    #this is typically done to save memory for small models (i.e. Qwen2.5-0.5B)
    #so we explicitly untie the weights and copy the embedding weights to the lm_head
    #this ensures that only the last layer is stochastic during VI approaches
    if model_config.tie_word_embeddings:
        embed_weights = model.get_input_embeddings().weight.detach().clone()
        sd = {'weight': embed_weights}
        model.lm_head.load_state_dict(sd)
        model.config.tie_word_embeddings = False
        print("copied embedding weights to lm_head weights")
    
    #following the approach of Laplace LoRA
    #we only keep the classifier weights for the target classes
    if class_ids is not None:
        classifier_weights = model.lm_head.weight.detach().clone()
        new_head = torch.nn.Linear(
            in_features=classifier_weights.shape[1],
            out_features=len(class_ids),
            bias=False,
            dtype=classifier_weights.dtype,
            device=classifier_weights.device,
        )
        selected_weights = classifier_weights[class_ids]
        sd = {'weight': selected_weights}
        new_head.load_state_dict(sd)
        model.register_module('lm_head', new_head)
        model.config.vocab_size = len(class_ids)
   
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


#batch is a list of length N (which is NOT the batch size)
#each list item is either a tensor os shape B x ... or a dict of such tensors, where B is the batch size
#the output should be a list of length num_chunks with each item being a list of length N
def split_batch(inputs, labels, num_chunks=1):
    chunked_inputs = {}
    for key, value in inputs.items():
        chunked_inputs[key] = torch.chunk(value, num_chunks)
    chunked_labels = torch.chunk(labels, num_chunks)
    
    split_batches = []
    for i in range(num_chunks):
        split_batch = ({key: chunked_inputs[key][i] for key in chunked_inputs}, chunked_labels[i])
        split_batches.append(split_batch)
    
    return split_batches
