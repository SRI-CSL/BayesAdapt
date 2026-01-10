import os
import math
import torch
from peft import get_peft_model
from transformers import AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from bayesadapt.lorawrappers.utils import wrap_lora_layers 
# from bayesadapt.lorawrappers import cls2name
from hydra.utils import instantiate
from hydra.core.hydra_config import HydraConfig

def average_log_probs(sample_logits):
    B, n_samples, C = sample_logits.shape
    sample_log_probs = torch.log_softmax(sample_logits, dim=-1)
    avg_log_probs = torch.logsumexp(sample_log_probs, dim=1) - math.log(n_samples)
    return avg_log_probs


def infer_logdir_from_cfg(cfg):
    if 'logdir' in cfg and cfg.logdir != 'infer':
        return cfg.logdir
    
    # model_family, model_name = cfg.hf_model.split('/')
    if 'lora' in cfg:
        r = cfg.lora.config.r
        if 'wrapper' in cfg.lora:
            wrapper_name = HydraConfig.get().runtime.choices['lora/wrapper']
            # wrapper_cls_name = cfg.lora.wrapper._target_.split('.')[-1]
            # wrapper_name = cls2name[wrapper_cls_name]
        else:
            wrapper_name = 'mle' if cfg.optim.weight_decay == 0 else 'map'
    else:
        wrapper_name = 'zeroshot'
        r = 0
    
    # collate_fn_name = cfg.collate_fn._target_.split('.')[-1].split('_')[0].lower()
    # prompt_type = 'instruct' if cfg.dataset.instruct else 'base'
    
    logdir = os.path.join(
        'logs',
        cfg.hf_model,
        # model_family,
        # model_name,
        f"{cfg.quantize_bits}bit",
        wrapper_name,
        f"rank{r}",
        HydraConfig.get().runtime.choices.dataset,
        HydraConfig.get().runtime.choices.collate_fn,
        f"seed{cfg.seed}",
    )
    return logdir

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
        peft_config = instantiate(cfg.lora.config, lora_alpha=cfg.lora.config.r*2)
        peft_config.target_modules = list(peft_config.target_modules) #make sure it's a list, otherwise save_pretrained fails
        model = get_peft_model(model, peft_config)
        
        if cfg.lora.checkpoint is not None:
            assert os.path.exists(cfg.lora.checkpoint), f"Checkpoint {cfg.lora.checkpoint} does not exist"
            sd = torch.load(cfg.lora.checkpoint, map_location='cpu')
            model.load_state_dict(sd, strict=False)
            print('unwrapped lora loaded from', cfg.lora.checkpoint)

        for name, param in model.named_parameters():
            if param.requires_grad: #only update LoRA parameters
                param.requires_grad = cfg.lora.requires_grad

        if 'wrapper' in cfg.lora:
            wrapper_fn = instantiate(cfg.lora.wrapper)
            wrap_lora_layers(model, wrapper_fn, cfg.lora.wrapper.target_modules)
            model = model.to(device) #make sure modified layers are on the right device

    #if not cfg.load_before_wrap and cfg.checkpoint is not None:
    if cfg.checkpoint is not None:
        assert os.path.exists(cfg.checkpoint), f"Checkpoint {cfg.checkpoint} does not exist"
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
