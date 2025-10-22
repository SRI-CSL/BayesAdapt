import os
import numpy  # needed (don't change it)
import math
import json
import torch
import torch.nn.functional as F
import wandb
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm, trange
from accelerate.utils import set_seed
from peft import get_peft_model
from lorawrappers import VILoraWrapper
from transformers import AutoModelForCausalLM, AutoTokenizer #, BitsAndBytesConfig
from lorawrappers.utils import wrap_lora_layers 

@hydra.main(config_path="../conf", config_name="default", version_base=None)
def main(cfg):
    print(cfg)
    set_seed(cfg.seed)
    os.makedirs(cfg.logdir, exist_ok=True)
    yaml_str = OmegaConf.to_yaml(cfg)
    with open(os.path.join(cfg.logdir, "config.yaml"), "w") as f:
        f.write(yaml_str)

    tokenizer = AutoTokenizer.from_pretrained(cfg.hf_model, trust_remote_code=True)
    dataset = instantiate(cfg.dataset, tokenizer)
    dataset.get_loaders()
    train_loader = dataset.train_dataloader
    target_ids = dataset.target_ids.squeeze(-1)

    device = torch.device(f"cuda:{cfg.gpu_id}" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.hf_model, 
        quantization_config=None,
        device_map=device,
        torch_dtype=torch.bfloat16
    )

    peft_config = instantiate(cfg.lora.config)
    peft_config.target_modules = list(peft_config.target_modules) #make sure it's a list, otherwise save_pretrained fails
    model = get_peft_model(model, peft_config)
    if 'wrapper' in cfg.lora:
        wrapper_fn = instantiate(cfg.lora.wrapper)
        wrap_lora_layers(model, wrapper_fn)
        model = model.to(device) #make sure modified layers are on the right device
    
    model.print_trainable_parameters()
    num_trainable_params, total_params = model.get_nb_trainable_parameters()
    with open(os.path.join(cfg.logdir, "num_params.json"), "w") as f:
        json.dump({'trainable': num_trainable_params, 'total': total_params}, f)

    decay_params, no_decay_params = [], []
    for n, p in model.named_parameters():
        if p.requires_grad:
            if any(nd in n for nd in cfg.optim.no_decay):
                no_decay_params.append(p)
            else:
                decay_params.append(p)
    params = [
        {"params": decay_params, "weight_decay": cfg.optim.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    nll_optimizer = instantiate(cfg.optim.nll_optimizer, params)
    nll_scheduler = instantiate(cfg.optim.nll_scheduler, nll_optimizer)
    if 'kl_optimizer' in cfg.optim:
        kl_optimizer = instantiate(cfg.optim.kl_optimizer, model.parameters())
        kl_scheduler = instantiate(cfg.optim.kl_scheduler, kl_optimizer, num_samples=dataset.num_samples)

    n_epochs = math.ceil(cfg.optim.max_train_steps / len(train_loader))
    earlystop_n_epochs = 0
    if cfg.optim.early_stop_steps > 0:
        earlystop_n_epochs = math.ceil(cfg.optim.early_stop_steps / len(train_loader))
    
    wandb.init(
        project=cfg.wandb.project,
        name=cfg.wandb.name,
        entity=os.environ.get("WANDB_ENTITY", None),
        config=dict(cfg),
    )
        
    model.train()
    for epoch in trange(n_epochs, desc="Epoch"):
        if cfg.optim.early_stop_steps > 0 and epoch >= earlystop_n_epochs:
            break
        for batch in tqdm(train_loader, leave=False):
            batch = [b.to(device) for b in batch]
            inputs, labels, _ = batch
            
            logits = []
            for i in range(cfg.n_train_samples):
                logits_i = model(**inputs).logits[:, -1, target_ids]
                logits.append(logits_i)
            logits = torch.stack(logits, dim=1) # (B, num_samples, num_classes)
            log_probs = torch.log_softmax(logits, dim=-1)
            
            B, num_samples, num_classes = logits.shape
            labels = labels.unsqueeze(-1).expand(B, num_samples)
            acc = (log_probs.argmax(dim=-1) == labels).float().mean()
            
            nll_vals = F.nll_loss(
                log_probs.view(B * num_samples, num_classes),
                labels.reshape(B * num_samples),
                reduction="none"
            ).reshape(B, num_samples)
            nll_loss = nll_vals.mean()
            assert not math.isnan(nll_loss)
            nll_loss.backward()
            nll_optimizer.step()
            nll_optimizer.zero_grad()
            nll_scheduler.step()

            log = {
                'train/nll_loss': nll_loss.item(),
                'train/acc': acc.item(),
                'train/nll_lr': nll_optimizer.param_groups[0]["lr"],
            }

            if 'kl_optimizer' in cfg.optim:
                kl_divs = []
                for module in model.modules():
                    if isinstance(module, VILoraWrapper):
                        kl_divs.append(module.kl_div)

                kl_loss = torch.sum(torch.stack(kl_divs), dim=0)
                assert not math.isnan(kl_loss)
                kl_loss.backward()
                kl_optimizer.step()
                kl_optimizer.zero_grad()
                kl_scheduler.step()
                log['train/kl_loss'] = kl_loss.item()
                log['train/elbo'] = (nll_loss + kl_loss).item()
                log['train/kl_lr'] = kl_optimizer.param_groups[0]["lr"]

            wandb.log(log)

    sd = model.state_dict()
    sd = {k: v for k, v in sd.items() if 'lora_' in k}
    torch.save(sd, os.path.join(cfg.logdir, "state_dict.pt"))
    wandb.finish()
    
if __name__ == "__main__":
    main()
