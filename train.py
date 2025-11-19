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
from transformers import AutoTokenizer
from bayesadapt.utils import load_model, split_batch
from bayesadapt.lorawrappers import VILoraWrapper

@hydra.main(config_path="./conf", config_name="default", version_base=None)
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
    model = load_model(cfg, device, class_ids=target_ids)
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)
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
        for full_batch in tqdm(train_loader, leave=False):
            full_batch = [b.to(device) for b in full_batch]
            inputs, labels, _ = full_batch
            full_batch_size = labels.size(0)
            subbatches = split_batch(inputs, labels, num_chunks=cfg.optim.grad_accum_steps)
            full_nll_loss, num_correct = 0.0, 0.0
            for batch in subbatches:
                inputs, labels = batch
                logits = []
                for i in range(cfg.samples.train.backbone):
                    model_output = model(**inputs, output_hidden_states=True)
                    feats = model_output.hidden_states[-1][:, -1] #last layer, last token, (batch_size, hidden_size)
                    for j in range(cfg.samples.train.last_layer):
                        logits_ij = model.lm_head(feats)#[:, target_ids]  # (batch_size, n_classes)
                        logits.append(logits_ij)
                logits = torch.stack(logits, dim=1) # (B, num_samples, num_classes)
                log_probs = torch.log_softmax(logits, dim=-1)
                
                B, num_samples, num_classes = logits.shape
                labels = labels.unsqueeze(-1).expand(B, num_samples)
                num_correct += (log_probs.argmax(dim=-1) == labels).float().sum()
                # acc = (log_probs.argmax(dim=-1) == labels).float().mean()
                
                nll_vals = F.nll_loss(
                    log_probs.view(B * num_samples, num_classes),
                    labels.reshape(B * num_samples),
                    reduction="none"
                ).reshape(B, num_samples)
                nll_loss = nll_vals.mean() / len(subbatches)
                assert not math.isnan(nll_loss)
                nll_loss.backward() 
                full_nll_loss += nll_loss.item()
            
            # nll_loss.backward()
            nll_optimizer.step()
            nll_optimizer.zero_grad()
            nll_scheduler.step()

            log = {
                'train/nll_loss': (full_nll_loss / len(subbatches)),
                'train/acc': (num_correct / full_batch_size).item(),
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
