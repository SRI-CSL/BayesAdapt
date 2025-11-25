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
from torchmetrics.functional import calibration_error, accuracy

import sys
#sys.path.append('./laplace-lora')  # noqa
from bayesadapt.laplace import Laplace


class WrappedModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, **kwargs):
        kwargs.pop('labels', None)
        logits = self.model(**kwargs).logits
        return logits[:, -1, :]

@hydra.main(config_path="./conf", config_name="default", version_base=None)
def main(cfg):
    print(cfg)
    os.makedirs(cfg.logdir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg.hf_model, trust_remote_code=True)
    dataset = instantiate(cfg.dataset, tokenizer)
    dataset.get_loaders()
    train_loader = dataset.train_dataloader
    test_loader = dataset.test_dataloader
    target_ids = dataset.target_ids.squeeze(-1)

    device = torch.device(f"cuda:{cfg.gpu_id}" if torch.cuda.is_available() else "cpu")
    model = load_model(cfg, device, class_ids=target_ids)
    model.eval()

    model = WrappedModel(model)
    la = Laplace(model, 'classification', prior_precision=1.0, subset_of_weights='all', hessian_structure='kron')
    la.fit(train_loader)
    prior_precision = la.optimize_prior_precision(
        method='marglik', 
        n_steps=100,
        lr=1e-1
    )
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    results = []
    samples = 100000
    seeds = torch.arange(cfg.seed, cfg.seed + cfg.n_eval_trials)
    for seed in seeds:
        set_seed(seed.item())
        test_probs, test_labels, elapsed_times, peak_memories = [], [], [], []
        for batch in tqdm(test_loader):
            # batch = {k: v.float() for k, v in batch.items()}
            #batch = {'input_ids': batch[0]['input_ids'], 'attention_mask': batch[0]['attention_mask'], 'labels': batch[1]}
            # labels = batch[1].to(device)
            test_labels.append(batch[1])
            batch = batch[0] #just in input_ids/attention_mask
            batch = {k: v.to(device) for k, v in batch.items()}
            
            torch.cuda.reset_peak_memory_stats(device)
            start_event.record()

            with torch.no_grad():
                f_mu, f_var = la._glm_predictive_distribution(batch)
            f_mu = f_mu.expand(samples, -1, -1)
            f_var = f_var.expand(samples, -1, -1, -1)
            eye = torch.eye(f_var.shape[-1], device=f_var.device)
            stabilized_var = f_var + (eye * 1e-6)
            L = torch.linalg.cholesky(stabilized_var).to(f_mu.dtype)
            noise = torch.randn_like(f_mu).unsqueeze(-1)
            perturbation = (L @ noise).squeeze(-1)
            logits = f_mu + perturbation
            sample_probs = torch.softmax(logits, dim=-1)
            probs = sample_probs.mean(dim=0)
           
            end_event.record()
            torch.cuda.synchronize()
            peak_memories.append(torch.cuda.max_memory_allocated(device))
            elapsed_times.append(start_event.elapsed_time(end_event))

            # log_probs = torch.log(probs)
            test_probs.append(probs.cpu())

        test_probs = torch.cat(test_probs, dim=0)
        test_preds = test_probs.argmax(dim=-1)
        test_logprobs = torch.log(test_probs)
        test_labels = torch.cat(test_labels, dim=0)


        elapsed_times = torch.tensor(elapsed_times[5:]) / 1000.0 #convert to seconds
        peak_memories = torch.tensor(peak_memories[5:]) / (1024 ** 3) #convert to GB
        
        result = {'seed': seed.item()}
        result['latency'] = elapsed_times.mean().item()
        result['peak_memory'] = peak_memories.mean().item()

        metric_kwargs = {'task': 'multiclass', 'num_classes': len(target_ids)}
        result['ACC'] = accuracy(test_preds, test_labels, **metric_kwargs).item()
        result['ECE'] = calibration_error(test_probs, test_labels, n_bins=15, **metric_kwargs).item()
        result['NLL'] = F.nll_loss(test_logprobs, test_labels, reduction="mean").item()
        results.append(result)
        print(result)

        # logits = f_mu + (torch.linalg.cholesky(f_var + torch.eye(f_var.shape[-1]).to(f_var.device)*1e-6).to(f_mu.dtype) @ torch.randn_like(f_mu).unsqueeze(-1).to(f_mu.dtype).to(accelerator.device)).squeeze(-1)
        # logits = torch.softmax(logits, dim=-1).mean(0)
        
        # predictions = logits.argmax(dim=-1)
    import ipdb; ipdb.set_trace() # noqa
    
    
if __name__ == "__main__":
    main()
