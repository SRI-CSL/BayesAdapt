import os
import json
import torch
import torch.nn.functional as F
import hydra
from hydra.utils import instantiate
from tqdm import tqdm, trange
from accelerate.utils import set_seed
from transformers import AutoProcessor
from torchmetrics.functional import calibration_error, accuracy
from bayesadapt.utils import load_model, infer_logdir_from_cfg, average_log_probs, load_dataloader
# from bayesadapt.datasets.collate import base_collate_fn, instruct_collate_fn
from torch.utils.data import DataLoader

import torch.nn.functional as F

def entropy_from_logits(logits):
    log_probs = F.log_softmax(logits, dim=1)
    probs = torch.exp(log_probs)
    return -torch.sum(probs * log_probs, dim=1)

@hydra.main(config_path="./conf", config_name="default", version_base=None)
def main(cfg):
    cfg.logdir = infer_logdir_from_cfg(cfg)
    evaluate(cfg)

def evaluate(cfg, optimizer, model=None, split='test', verbose=True, save=True):
    if verbose:
        print(cfg)
    os.makedirs(cfg.logdir, exist_ok=True)

    dataloader = load_dataloader(cfg, split)

    device = torch.device(f"cuda:{cfg.gpu_id}" if torch.cuda.is_available() else "cpu")

    if model is None:
        model = load_model(cfg, device, class_ids=dataloader.class_ids)
    else:
        model.to(device)
    
    try:
        num_trainable_params, total_params = model.get_nb_trainable_parameters()
        num_base_params = total_params - num_trainable_params
    except: #not a LoRA model
        num_base_params = sum(p.numel() for p in model.parameters())
        num_trainable_params = 0
        total_params = num_base_params

    params_info = {'num_trainable_params': num_trainable_params, 'num_total_params': total_params, 'num_base': num_base_params}

    model.eval()
    results = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    seeds = torch.arange(cfg.seed, cfg.seed + cfg.n_eval_trials)
    all_test_logits = []
    for seed in seeds:
        set_seed(seed.item())
        test_logits, test_labels, elapsed_times, peak_memories = [], [], [], []
        for batch in tqdm(dataloader, desc="Testing", disable=not cfg.pbar):
            batch = [b.to(device) for b in batch]
            #inputs, labels, _ = batch
            inputs, labels = batch
            sample_logits = []
                
            torch.cuda.reset_peak_memory_stats(device)
            start_event.record()
            for i in range(cfg.samples.test.backbone):
                with optimizer.sampled_params():
                    with torch.no_grad() and torch.inference_mode():
                        model_output = model(**inputs, output_hidden_states=True)
                        feats = model_output.hidden_states[-1][:, -1] #last layer, last token, (batch_size, hidden_size)

                    for j in range(cfg.samples.test.last_layer):
                        with torch.no_grad() and torch.inference_mode():
                            cls_logits = model.lm_head(feats)#[:, target_ids]  # (batch_size, n_classes)
                        sample_logits.append(cls_logits)
            end_event.record()
            torch.cuda.synchronize()
            peak_memories.append(torch.cuda.max_memory_allocated(device))
            elapsed_times.append(start_event.elapsed_time(end_event))

            sample_logits = torch.stack(sample_logits, dim=1) # (batch_size, n_samples, n_classes)
            # sample_logits = sample_logits.to(torch.float64) #use higher precision for stability
            # avg_log_probs = average_log_probs(sample_logits) # (batch_size, n_classes)
            # sample_probs = torch.softmax(sample_logits, dim=-1) # (batch_size, n_samples, n_classes)
            # avg_probs = sample_probs.mean(dim=1) # (batch_size, n_classes)
            # avg_log_probs = torch.log(avg_probs)
            test_logits.append(sample_logits.cpu())
            test_labels.append(labels.cpu())
        
        test_logits = torch.cat(test_logits, dim=0) # (num_examples, n_samples, n_classes)
        all_test_logits.append(test_logits)

        test_logits = test_logits.to(torch.float64) #use higher precision for stability
        test_logprobs = average_log_probs(test_logits) # (num_examples, n_classes)
        test_probs = torch.exp(test_logprobs)
        # test_probs = torch.cat(test_probs, dim=0)
        # test_logprobs = torch.log(test_probs)
        test_preds = test_probs.argmax(dim=-1)
        test_labels = torch.cat(test_labels, dim=0)

        #exclude the first couple results to skip warmup effects
        elapsed_times = torch.tensor(elapsed_times[5:]) / 1000.0 #convert to seconds
        peak_memories = torch.tensor(peak_memories[5:]) / (1024 ** 3) #convert to GB
        
        result = {'seed': seed.item()}
        result.update(params_info)
        result['latency'] = elapsed_times.mean().item()
        result['peak_memory'] = peak_memories.mean().item()
        
        metric_kwargs = {'task': 'multiclass', 'num_classes': len(dataloader.class_ids)}
        result['ACC'] = accuracy(test_preds, test_labels, **metric_kwargs).item()
        result['ECE'] = calibration_error(test_probs, test_labels, n_bins=15, **metric_kwargs).item()
        result['NLL'] = F.nll_loss(test_logprobs, test_labels, reduction="mean").item()
        results.append(result)
        if verbose:
            print(result)
        

    if save:
        all_test_logits = torch.stack(all_test_logits, dim=0) # (n_trials, num_examples, n_samples, n_classes)
        torch.save(all_test_logits, os.path.join(cfg.logdir, "test_logits.pt"))
        torch.save(test_labels, os.path.join(cfg.logdir, "test_labels.pt"))
        json_path = os.path.join(cfg.logdir, "results.json")
        with open(json_path, "w") as f:
            json.dump(results, f)
            f.write("\n")

    if verbose:
        print(results)
    return results
    
if __name__ == "__main__":
    main()
