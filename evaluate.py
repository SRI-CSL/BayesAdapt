import os
import json
import torch
import torch.nn.functional as F
import hydra
from hydra.utils import instantiate
from tqdm import tqdm, trange
from accelerate.utils import set_seed
from transformers import AutoTokenizer
from torchmetrics.functional import calibration_error, accuracy
from bayesadapt.utils import load_model

@hydra.main(config_path="./conf", config_name="default", version_base=None)
def main(cfg):
    print(cfg)
    os.makedirs(cfg.logdir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg.hf_model, trust_remote_code=True)
    dataset = instantiate(cfg.dataset, tokenizer)
    dataset.get_loaders()
    test_loader = dataset.test_dataloader
    target_ids = dataset.target_ids.squeeze(-1)

    device = torch.device(f"cuda:{cfg.gpu_id}" if torch.cuda.is_available() else "cpu")
    model = load_model(cfg, device, class_ids=target_ids)

    model.eval()
    results = []
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    seeds = torch.arange(cfg.seed, cfg.seed + cfg.n_eval_trials)
    for seed in seeds:
        set_seed(seed.item())
        test_probs, test_labels, elapsed_times, peak_memories = [], [], [], []
        for batch in tqdm(test_loader, desc="Testing"):
            batch = [b.to(device) for b in batch]
            inputs, labels, _ = batch
            sample_logits = []
            
            torch.cuda.reset_peak_memory_stats(device)
            start_event.record()
            for i in range(cfg.samples.test.backbone):
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
            sample_probs = torch.softmax(sample_logits, dim=-1) # (batch_size, n_samples, n_classes)
            avg_probs = sample_probs.mean(dim=1) # (batch_size, n_classes)
            test_probs.append(avg_probs.cpu())
            test_labels.append(labels.cpu())

        test_probs = torch.cat(test_probs, dim=0)
        test_logprobs = torch.log(test_probs)
        test_preds = test_probs.argmax(dim=-1)
        test_labels = torch.cat(test_labels, dim=0)

        #exclude the first couple results to skip warmup effects
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

    json_path = os.path.join(cfg.logdir, "results.json")
    with open(json_path, "w") as f:
        json.dump(results, f)
    print(results)
    
if __name__ == "__main__":
    main()
