import os
import json
import torch
import torch.nn.functional as F
import hydra
from hydra.utils import instantiate
from tqdm import tqdm, trange
from accelerate.utils import set_seed
from peft import get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer#, BitsAndBytesConfig
from lorawrappers.utils import wrap_lora_layers 
from torchmetrics.functional import calibration_error, accuracy

@hydra.main(config_path="../conf", config_name="default", version_base=None)
def main(cfg):
    print(cfg)
    os.makedirs(cfg.logdir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg.hf_model, trust_remote_code=True)
    dataset = instantiate(cfg.dataset, tokenizer)
    dataset.get_loaders()
    test_loader = dataset.test_dataloader
    target_ids = dataset.target_ids.squeeze(-1)

    device = torch.device(f"cuda:{cfg.gpu_id}" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.hf_model, 
        quantization_config=None,
        device_map=device,
        torch_dtype=torch.bfloat16
    )

    if 'lora' in cfg:
        peft_config = instantiate(cfg.lora.config)
        model = get_peft_model(model, peft_config)
        if 'wrapper' in cfg.lora:
            wrapper_fn = instantiate(cfg.lora.wrapper)
            wrap_lora_layers(model, wrapper_fn)
            model = model.to(device) #make sure modified layers are on the right device
        sd_path = os.path.join(cfg.logdir, "state_dict.pt")
        if os.path.exists(sd_path):
            sd = torch.load(sd_path)
            model.load_state_dict(sd, strict=False)
            print('model loaded from', sd_path)
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
            logits = []
            
            torch.cuda.reset_peak_memory_stats(device)
            start_event.record()
            for i in range(cfg.n_test_samples):
                with torch.no_grad() and torch.inference_mode():
                    all_logits = model(**inputs).logits
                    logits_i = all_logits[:, -1, target_ids]
                logits.append(logits_i)
            end_event.record()
            torch.cuda.synchronize()
            peak_memories.append(torch.cuda.max_memory_allocated(device))
            elapsed_times.append(start_event.elapsed_time(end_event))

            logits = torch.stack(logits, dim=1) # (batch_size, n_samples, n_classes)
            sample_probs = torch.softmax(logits, dim=-1) # (batch_size, n_samples, n_classes)
            probs = sample_probs.mean(dim=1) # (batch_size, n_classes)
            test_probs.append(probs.cpu())
            test_labels.append(labels.cpu())

        test_probs = torch.cat(test_probs, dim=0)
        test_logprobs = torch.log(test_probs)
        test_preds = test_probs.argmax(dim=-1)
        test_labels = torch.cat(test_labels, dim=0)

        #exclude the first couple results to skip warmup effects
        elapsed_times = torch.tensor(elapsed_times[5:]) / 1000.0 #convert to seconds
        peak_memories = torch.tensor(peak_memories[5:]) / (1024 ** 3) #convert to GB
        
        result = {'seed': seed.item()}
        result['elapsed_time'] = {
            'mean': elapsed_times.mean().item(),
            'std': elapsed_times.std().item(),
            'min': elapsed_times.min().item(),
            'max': elapsed_times.max().item(),
        }
        result['peak_memory'] = {
            'mean': peak_memories.mean().item(),
            'std': peak_memories.std().item(),
            'min': peak_memories.min().item(),
            'max': peak_memories.max().item(),
        }
        metric_kwargs = {'task': 'multiclass', 'num_classes': len(target_ids)}
        result['test_acc'] = accuracy(test_preds, test_labels, **metric_kwargs).item()
        result['test_ece'] = calibration_error(test_probs, test_labels, n_bins=15, **metric_kwargs).item()
        result['test_nll'] = F.nll_loss(test_logprobs, test_labels, reduction="mean").item()
        results.append(result)
        print(result)
    
    json_path = os.path.join(cfg.logdir, "results.json")
    with open(json_path, "w") as f:
        json.dump(results, f)
    print(results)
    
if __name__ == "__main__":
    main()
