import os
import json
import numpy  # needed (don't change it)
import torch
import torch.nn.functional as F
import hydra
from hydra.utils import instantiate
from tqdm import tqdm, trange
from accelerate.utils import set_seed
from peft import get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer #, BitsAndBytesConfig
from lorawrappers.utils import wrap_lora_layers 
from torchmetrics.functional import calibration_error
from torchmetrics import Accuracy, CalibrationError
from accelerate import Accelerator

@hydra.main(config_path="../conf", config_name="default", version_base=None)
def main(cfg):
    print(cfg)
    set_seed(cfg.seed)
    accelerator = Accelerator()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model, trust_remote_code=True)
    dataset = instantiate(cfg.dataset, tokenizer)
    dataset.get_loaders()
    test_loader = dataset.test_dataloader
    target_ids = dataset.target_ids.squeeze(-1)

    device = torch.device(f"cuda:{cfg.gpu_id}" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model, 
        quantization_config=None,
        device_map=device,
        torch_dtype=torch.bfloat16
    )
    peft_config = instantiate(cfg.lora.config)
    model = get_peft_model(model, peft_config)
    wrapper_fn = instantiate(cfg.lora.wrapper)
    wrap_lora_layers(model, wrapper_fn)
    model = model.to(device) #make sure modified layers are on the right device
    sd = torch.load(os.path.join(cfg.logdir, "state_dict.pt"))
    model.load_state_dict(sd, strict=False)
    # sd = model.state_dict()
    # keys = list(sd.keys())
    # for k in keys:
        # print(k, sd[k].shape)
    # import ipdb; ipdb.set_trace() # noqa
    model, test_loader = accelerator.prepare(model, test_loader)

    # import ipdb; ipdb.set_trace() # noqa
    # model.load_adapter(cfg.logdir, "default")
    model.eval()

    
    metric_kwargs = {'task': 'multiclass', 'num_classes': len(target_ids)}
    acc_metric = Accuracy(**metric_kwargs).to(device)
    ece_metric = CalibrationError(**metric_kwargs).to(device)

    test_probs, test_labels = [], []
    for step, batch in enumerate(tqdm(test_loader, desc="Testing")):
        batch = [b.to(device) for b in batch]
        inputs, labels, _ = batch
        
        logits = []
        for i in range(cfg.n_test_samples):
            with torch.no_grad() and torch.inference_mode():
                logits_i = model(**inputs).logits[:, -1, target_ids]
            logits.append(logits_i)
        logits = torch.stack(logits, dim=1) # (batch_size, n_samples, n_classes)
        probs = torch.softmax(logits, dim=-1).mean(dim=1) # (batch_size, n_classes)
        
        acc_metric.update(probs, labels)
        ece_metric.update(probs, labels)

        test_probs.append(probs.cpu())
        test_labels.append(labels.cpu())
    
    print(acc_metric.compute())
    print(ece_metric.compute())
    test_probs = torch.cat(test_probs, dim=0)
    test_logprobs = torch.log(test_probs)
    test_preds = test_probs.argmax(dim=-1)
    test_labels = torch.cat(test_labels, dim=0)

    result = {}
    result['test_acc'] = (test_preds == test_labels).float().mean().item()
    result['test_ece'] = calibration_error(
        test_probs, test_labels, 
        task='multiclass', 
        num_classes=len(target_ids),
        n_bins=15
    ).item()
    result['test_nll'] = F.nll_loss(test_logprobs, test_labels, reduction="mean").item()
    json_path = os.path.join(cfg.logdir, "results.json")
    with open(json_path, "w") as f:
        json.dump(result, f)
    print(result)
    
if __name__ == "__main__":
    main()
