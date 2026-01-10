import os
import numpy  # needed (don't change it)
import torch
import torch.nn.functional as F
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm, trange
from accelerate.utils import set_seed
from transformers import AutoProcessor
from bayesadapt.utils import load_model, split_batch, infer_logdir_from_cfg
from bayesadapt.lorawrappers import TFBLoraWrapper, VILoraWrapper
from torch.utils.data import DataLoader
from evaluate import evaluate

def set_cov(model, beta):
    for module in model.modules():
        if isinstance(module, TFBLoraWrapper):
            module.set_cov(beta)

@hydra.main(config_path="./conf", config_name="default", version_base=None)
def main(cfg):
    cfg.logdir = infer_logdir_from_cfg(cfg)
    train(cfg)

def train(cfg):
    print(cfg)
    import ipdb; ipdb.set_trace() # noqa
    set_seed(cfg.seed)
    os.makedirs(cfg.logdir, exist_ok=True)
    yaml_str = OmegaConf.to_yaml(cfg)
    with open(os.path.join(cfg.logdir, "config.yaml"), "w") as f:
        f.write(yaml_str)

    tokenizer = AutoProcessor.from_pretrained(cfg.hf_model, trust_remote_code=True)
    dataset = instantiate(cfg.dataset, split='train')
    # train_loader = DataLoader(
        # dataset,
        # batch_size=cfg.optim.batch_size,
        # shuffle=True,
        # num_workers=1,
        # persistent_workers=True,
        # collate_fn=instantiate(cfg.collate_fn, tokenizer)
    # )
    target_ids = tokenizer.convert_tokens_to_ids(dataset.labels)

    device = torch.device(f"cuda:{cfg.gpu_id}" if torch.cuda.is_available() else "cpu")
    model = load_model(cfg, device, class_ids=target_ids)

    set_cov(model, beta=0.0)
    results = evaluate(cfg, model=model, split='test', verbose=True, save=False)
    orig_nll = evaluate(cfg, model=model, split='train', verbose=False, save=False)[0]['NLL']
    print(f"Original NLL: {orig_nll}")
    
    #low, high = 0.001, 0.015
    low, high = cfg.optim.low_start, cfg.optim.high_start
    best = high  
    for _ in range(cfg.optim.max_train_steps):
        mid = (low + high) / 2
        set_cov(model, beta=mid)
        new_nll = evaluate(cfg, model=model, split='train', verbose=False, save=False)[0]['NLL']
        print(f"Testing beta: {mid}, NLL: {new_nll}")

        #loss_change_ratio = (abs(current_nll_loss.item() - ori_nll_loss.item()) / ori_nll_loss.item())/self.all_ori_predicted_classes.size(0)
        
        ratio = abs(new_nll - orig_nll) / orig_nll / len(dataset)
        print(f"Loss change ratio: {ratio}")

        if ratio > cfg.optim.target_ratio:
            best = mid
            high = mid
        else:
            low = mid
    
    print(f"Best beta found: {best}")
    set_cov(model, beta=best)
    test_result = evaluate(cfg, model=model, split='test', verbose=True, save=False)
    import ipdb; ipdb.set_trace() # noqa

    sd = model.state_dict()
    sd = {k: v for k, v in sd.items() if 'lora_' in k}
    torch.save(sd, os.path.join(cfg.logdir, "state_dict.pt"))
    return model
    
if __name__ == "__main__":
    main()
