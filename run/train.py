import os
import numpy  # needed (don't change it)
import math
import torch
import torch.nn.functional as F
import wandb
import hydra
from hydra.utils import instantiate
from tqdm import tqdm, trange
from accelerate.utils import set_seed
from peft import get_peft_model, LoraConfig
from lorawrappers import VILoraWrapper
from transformers import AutoModelForCausalLM, AutoTokenizer #, BitsAndBytesConfig
from lorawrappers.utils import wrap_lora_layers 
from accelerate import Accelerator
from torchmetrics.functional import calibration_error
from torchmetrics import Accuracy, CalibrationError

@hydra.main(config_path="../conf", config_name="default", version_base=None)
def main(cfg):
    print(cfg)
    set_seed(cfg.seed)
    # os.putenv("MKL_SERVICE_FORCE_INTEL", "1")
    # os.putenv("NPY_MKL_FORCE_INTEL", "1")
    # accelerator = Accelerator()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model, trust_remote_code=True)
    dataset = instantiate(cfg.dataset, tokenizer)
    dataset.get_loaders()
    train_loader = dataset.train_dataloader
    target_ids = dataset.target_ids.squeeze(-1)

    device = torch.device(f"cuda:{cfg.gpu_id}" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model, 
        quantization_config=None,
        device_map=device,
        torch_dtype=torch.bfloat16
    )
    peft_config = instantiate(cfg.lora.config)
    # peft_config.target_modules = list(peft_config.target_modules) #make sure it's a list, otherwise save_pretrained fails
    # peft_config = LoraConfig(
        # task_type="CAUSAL_LM",
        # inference_mode=False,
        # r=cfg.lora.config.r,
        # lora_alpha=cfg.lora.config.lora_alpha,
        # lora_dropout=cfg.lora.config.lora_dropout,
        # target_modules=list(cfg.lora.config.target_modules)
    # )

    model = get_peft_model(model, peft_config)
    wrapper_fn = instantiate(cfg.lora.wrapper)
    wrap_lora_layers(model, wrapper_fn)
    model = model.to(device) #make sure modified layers are on the right device

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
    kl_optimizer = instantiate(cfg.optim.kl_optimizer, model.parameters())
    kl_scheduler = instantiate(cfg.optim.kl_scheduler, kl_optimizer, num_samples=dataset.num_samples)

    # model, train_loader, nll_optimizer, nll_scheduler, kl_optimizer, kl_scheduler = accelerator.prepare(
        # model, train_loader, nll_optimizer, nll_scheduler, kl_optimizer, kl_scheduler
    # )

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
            nll_loss.backward()
            # accelerator.backward(nll_loss)
            nll_optimizer.step()
            nll_optimizer.zero_grad()
            nll_scheduler.step()

            kl_divs = []
            for module in model.modules():
                if isinstance(module, VILoraWrapper):
                    kl_divs.append(module.kl_div)

            if len(kl_divs) > 0:
                kl_loss = torch.sum(torch.stack(kl_divs), dim=0)
                kl_loss.backward()
                pi = kl_scheduler.last_pi
                # accelerator.backward(kl_loss)
                kl_optimizer.step()
                kl_optimizer.zero_grad()
                kl_scheduler.step()
            else:
                kl = torch.tensor(0.0).to(device)

            assert not math.isnan(nll_loss)
            assert not math.isnan(kl_loss)
            # wandb.log({
                # "train/acc": acc.item(),
                # "train/nll": nll_loss.item(),
                # "train/kl": kl_loss.item(),
                # "train/elbo": (nll_loss + kl_loss).item(),
                # "train/nll_lr": nll_optimizer.param_groups[0]["lr"],
                # "train/kl_lr": kl_optimizer.param_groups[0]["lr"],
            # })

            wandb.log({
                "train_acc": acc.item() * 100,
                "train_nll_loss": nll_loss.item(),
                "kl_loss": kl_loss.item() * pi,
                "elbo_loss": (nll_loss + kl_loss).item(),
                "lr": nll_optimizer.param_groups[0]["lr"],
                #"pi": kl_scheduler.last_pi,
                "pi": pi,
                # "train/kl_lr": kl_optimizer.param_groups[0]["lr"],
            })


    #for the Qwen models, the lm_head and embedding layers are tied
    #setting save_embedding_layers to False ensures that base embedding weights arent saved needlessly
    #the lora weights for the lm_head/embedding layers are still saved 
    # save_folder = f"checkpoints/{args.modelwrapper}/{args.model}/{args.dataset}/{args.checkpoint_name}"
    if not os.path.exists(cfg.logdir):
        os.makedirs(cfg.logdir)
    
    # import ipdb; ipdb.set_trace() # noqa
    sd = model.state_dict()
    sd = {k: v for k, v in sd.items() if 'lora_' in k}
    torch.save(sd, os.path.join(cfg.logdir, "state_dict.pt"))

    # model.base_model = accelerator.unwrap_model(model.base_model)
    # model.save_pretrained(cfg.logdir, save_function=accelerator.save, save_embedding_layers=False)
    # import ipdb; ipdb.set_trace() # noqa
    #model.save_pretrained(cfg.logdir, save_embedding_layers=False)
    wandb.finish()
    
    # test_loader = dataset.test_dataloader
    # test_loader = accelerator.prepare(test_loader)
    # model.eval()

    # metric_kwargs = {'task': 'multiclass', 'num_classes': len(target_ids)}
    # acc_metric = Accuracy(**metric_kwargs).to(device)
    # ece_metric = CalibrationError(**metric_kwargs).to(device)
    # test_probs, test_labels = [], []
    # for step, batch in enumerate(tqdm(test_loader, desc="Testing")):
        # batch = [b.to(device) for b in batch]
        # inputs, labels, _ = batch

        # assert cfg.n_test_samples == 10
        
        # logits = []
        # for i in range(cfg.n_test_samples):
            # with torch.no_grad() and torch.inference_mode():
                # logits_i = model(**inputs).logits[:, -1, target_ids]
            # logits.append(logits_i)
        # logits = torch.stack(logits, dim=1) # (batch_size, n_samples, n_classes)
        # probs = torch.softmax(logits, dim=-1).mean(dim=1) # (batch_size, n_classes)
        
        # acc_metric.update(probs, labels)
        # ece_metric.update(probs, labels)

        # test_probs.append(probs.cpu())
        # test_labels.append(labels.cpu())
    
    # print(acc_metric.compute())
    # print(ece_metric.compute())
    # test_probs = torch.cat(test_probs, dim=0)
    # test_logprobs = torch.log(test_probs)
    # test_preds = test_probs.argmax(dim=-1)
    # test_labels = torch.cat(test_labels, dim=0)

    # result = {}
    # result['test_acc'] = (test_preds == test_labels).float().mean().item()
    # result['test_ece'] = calibration_error(
        # test_probs, test_labels, 
        # task='multiclass', 
        # num_classes=len(target_ids),
        # n_bins=15
    # ).item()
    # result['test_nll'] = F.nll_loss(test_logprobs, test_labels, reduction="mean").item()
    # print(result)





if __name__ == "__main__":
    main()
