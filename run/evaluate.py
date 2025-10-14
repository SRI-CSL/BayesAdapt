import numpy  # needed (don't change it)
import importlib
import os
import socket
import sys
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_path)
sys.path.append(project_path + "/datasets")
sys.path.append(project_path + "/models")
sys.path.append(project_path + "/main")
import datetime
import uuid
from argparse import ArgumentParser
import setproctitle
import torch
from utils.args import add_management_args, add_experiment_args
from utils import create_if_not_exists
# from utils.continual_training import train as ctrain
from run import *
from accelerate.utils import set_seed
from accelerate import Accelerator
import wandb
from peft.tuners.lora import LoraLayer, Linear
from lorawrappers import *
from tqdm import tqdm, trange
import math
import torch
import torch.nn.functional as F
from torch.optim import SGD, Adam, AdamW
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer
from peft import get_peft_model, LoraConfig, PeftModel, PeftConfig
from peft.tuners.lora import LoraLayer, Linear
from schedulers import BLoBKLScheduler, BLoBNLLScheduler
from torchmetrics.functional import calibration_error
from dataset.S2SDataset_Classification import S2SDataset_Classification

def modify_lora_layers(module):
    """
    Recursively go through the model and modify LoraLayer instances.
    """
    for name, child in module.named_children():
        if isinstance(child, LoraLayer) and isinstance(child, Linear):
            module._modules[name] = ScalablLoraWrapper(child, bayes_eps=0.05)
        else:
            modify_lora_layers(child)

def lecun_fix():
    # Yann moved his website to CloudFlare. You need this now
    from six.moves import urllib  # pyright: ignore

    opener = urllib.request.build_opener()
    opener.addheaders = [("User-agent", "Mozilla/5.0")]
    urllib.request.install_opener(opener)


def parse_args():
    parser = ArgumentParser(description="Bayesian LoRA", allow_abbrev=False)
    add_management_args(parser)
    add_experiment_args(parser)
    args = parser.parse_known_args()[0]

    # add model-specific arguments
    mod = importlib.import_module("modelwrappers." + args.modelwrapper)
    get_parser = getattr(mod, "get_parser")
    parser = get_parser()  # the real parsing happens.
    args = parser.parse_args()

    # set random seed
    if args.seed is not None:
        set_seed(args.seed)

    return args


def main(args=None):
    lecun_fix()
    if args is None:
        args = parse_args()

    os.putenv("MKL_SERVICE_FORCE_INTEL", "1")
    os.putenv("NPY_MKL_FORCE_INTEL", "1")
    args.conf_jobnum = str(uuid.uuid4())
    args.conf_timestamp = str(datetime.datetime.now())
    args.conf_host = socket.gethostname()
    # dataset = get_dataset(args.dataset_type, args)
    # dataset.get_loaders()
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True
    )
    
    dataset = S2SDataset_Classification(
        tokenizer,
        dataset=args.dataset,
        testing_set=args.testing_set,
        add_space=args.add_space,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        is_s2s=args.is_s2s,
    )
    dataset.get_loaders()



    args.outdim = dataset.num_labels
    args.num_samples = dataset.num_samples
    setproctitle.setproctitle("{}_{}_BLoB-lora".format(args.model, args.dataset))
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        config=vars(args),
        name=args.wandb_name
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model, 
        quantization_config=None,
        device_map='auto', 
        torch_dtype=torch.bfloat16
    )
    device = model.device

    if args.apply_classhead_lora:
        target_modules = ["q_proj", "v_proj", "lm_head"]
    elif args.apply_qkv_head_lora:
        target_modules = ["q_proj", "v_proj", "k_proj", "lm_head"]
    else:
        target_modules = ["q_proj", "v_proj"]

    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        inference_mode=False,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
    )
    model = get_peft_model(model, peft_config)
    modify_lora_layers(model)


    save_folder = f"checkpoints/{args.modelwrapper}/{args.model}/{args.dataset}/{args.checkpoint_name}"
    model.load_adapter(save_folder, "default")
    model = model.to(device)

    test_loader = dataset.test_dataloader
    # if args.testing_set == "train_val":
        # val_loader = dataset.val_dataloader
        # val_loader = accelerator.prepare(val_loader)
        # val_loader = val_loader

    if args.dataset_type == "mcdataset":
        target_ids = dataset.target_ids.squeeze(-1)

    model.eval()
    test_probs, test_labels = [], []
    for step, batch in enumerate(tqdm(test_loader, desc="Testing")):
        batch = [b.to(device) for b in batch]
        inputs, labels, _ = batch
        
        logits = []
        for i in range(args.bayes_eval_n_samples_final):
            with torch.no_grad():
                logits_i = model(**inputs).logits[:, -1, target_ids]
            logits.append(logits_i)
        logits = torch.stack(logits, dim=1)
        probs = torch.softmax(logits, dim=-1).mean(dim=1)
        test_probs.append(probs)
        test_labels.append(labels)
    test_probs = torch.cat(test_probs, dim=0)
    test_logprobs = torch.log(test_probs)
    test_preds = test_probs.argmax(dim=-1)
    test_labels = torch.cat(test_labels, dim=0)
    test_acc = (test_preds == test_labels).float().mean().item()
    test_nll = F.nll_loss(test_logprobs, test_labels, reduction="mean").item()
    test_ece = calibration_error(
        test_probs, test_labels, 
        task='multiclass', 
        num_classes=len(target_ids),
        n_bins=15
    ).item()
    wandb.log({
        "test/acc": test_acc,
        "test/nll": test_nll,
        "test/ece": test_ece,
    })
    print(f"Test Acc: {test_acc}, Test ECE: {test_ece}, Test NLL: {test_nll}")
    
    wandb.finish()


if __name__ == "__main__":
    main()
