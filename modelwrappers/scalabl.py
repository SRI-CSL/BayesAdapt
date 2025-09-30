import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Any, List, Optional, Union
from tqdm import tqdm
from .wrapperbase import WrapperBase, optimizer_dict
from utils.args import add_management_args, add_experiment_args, ArgumentParser
from run.evaluation import *
from transformers import PreTrainedModel
from peft.config import PeftConfig
from peft.tuners.lora import LoraLayer, Linear
# from scalabl_layers import ScalaBLLinear, ScalablLoraWrapper
# from blob_layers import BlobLinear, BlobLoraWrapper
from schedulers import BLoBKLScheduler, BLoBNLLScheduler
from lorawrappers import BlobLoraWrapper, ScalablLoraWrapper, VILoraWrapper, MCDropoutLoraWrapper, DeepEnsembleLoraWrapper, LoraWrapper, SVDLoraWrapper

## Model Specific Argument Parsing
def get_parser() -> ArgumentParser:
    parser = ArgumentParser(description="Bayesian By Backprop, ScalaBL.")
    add_management_args(parser)
    add_experiment_args(parser)
    # ScalaBL-specific arguments.
    parser.add_argument("--bayes-train-n-samples", type=int, default=1)
    parser.add_argument(
        "--bayes-eval-n-samples",
        type=int,
        default=1,
        help="Number of samples to use for evaluation during training.",
    )
    parser.add_argument(
        "--bayes-eval-n-samples-final",
        type=int,
        default=10,
        help="Number of samples to use for evaluation.",
    )

    parser.add_argument("--bayes-eps", type=float, default=0.05)
    parser.add_argument("--bayes-gamma", type=float, default=8)
    parser.add_argument("--bayes-kllr", type=float, default=0.02)
    parser.add_argument("--bayes-beta", type=float, default=0.2)
    parser.add_argument(
        "--bayes-inference-notsample",
        action="store_true",
        help="Whether to sample during inference.",
    )
    parser.add_argument(
        "--bayes-klreweighting", action="store_true", help="Whether to use reweighting."
    )
    parser.add_argument('--bayes-datasetrescaling', action='store_true',
                        help='Whether to use datasetrescaling.')

    return parser

@dataclass
class ScalaBLConfig:
    bayes_eps: float = field(metadata={"help": "Bayes epsilon"})
    bayes_gamma: float = field(metadata={"help": "Bayes gamma"})
    bayes_beta: float = field(metadata={"help": "Bayes beta"})

class ScalaBL(WrapperBase):
    """ScalaBL model."""

    def __init__(
        self,
        model: PreTrainedModel,
        peft_config: PeftConfig,
        args,
        accelerator,
        adapter_name: str = "default",
    ):
        super().__init__(model, peft_config, args, accelerator, adapter_name)

        self.blobconfig = ScalaBLConfig(
            bayes_eps=self.args.bayes_eps,
            bayes_gamma=self.args.bayes_gamma,
            bayes_beta=self.args.bayes_beta,
        )
        self._modify_lora_layers(self.base_model)
        if args.load_lora_path is not None:
            self.load_adapter(args.load_lora_path, adapter_name)

        self.train_n_samples = self.args.bayes_train_n_samples
        self.eval_n_samples = self.args.bayes_eval_n_samples
        # self.klreweighting = self.args.bayes_klreweighting

    def _modify_lora_layers(self, module):
        """
        Recursively go through the model and modify LoraLayer instances.
        """
        for name, child in module.named_children():
            if isinstance(child, LoraLayer) and isinstance(child, Linear):
                module._modules[name] = ScalablLoraWrapper(child, bayes_eps=self.blobconfig.bayes_eps)
                #module._modules[name] = MCDropoutLoraWrapper(child, bayes_eps=self.blobconfig.bayes_eps)
                #module._modules[name] = MCDropoutLoraWrapper(child)
                #module._modules[name] = DeepEnsembleLoraWrapper(child)
                # module._modules[name] = LoraWrapper(child)
            else:
                self._modify_lora_layers(child)

    def forward_logits(self, batch, sample=True, n_samples=1, **kwargs) -> torch.Tensor:
        if self.args.dataset_type == "mcdataset":
            inputs, _, _ = batch
            if not sample:
                self.sample(self.base_model, False)
                output = self.base_model(**inputs)
                logits = output.logits[:, -1, self.target_ids]
                self.sample(self.base_model, True)
                return logits
            else:
                logits_list = []
                for _ in range(n_samples):
                    inputs = inputs.to(self.device)
                    output = self.base_model(**inputs)
                    logits = output.logits[:, -1, self.target_ids]
                    logits_list.append(logits)
                return torch.stack(logits_list, dim=1)
        else:
            if not sample:
                self.sample(self.base_model, False)
                res = self.base_model(**batch).logits
                self.sample(self.base_model, True)
                return res
            else:
                res = []
                for _ in range(n_samples):
                    res.append(self.base_model(**batch).logits)
                return torch.stack(res, dim=1)

    def fit(self, train_loader, eval_loader):
        # nll_losses = AverageMeter()
        # kl_losses = AverageMeter()
        # elbo_losses = AverageMeter()
        # accs = AverageMeter()
        samples_seen = 0
        with tqdm(
            total=len(train_loader),
            desc=f"Epoch {self.args.epoch+1}/{self.args.n_epochs}",
            leave=False,
        ) as pbar:
            for i, batch in enumerate(train_loader):
                if self.args.dataset_type == "mcdataset":
                    _, golds, _ = batch
                elif self.args.dataset_type == "bertds":
                    golds = batch["labels"]
                else:
                    raise NotImplementedError(
                        f"Dataset type {self.args.dataset_type} not implemented."
                    )
                logits = self.forward_logits(
                    batch, sample=True, n_samples=self.train_n_samples
                )
                B, num_samples, num_classes = logits.shape
                golds = golds.unsqueeze(-1).expand(B, num_samples)
                log_probs = torch.log_softmax(logits, dim=-1)
                nll = self.loss(
                    log_probs.view(B * num_samples, num_classes),
                    golds.reshape(B * num_samples),
                    reduction="none"
                ).reshape(B, num_samples)
                # logits = logits.mean(dim=1)
                # output = torch.log_softmax(logits, dim=1)
                # num_classes = output.size(1)
                # nll = self.loss(output, golds, reduction="mean")
                nll = nll.mean()

                acc = (log_probs.argmax(dim=-1) == golds).float().mean()
                
                self.accelerator.backward(nll)
                self.nll_optimizer.step()
                self.nll_optimizer.zero_grad()
                self.nll_scheduler.step()

                kl_divs = []
                for module in self.base_model.modules():
                    if isinstance(module, VILoraWrapper):
                        kl_divs.append(module.kl_div)

                if len(kl_divs) > 0:
                    kl = torch.sum(torch.stack(kl_divs), dim=0)
                    self.pi = self.kl_scheduler.scheduler.last_pi
                    self.accelerator.backward(kl)
                    self.kl_optimizer.step()
                    self.kl_optimizer.zero_grad()
                    self.kl_scheduler.step()
                else:
                    kl = torch.tensor(0.0)
                    self.pi = 0.0

                # acc = accuracy_topk(output.data, golds)

                loss, acc, nll_loss, kl = (
                    (kl + nll).detach().float().cpu().numpy(),
                    acc.item(),
                    nll.detach().float().cpu().numpy(),
                    kl.detach().float().cpu().numpy() * self.pi,
                )

                if self.args.dataset_type == "mcdataset":
                    _, classes, _ = batch
                    references = self.accelerator.gather(classes)
                else:
                    references = self.accelerator.gather(batch["labels"])
                if self.accelerator.num_processes > 1:
                    if i == len(train_loader) - 1:
                        references = references[
                            : len(train_loader.dataset) - samples_seen
                        ]
                    else:
                        samples_seen += references.shape[0]
                len_batch = references.shape[0]
                # kl_losses.update(kl, len_batch)
                # nll_losses.update(nll_loss, len_batch)
                # elbo_losses.update(loss, len_batch)
                # accs.update(acc, len_batch)

                assert not math.isnan(nll_loss)
                assert not math.isnan(kl)
                if self.accelerator.is_local_main_process:
                    if self.wandb_logger is not None:
                        self.wandb_logger.log({
                            "train_acc": acc,
                            "train_nll_loss": nll_loss,
                            "kl_loss": kl,
                            "elbo_loss": loss,
                            "lr": self.opt.param_groups[0]["lr"],
                            "pi": self.pi,
                        })

                self.step += self.accelerator.num_processes
                pbar.update(1)
                if self.step >= self.args.eval_per_steps:
                    self.step -= self.args.eval_per_steps
                    self.evaluate(eval_loader)

    def evaluate(self, eval_loader):
        print("self.eval_n_samples:", self.eval_n_samples)
        self.eval()
        status = self.training
        nlls = AverageMeter()
        metric_kwargs = {"task": "multiclass", "num_classes": self.num_classes}
        acc_metric = Accuracy(**metric_kwargs).to(self.accelerator.device)
        ece_metric = CalibrationError(**metric_kwargs, n_bins=self.args.num_bins).to(
            self.accelerator.device
        )
        briers = AverageMeter()

        samples_seen = 0
        for step, batch in enumerate(tqdm(eval_loader, desc="Evaluating")):
            with torch.no_grad() and torch.inference_mode():
                logits = self.forward_logits(
                    batch,
                    sample=not self.args.bayes_inference_notsample,
                    n_samples=self.eval_n_samples,
                ).detach()
                if self.args.dataset_type == "mcdataset":
                    _, labels, _ = batch
                else:
                    labels = batch["labels"]
                logits, labels = self.accelerator.gather([logits, labels])
                if self.accelerator.num_processes > 1:
                    if step == len(eval_loader) - 1:
                        labels = labels[: len(eval_loader.dataset) - samples_seen]
                        logits = logits[: len(eval_loader.dataset) - samples_seen]
                    else:
                        samples_seen += labels.shape[0]
                probs = torch.softmax(logits, dim=-1).mean(dim=1)
                std = torch.softmax(logits, dim=-1).std(dim=1).mean()

                acc_metric(probs, labels)
                ece_metric(probs, labels)
                nll = self.loss(torch.log(probs), labels, reduction="mean")
                if torch.isnan(nll):
                    if self.accelerator.is_local_main_process:
                        print("nll:", nll)
                        print("probs:", probs)
                        print("logits:", logits)
                        exit()
                nlls.update(nll)

                brier = (
                    (probs - F.one_hot(labels, num_classes=logits.size(-1)))
                    .pow(2)
                    .sum(dim=-1)
                    .mean()
                )
                briers.update(brier)

        val_acc = acc_metric.compute().item()
        val_ece = ece_metric.compute().item()
        val_nll = nlls.avg
        val_brier = briers.avg
        self.train(status)

        if self.accelerator.is_local_main_process:
            if self.wandb_logger is not None:
                self.wandb_logger.log(
                    {
                        "val_acc": val_acc,
                        "val_ece": val_ece,
                        "val_nll": val_nll,
                        "std": std,
                        "val_brier": val_brier,
                    }
                )
        return val_acc, val_ece, val_nll, val_brier


    def prepare_for_fit_evaluate(self, dataset, wandb_logger=None):
        """
        Prepare the model for training and evaluation.
        """
        self.wandb_logger = wandb_logger
        train_loader, test_loader = dataset.train_dataloader, dataset.test_dataloader
        # assert self.args.max_train_steps != 0 and self.args.n_epochs == 0
        warmup_steps = int(self.args.max_train_steps * self.args.warmup_ratio)
        num_update_steps_per_epoch = len(train_loader)
        self.args.n_epochs = (
            math.ceil(self.args.max_train_steps / num_update_steps_per_epoch)
            if self.args.ood_ori_dataset is None else 0
        )

        no_decay = ["bias", "LayerNorm.weight"]
        decay_params, no_decay_params = [], []
        for n, p in self.named_parameters():
            if p.requires_grad:
                if any(nd in n for nd in no_decay):
                    no_decay_params.append(p)
                else:
                    decay_params.append(p)
        
        self.nll_optimizer = optimizer_dict[self.args.opt](
            params=[
                {"params": decay_params, "weight_decay": self.args.opt_wd},
                {"params": no_decay_params, "weight_decay": 0.0},
            ],
            lr=self.args.lr,
            eps=self.args.adam_epsilon, 
        )

        self.nll_scheduler = BLoBNLLScheduler(
            self.nll_optimizer,
            warmup_steps=warmup_steps,
            total_steps=self.args.max_train_steps,
        )
        
        self.kl_optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.args.bayes_kllr
        )
        self.kl_scheduler = BLoBKLScheduler(
            self.kl_optimizer,
            warmup_steps=warmup_steps,
            total_steps=self.args.max_train_steps,
            num_samples=dataset.num_samples,
            batch_size=self.args.batch_size,
            gamma=self.args.bayes_gamma,
            use_exponential=self.args.bayes_klreweighting,
        )

        if self.args.testing_set == "train_val":
            val_loader = dataset.val_dataloader
            val_loader = self.accelerator.prepare(val_loader)
            self.val_loader = val_loader

        if self.args.dataset_type == "mcdataset":
            self.target_ids = dataset.target_ids.squeeze(-1)

        if self.args.early_stop_steps > 0:
            self.earlystop_n_epochs = (
                math.ceil(self.args.early_stop_steps / num_update_steps_per_epoch)
                if self.args.ood_ori_dataset is None
                else 0
            )
        else:
            self.earlystop_n_epochs = 0
        if self.accelerator.is_local_main_process:
            print("len(train_loader):", len(train_loader))
            print("num of epochs:", self.args.n_epochs)
        self.step = 0

        self.base_model = self.base_model.to(self.device)
        (
            self.base_model,
            self.nll_optimizer,
            train_loader,
            test_loader,
            self.nll_scheduler,
            self.kl_scheduler,
            self.kl_optimizer,
        ) = self.accelerator.prepare(
            self.base_model,
            self.nll_optimizer,
            train_loader,
            test_loader,
            self.nll_scheduler,
            self.kl_scheduler,
            self.kl_optimizer,
        )

        self.train_loader = train_loader
        self.test_loader = test_loader

        # for n, p in self.named_parameters():
            # if p.requires_grad:
                # print(n, p.shape)
