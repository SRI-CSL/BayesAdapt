import os
import math
import torch
import torch.nn.functional as F
from hydra.utils import instantiate
from .trainer import Trainer

class IVONTrainer(Trainer):

    @property
    def wrapper_name(self):
        return 'ivon'

    def load_optimizer(self):
        params = [p for p in self.model.parameters() if p.requires_grad] #needed for IVON to work
        ess = 5000*math.sqrt(len(self.trainloader.dataset))
        self.nll_optimizer = instantiate(self.cfg.optim.nll_optimizer, params, ess=ess)
        self.nll_scheduler = instantiate(self.cfg.optim.nll_scheduler, self.nll_optimizer)

        if self.cfg.load_pretrained_checkpoint:
            optimizer_path = os.path.join(self.cfg.expdir, 'optimizer.pt')
            if os.path.exists(optimizer_path):
                optimizer_sd = torch.load(optimizer_path, map_location='cpu')
                self.nll_optimizer.load_state_dict(optimizer_sd['nll_optimizer'])
                self.nll_scheduler.load_state_dict(optimizer_sd['nll_scheduler'])
                self.nll_optimizer.current_step = optimizer_sd['current_step']
                print(f"Loaded optimizer state from {optimizer_path}")
            else:
                print(f"No optimizer state found at {optimizer_path}, initializing new optimizer.")

    
    def save_model(self):
        super().save_model()
        optimizer_sd = {
            'nll_optimizer': self.nll_optimizer.state_dict(),
            'nll_scheduler': self.nll_scheduler.state_dict(),
            'current_step': self.nll_optimizer.current_step,
        }
        torch.save(optimizer_sd, os.path.join(self.expdir, 'optimizer.pt'))

    def evaluate_step(self, batch):
        inputs, labels = batch
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.reset_peak_memory_stats(self.device)
        start_event.record()
        with torch.no_grad() and torch.inference_mode():
            with self.nll_optimizer.sampled_params(train=False):
                logits = self.compute_logits(inputs)
        end_event.record()
        torch.cuda.synchronize()
        peak_memory = torch.cuda.max_memory_allocated(self.device)
        elapsed_time = start_event.elapsed_time(end_event)
        return {
            'logits': logits,
            'elapsed_time': elapsed_time,
            'peak_memory': peak_memory,
        }


    def train_step(self, batch):
        self.nll_optimizer.zero_grad()
        inputs, labels = batch

        with self.nll_optimizer.sampled_params(train=True):
            logits = self.compute_logits(inputs)
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

        self.nll_optimizer.step()
        self.nll_scheduler.step()

        log = {
            'train/nll_loss': nll_loss.item(),
            'train/acc': acc.item(),
            'train/nll_lr': self.nll_optimizer.param_groups[0]["lr"],
        }
        return log
