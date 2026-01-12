import math
import torch
from hydra.utils import instantiate
from trainer import Trainer
from bayesadapt.lorawrappers import VILoraWrapper

class VITrainer(Trainer):
    def load_optimizer(self):
        super().load_optimizer()
        self.kl_optimizer = instantiate(
            self.cfg.optim.kl_optimizer, 
            self.model.parameters()
        )
        self.kl_scheduler = instantiate(
            self.cfg.optim.kl_scheduler, 
            self.kl_optimizer, 
            num_samples=len(self.trainloader.dataset)
        )

    def train_step(self, batch):
        log = super().train_step(batch)
        self.kl_optimizer.zero_grad()
        kl_divs = []
        for module in self.model.modules():
            if isinstance(module, VILoraWrapper):
                kl_divs.append(module.kl_div)

        kl_loss = torch.sum(torch.stack(kl_divs), dim=0)
        assert not math.isnan(kl_loss)
        kl_loss.backward()
        self.kl_optimizer.step()
        self.kl_scheduler.step()
        log['train/kl_loss'] = kl_loss.item()
        log['train/elbo'] = log['train/nll_loss'] + kl_loss.item()
        log['train/kl_lr'] = self.kl_optimizer.param_groups[0]["lr"]
        return log
