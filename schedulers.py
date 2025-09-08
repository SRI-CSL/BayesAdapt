import math
import torch
from torch.optim.lr_scheduler import _LRScheduler

class BLoBNLLScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        last_epoch: int = -1,
    ):
        self.warmup_steps = max(1, warmup_steps)
        self.total_steps = max(total_steps, self.warmup_steps + 1)
        super().__init__(optimizer, last_epoch)

    # ------------------------------------------------------------------ #
    def get_lr(self):
        step = self.last_epoch + 1  # PyTorch updates last_epoch *before* get_lr
        if step <= self.warmup_steps:
            factor = step / self.warmup_steps
        else:
            factor = max(
                0.0,
                (self.total_steps- step)
                / (self.total_steps - self.warmup_steps),
            )
        return [base_lr * factor for base_lr in self.base_lrs]

#implementation of scheduler from BLoB paper
class BLoBKLScheduler(_LRScheduler):
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 warmup_steps: int,
                 total_steps: int,
                 num_samples: int, #size of dataset
                 batch_size: int,
                 gamma: float,
                 use_exponential: bool = True,
                 last_epoch: int = -1):
        self.warmup_steps   = warmup_steps
        self.total_steps    = total_steps
        self.M = int(100 * (num_samples ** (math.pi / gamma)) / batch_size)

        self.use_exponential = use_exponential
        self._denom         = 2 ** (self.M + 1) - 1
        self.last_pi        = 0.0           # updated every .step()
        super().__init__(optimizer, last_epoch)

    # ---------------------------------------------------------------------
    def get_lr(self):
        step = self.last_epoch
        # 1) linear warm-up → linear decay
        if step < self.warmup_steps:
            lin = step / max(1, self.warmup_steps)
        else:
            lin = max(
                0.0,
                (self.total_steps - step) / max(1, self.total_steps - self.warmup_steps)
            )
        # 2) π_i
        if self.use_exponential:
            i  = (step % self.M) + 1            # 1 … M
            pi = 2 ** i / self._denom
        else:
            pi = 1.0 / self.M
        self.last_pi = pi                      # expose for logging
        scale = lin * pi
        return [base_lr * scale for base_lr in self.base_lrs]
