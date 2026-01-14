import torch
from bayesadapt.laplace import Laplace
from .trainer import Trainer

class WrappedModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, **kwargs):
        kwargs.pop('labels', None)
        logits = self.model(**kwargs).logits
        return logits[:, -1, :]

class LaplaceTrainer(Trainer):
    @property
    def wrapper_name(self):
        return 'laplace'

    def save_model(self):
        sd = self.model.state_dict()
        sd = {k: v for k, v in sd.items() if 'lora_' in k}
        torch.save(sd, os.path.join(self.expdir, "state_dict.pt"))
        print(f"Model saved to {self.expdir}")
    
    def compute_logits(self, inputs):
        samples = 100000
        with torch.no_grad():
            f_mu, f_var = self.la._glm_predictive_distribution(inputs)
        f_mu = f_mu.expand(samples, -1, -1)
        f_var = f_var.expand(samples, -1, -1, -1)
        eye = torch.eye(f_var.shape[-1], device=f_var.device)
        stabilized_var = f_var + (eye * 1e-6)
        L = torch.linalg.cholesky(stabilized_var).to(f_mu.dtype)
        noise = torch.randn_like(f_mu).unsqueeze(-1)
        perturbation = (L @ noise).squeeze(-1)
        logits = f_mu + perturbation
        logits = logits.permute(1, 0, 2)  # (batch_size, num_samples, num_classes)
        return logits.detach()
        # sample_probs = torch.softmax(logits, dim=-1)
        # probs = sample_probs.mean(dim=0)


    def evaluate_step(self, batch):
        inputs, labels = batch
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.reset_peak_memory_stats(self.device)
        start_event.record()
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

    
    def evaluate(self):
        self.model.eval()
        self.model = WrappedModel(self.model)
        self.la = Laplace(
            self.model, 
            'classification', 
            prior_precision=self.cfg.optim.prior_precision,
            subset_of_weights=self.cfg.optim.subset_of_weights,
            hessian_structure=self.cfg.optim.hessian_structure,
        )
        self.la.fit(self.trainloader)
        prior_precision = self.la.optimize_prior_precision(
            method=self.cfg.optim.method,
            n_steps=self.cfg.optim.n_steps,
            lr=self.cfg.optim.lr,
        )
        super().evaluate()
        # self.save_model()
