import os
import torch
from bayesadapt.lorawrappers import TFBLoraWrapper
from .trainer import Trainer

def set_cov(model, beta):
    for module in model.modules():
        if isinstance(module, TFBLoraWrapper):
            module.set_cov(beta)

class BinarySearcher(Trainer):
    @property
    def param_counts(self):
        counts = super().param_counts
        counts['num_trainable_params'] += 1  # for beta
        counts['num_total_params'] += 1
        return counts

    @property
    def model(self):
        if self._model is None:
            self.load_model()
            if 'lora' in self.cfg:
                self.load_lora()
                self.wrapped = False
                # if 'wrapper' in self.cfg.lora:
                    # self.wrap_lora_layers()

            # if self.cfg.load_pretrained_checkpoint:
                # checkpoint_path = os.path.join(self.expdir, "state_dict.pt")
                # sd = torch.load(checkpoint_path, map_location='cpu')
                # self._model.load_state_dict(sd, strict=False)
                # print('model loaded from', checkpoint_path)
            
            # params_info_path = os.path.join(self.expdir, "param_counts.json")
            # with open(params_info_path, "w") as f:
                # json.dump(self.param_counts, f)
        
        return self._model


    def load_model(self):
        super().load_model()
        sd_path = os.path.join(self.expdir, "state_dict.pt")
        if os.path.exists(sd_path):
            sd = torch.load(sd_path, map_location='cpu')
            self.beta = sd['beta']
        else:
            self.beta = 0.0  # default value
            print("No saved state dict found. Initializing beta to 0.0")
    

    def save_model(self):
        sd = {'beta': self.beta}
        torch.save(sd, os.path.join(self.expdir, "state_dict.pt"))
        print(f"Saved state dict with beta: {self.beta}")

    def train(self, save=True, use_wandb=False):
        if not self.cfg.lora.load_mle_checkpoint:
            super().train(save=save, use_wandb=use_wandb)
        if not self.wrapped:
            self.wrap_lora_layers()
            self.wrapped = True

        self.model.eval()
        set_cov(self.model, beta=0.0)
        # orig_nll = super().evaluate(use_train=True, save=False)[0]['NLL']
        metrics, logits = super().evaluate(use_train=True, save=False)
        orig_nll = metrics[0]['NLL']
        print(f"Original NLL with beta=0.0: {orig_nll}")
        low, high = self.cfg.optim.low_start, self.cfg.optim.high_start
        best = high  
        for t in range(5):
            mid = (low + high) / 2
            print(t, low, high, mid)
            set_cov(self.model, beta=mid)
            metrics, logits = super().evaluate(use_train=True, save=False)
            new_nll = metrics[0]['NLL']

            #loss_change_ratio = (abs(current_nll_loss.item() - ori_nll_loss.item()) / ori_nll_loss.item())/self.all_ori_predicted_classes.size(0)
            
            ratio = abs(new_nll - orig_nll) / orig_nll #/ len(self.trainloader.dataset)
            print(ratio)
        
            if ratio > self.cfg.optim.target_ratio:
                best = mid
                high = mid
            else:
                low = mid

        
        self.beta = best
        if save:
            self.save_model()
        # print(f"Best beta found: {best}")
        # set_cov(self.model, beta=best)
        # return super().evaluate(use_train=False, save=True)

    def evaluate(self, **kwargs):
        set_cov(self.model, beta=self.beta)
        return super().evaluate(**kwargs)
