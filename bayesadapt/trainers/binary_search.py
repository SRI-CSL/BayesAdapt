from bayesadapt.lorawrappers import TFBLoraWrapper, VILoraWrapper
from .trainer import Trainer

def set_cov(model, beta):
    for module in model.modules():
        if isinstance(module, TFBLoraWrapper):
            module.set_cov(beta)

class BinarySearcher(Trainer):
    def evaluate(self):
        self.model.eval()
        set_cov(self.model, beta=0.0)
        orig_nll = super().evaluate(use_train=True, save=False)[0]['NLL']
        low, high = self.cfg.optim.low_start, self.cfg.optim.high_start
        best = high  
        for _ in range(self.cfg.optim.max_train_steps):
            mid = (low + high) / 2
            set_cov(self.model, beta=mid)
            new_nll = super().evaluate(use_train=True, save=False)[0]['NLL']
            print(f"Testing beta: {mid}, NLL: {new_nll}")

            #loss_change_ratio = (abs(current_nll_loss.item() - ori_nll_loss.item()) / ori_nll_loss.item())/self.all_ori_predicted_classes.size(0)
            
            ratio = abs(new_nll - orig_nll) / orig_nll / len(self.trainloader.dataset)
            print(f"Loss change ratio: {ratio}")

            if ratio > self.cfg.optim.target_ratio:
                best = mid
                high = mid
            else:
                low = mid
        
        print(f"Best beta found: {best}")
        set_cov(self.model, beta=best)
        return super().evaluate(use_train=False, save=True)
