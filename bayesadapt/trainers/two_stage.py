from .trainer import Trainer

class TwoStageTrainer(Trainer):
    @property
    def model(self):
        if self._model is None:
            self.load_model()
            if 'lora' in self.cfg:
                self.load_lora()
                self.wrapped = False
        return self._model


    def train(self, save=True, use_wandb=True, validation_dataset=None):
        self.model.train()
        if not self.wrapped: #train unwrapped MLE first
            super().train(save=False, use_wandb=False) 
            for name, param in self._model.named_parameters():
                if param.requires_grad: 
                    param.requires_grad = False 
            self.wrap_lora_layers()

        self.update_dataloaders(train_dataset=validation_dataset) 
        self.cfg.optim.nll_optimizer.lr = 1e-3
        super().train(save=save, use_wandb=use_wandb)
