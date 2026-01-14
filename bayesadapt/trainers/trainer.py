import os
import numpy  # needed (don't change it)
import math
import json
import torch
import torch.nn.functional as F
import wandb
import hydra
from omegaconf import OmegaConf
from hydra.utils import instantiate
from tqdm import tqdm, trange
from accelerate.utils import set_seed
from transformers import AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig, AutoProcessor
from bayesadapt.utils import load_model, split_batch, infer_logdir_from_cfg, load_dataloader, average_log_probs
from bayesadapt.lorawrappers.utils import wrap_lora_layers 
from bayesadapt.lorawrappers import VILoraWrapper
from torch.utils.data import DataLoader
from hydra.core.hydra_config import HydraConfig
from torchmetrics.functional import calibration_error, accuracy
from peft import get_peft_model

def cycle(dataloader):
    while True:
        for data in dataloader:
            yield data

class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.trainset_name = HydraConfig.get().runtime.choices['dataset@train_dataset']
        self.testset_name = HydraConfig.get().runtime.choices['dataset@test_dataset']
        set_seed(cfg.seed)
        os.makedirs(self.expdir, exist_ok=True)
        yaml_str = OmegaConf.to_yaml(cfg)
        with open(os.path.join(self.expdir, "config.yaml"), "w") as f:
            f.write(yaml_str)
        
        self.device = torch.device(f"cuda:{cfg.gpu_id}" if torch.cuda.is_available() else "cpu")
        self.load_processor()
        self.load_dataloaders()
        if hasattr(self.processor, 'tokenizer'):
            self.class_ids = self.processor.tokenizer.convert_tokens_to_ids(self.trainloader.dataset.labels)
        else:
            self.class_ids = self.processor.convert_tokens_to_ids(self.trainloader.dataset.labels)
        self.load_model()
        if 'lora' in self.cfg:
            self.load_lora()
            if 'wrapper' in self.cfg.lora:
                self.wrap_lora_layers()

        if self.cfg.load_pretrained_checkpoint:
            checkpoint_path = os.path.join(self.expdir, "state_dict.pt")
            sd = torch.load(checkpoint_path, map_location='cpu')
            self.model.load_state_dict(sd, strict=False)
            print('model loaded from', checkpoint_path)

        params_info_path = os.path.join(self.expdir, "param_counts.json")
        with open(params_info_path, "w") as f:
            json.dump(self.param_counts, f)

    @property
    def wrapper_name(self):
       if 'lora' in self.cfg:
           if 'wrapper' in self.cfg.lora:
               wrapper_name = HydraConfig.get().runtime.choices['lora/wrapper']
           else: #standard lora, no wrapper
               wrapper_name = 'mle' if self.cfg.optim.weight_decay == 0 else 'map'
       else:
           wrapper_name = 'zeroshot'
       return wrapper_name

    @property
    def expdir(self):
        r = self.cfg.lora.config.r if 'lora' in self.cfg else 0
        # train_dataset = HydraConfig.get().runtime.choices['dataset@train_dataset']
        # test_dataset = HydraConfig.get().runtime.choices['dataset@test_dataset']
        expdir = os.path.join(
            'logs',
            self.cfg.hf_model,
            f"{self.cfg.quantize_bits}bit",
            self.wrapper_name,
            f"rank{r}",
            HydraConfig.get().runtime.choices.collate_fn,
            f"seed{self.cfg.seed}",
            self.trainset_name,
        )
        return expdir

    @property
    def quantization_config(self):
        if self.cfg.quantize_bits == 4:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        elif self.cfg.quantize_bits == 8:
            return BitsAndBytesConfig(load_in_8bit=True)
        elif self.cfg.quantize_bits == 16:
            return None
        else:
            raise ValueError(f"Unsupported quantization bits: {cfg.quantize_bits}")
    
    @property
    def torch_dtype(self):
        if self.cfg.quantize_bits == 4 or self.cfg.quantize_bits == 16:
            return torch.bfloat16
        elif self.cfg.quantize_bits == 8:
            return torch.float16
        else:
            raise ValueError(f"Unsupported quantization bits: {self.cfg.quantize_bits}")

    @property
    def model_class(self):
        if 'Qwen3-VL' in self.cfg.hf_model:
            from transformers import Qwen3VLForConditionalGeneration as model_class
        elif 'gemma-3' in self.cfg.hf_model:
            from transformers import Gemma3ForConditionalGeneration as model_class
        else:
            model_class = AutoModelForCausalLM
        return model_class


    def load_model(self):
        self.model_config = AutoConfig.from_pretrained(self.cfg.hf_model)
        self.model = self.model_class.from_pretrained(
            self.cfg.hf_model, 
            quantization_config=self.quantization_config,
            device_map=self.device,
            dtype=self.torch_dtype,
            tie_word_embeddings=False,
        )
        #some models share the weights between the embedding layer and lm_head
        #this is typically done to save memory for small models (i.e. Qwen2.5-0.5B)
        #so we explicitly untie the weights and copy the embedding weights to the lm_head
        #this ensures that only the last layer is stochastic during VI approaches
        if self.model_config.tie_word_embeddings:
            embed_weights = self.model.get_input_embeddings().weight.detach().clone()
            sd = {'weight': embed_weights}
            self.model.lm_head.load_state_dict(sd)
            self.model.config.tie_word_embeddings = False
            print("copied embedding weights to lm_head weights")

        #following the approach of Laplace LoRA
        #we only keep the classifier weights for the target classes
        if self.class_ids is not None:
            classifier_weights = self.model.lm_head.weight.detach().clone()
            new_head = torch.nn.Linear(
                in_features=classifier_weights.shape[1],
                out_features=len(self.class_ids),
                bias=False,
                dtype=classifier_weights.dtype,
                device=classifier_weights.device,
            )
            selected_weights = classifier_weights[self.class_ids]
            sd = {'weight': selected_weights}
            new_head.load_state_dict(sd)
            self.model.register_module('lm_head', new_head)
            self.model.config.vocab_size = len(self.class_ids)

    def load_lora(self):
        self.peft_config = instantiate(self.cfg.lora.config, lora_alpha=self.cfg.lora.config.r*2)
        self.peft_config.target_modules = list(self.peft_config.target_modules) #make sure it's a list, otherwise save_pretrained fails
        self.model = get_peft_model(self.model, self.peft_config)
        
        #some methods like Laplace or TFB require starting from a pretrained MLE (unwrapped) LoRA checkpoint
        #we assume that the dir name is the name with the wrapper name replaced by 'mle'
        if self.cfg.lora.load_mle_checkpoint:
            mle_dir = self.expdir.replace(self.wrapper_name, 'mle')
            assert os.path.exists(mle_dir), f"Checkpoint dir {mle_dir} does not exist"
            mle_checkpoint = os.path.join(mle_dir, "state_dict.pt")
            sd = torch.load(mle_checkpoint, map_location='cpu')
            self.model.load_state_dict(sd, strict=False)
            print('unwrapped lora loaded from', mle_checkpoint)
        
        #include option to turn off grads of unwrapped LoRA parameters
        #this is mostly for temp scaling method
        for name, param in self.model.named_parameters():
            if param.requires_grad: #only update LoRA parameters
                param.requires_grad = self.cfg.lora.requires_grad

    def wrap_lora_layers(self):
        self.wrapper_fn = instantiate(self.cfg.lora.wrapper)
        wrap_lora_layers(self.model, self.wrapper_fn, self.cfg.lora.wrapper.target_modules)
        self.model = self.model.to(self.device) #make sure modified layers are on the right device


    @property
    def param_counts(self):
        try:
            num_trainable_params, total_params = self.model.get_nb_trainable_parameters()
            num_base_params = total_params - num_trainable_params
        except: #not a LoRA model
            num_base_params = sum(p.numel() for p in self.model.parameters())
            num_trainable_params = 0
            total_params = num_base_params
        return {
            'num_trainable_params': num_trainable_params, 
            'num_total_params': total_params, 
            'num_base': num_base_params
        }


    def load_optimizer(self):
        decay_params, no_decay_params = [], []
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                if any(nd in n for nd in self.cfg.optim.no_decay):
                    no_decay_params.append(p)
                else:
                    decay_params.append(p)
        params = [
            {"params": decay_params, "weight_decay": self.cfg.optim.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]
        self.nll_optimizer = instantiate(self.cfg.optim.nll_optimizer, params)
        self.nll_scheduler = instantiate(self.cfg.optim.nll_scheduler, self.nll_optimizer)

    def load_processor(self):
        self.processor = AutoProcessor.from_pretrained(
            self.cfg.hf_model, 
            trust_remote_code=True, 
            padding_side='left'
        )
        
    def load_dataloaders(self):
        train_dataset = instantiate(self.cfg.train_dataset, split=self.cfg.optim.train_split)
        test_dataset = instantiate(self.cfg.test_dataset, split='test')
        self.trainloader = DataLoader(
            train_dataset,
            batch_size=self.cfg.optim.batch_size,
            shuffle=True,
            num_workers=1,
            collate_fn=instantiate(self.cfg.collate_fn, self.processor)
        )
        self.testloader = DataLoader(
            test_dataset,
            batch_size=self.cfg.optim.batch_size,
            shuffle=False,
            num_workers=1,
            collate_fn=instantiate(self.cfg.collate_fn, self.processor)
        )

    def save_model(self):
        sd = self.model.state_dict()
        sd = {k: v for k, v in sd.items() if 'lora_' in k}
        torch.save(sd, os.path.join(self.expdir, "state_dict.pt"))
        print(f"Model saved to {self.expdir}")

    def compute_logits(self, inputs):
        logits = []
        if self.model.training:
            backbone_samples = self.cfg.samples.train.backbone
            last_layer_samples = self.cfg.samples.train.last_layer
        else:
            backbone_samples = self.cfg.samples.test.backbone
            last_layer_samples = self.cfg.samples.test.last_layer
        for i in range(backbone_samples):
            model_output = self.model(**inputs, output_hidden_states=True)
            feats = model_output.hidden_states[-1][:, -1] #last layer, last token, (batch_size, hidden_size)
            for j in range(last_layer_samples):
                logits_ij = self.model.lm_head(feats)#[:, target_ids]  # (batch_size, n_classes)
                logits.append(logits_ij)
        logits = torch.stack(logits, dim=1) # (B, num_samples, num_classes)
        return logits

    def train_step(self, batch):
        self.nll_optimizer.zero_grad()
        inputs, labels = batch
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
    
    def evaluate_step(self, batch):
        inputs, labels = batch
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        torch.cuda.reset_peak_memory_stats(self.device)
        start_event.record()
        with torch.no_grad() and torch.inference_mode():
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

    def train(self):
        # set_seed(self.cfg.seed)
        self.load_optimizer()
        wandb.init(
            project=self.cfg.wandb.project,
            name=self.expdir.replace('logs/', '').replace('/', '_'),
            entity=os.environ.get("WANDB_ENTITY", None),
            config=dict(self.cfg),
        )
        self.model.train()
        dataloader = cycle(self.trainloader)
        for step in trange(self.cfg.optim.max_train_steps, desc="Step", disable=not self.cfg.pbar):
            batch = next(dataloader)
            batch = [b.to(self.device) for b in batch]
            log = self.train_step(batch)
            wandb.log(log)
        del dataloader
        wandb.finish()
        self.save_model()

    def evaluate(self, use_train=False, save=True):
        self.model.eval()
        results = []
        seeds = torch.arange(self.cfg.seed, self.cfg.seed + self.cfg.n_eval_trials)
        for seed in seeds:
            set_seed(seed.item())
            dataloader = self.trainloader if use_train else self.testloader
            test_logits, test_labels, elapsed_times, peak_memories = [], [], [], []
            for batch in tqdm(dataloader, desc="Testing", disable=not self.cfg.pbar):
                batch = [b.to(self.device) for b in batch]
                inputs, labels = batch
                eval_output = self.evaluate_step(batch)
                test_logits.append(eval_output['logits'].cpu())
                test_labels.append(labels.cpu())
                elapsed_times.append(eval_output['elapsed_time'])
                peak_memories.append(eval_output['peak_memory'])
            
            test_logits = torch.cat(test_logits, dim=0) # (num_examples, n_samples, n_classes)
            test_logits = test_logits.to(torch.float64) #use higher precision for stability
            test_logprobs = average_log_probs(test_logits) # (num_examples, n_classes)
            test_probs = torch.exp(test_logprobs)
            test_preds = test_logprobs.argmax(dim=-1)
            test_labels = torch.cat(test_labels, dim=0)
            elapsed_times = torch.tensor(elapsed_times[5:]) / 1000.0 #convert to seconds
            peak_memories = torch.tensor(peak_memories[5:]) / (1024 ** 3) #convert to GB
            result = {'seed': seed.item()}
            # result.update(params_info)
            result['latency'] = elapsed_times.mean().item()
            result['peak_memory'] = peak_memories.mean().item()
            
            metric_kwargs = {'task': 'multiclass', 'num_classes': len(self.class_ids)}
            result['ACC'] = accuracy(test_preds, test_labels, **metric_kwargs).item()
            result['ECE'] = calibration_error(test_probs, test_labels, n_bins=15, **metric_kwargs).item()
            result['NLL'] = F.nll_loss(test_logprobs, test_labels, reduction="mean").item()
            results.append(result)
            print(result)
        
        if save:
            if self.trainset_name != self.testset_name:
                logdir = os.path.join(self.expdir, 'results', 'ood', self.testset_name)
            else:
                logdir = os.path.join(self.expdir, 'results', 'id')
            os.makedirs(logdir, exist_ok=True)
            json_path = os.path.join(logdir, "metrics.json")
            with open(json_path, "w") as f:
                json.dump(results, f)
                f.write("\n")
        return results
