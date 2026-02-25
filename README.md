# 🎒Bayesian Adaptation Gym
Bayesian Adaptation Gym (BAG) is a library for the Bayesian adaptation of LLMs and VLMs.

## ⚙️ Installation
BAG uses ``uv`` to manage requirements. Start by installing ``uv`` as described by the [official documentation](https://docs.astral.sh/uv/getting-started/installation).

Clone the code by running: ```git clone https://github.com/SRI-CSL/BayesAdapt.git```

Inside the ```BayesAdapt/``` directory run ```uv init``` to build the environment.

Then run ```source .venv/bin/activate``` to load the environment.

To use wandb, make sure the environment variable ```WANDB_ENTITY``` is set to your full wandb username.

## 🔬🧪 Running an experiment
BAG uses ```hydra``` configuration to define the parameters of an experiment, allowing us to control options from the command-line.
For example, we can train and evaluate a simple MLE adapter using the following Python command:
```bash
python train_and_evaluate.py \
    +lora=default \
    lora.config.r=8 \
    hf_model=Qwen/Qwen3-8B \
    dataset@train_dataset=winogrande_s \
    collate_fn=instruct \
    seed=0 \
    gpu_id=0
```
By default, this will automatically save a trained adapter and evaluation results to:

```logs/Qwen/Qwen3-8B/16bit/mle/rank8/instruct/seed0/winogrande_s```

### 🎁 Wrapping LoRA
From here its straightforward to apply a ```lorawrapper```. For example, for BLoB on the SLAKE dataset:
```bash
python train_and_evaluate.py \
    +lora=default \
    +lora/wrapper=blob \
    lora.config.r=8 \
    optim=vi \
    trainer=vi \
    optim.kl_optimizer.lr=0.01 \
    samples.test.backbone=10 \
    hf_model=Qwen/Qwen3-VL-8B-Instruct \
    dataset@train_dataset=slake \
    collate_fn=vlm \
    seed=0 \
    gpu_id=0
```

## 🛠️ Exteding the code 
### Adding a new LoRA wrapper
To demonstrate how to add a new LoRA wrapper we look at ```bayesadapt/lorawrappers/mcdropout.py``` as an example:
```python
import torch
from .lorawrapper import LoraWrapper

class MCDropoutLoraWrapper(LoraWrapper):
    def __init__(*args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor, *args, **kwargs):
        previous_dtype = x.dtype
        result = self.base_layer(x, *args, **kwargs)
        for active_adapter in self.active_adapters:
            if active_adapter not in self.lora_A.keys():
                continue
            x = x.to(self.lora_B[active_adapter].weight.dtype)
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]
            x = dropout.train()(x)  # always apply dropout even in eval mode
            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            result = result + lora_B(lora_A(x)) * scaling
        result = result.to(previous_dtype)
        return result
```
MCDropout is a simple approach which modifies the forward pass only slightly, but shows how we have complete control over the LoRA forward pass (note that ``lora_A`` and ``lora_B`` are just linear layers). We can also add new parameters and other state in the ``__init__`` function if desired.

Then to use a wrapper with ``hydra`` we just need to add new config file such as ``conf/lora/wrapper/mcdropout.yaml`` with the content:
```yaml
defaults:
  - default

_partial_: true
_target_: bayesadapt.lorawrappers.MCDropoutLoraWrapper
```
Any wrapper specific args can also be included here so they are controllable at the CLI. 


## 📚 Citation
It also acts as the official repo for:<br>
**Scalable Bayesian Low-Rank Adaptation of Large Language Models via Stochastic Variational Subspace Inference**<br>
Colin Samplawski, Adam D. Cobb, Manoj Acharya, Ramneet Kaur, Susmit Jha <br>
*Conference on Uncertainty in Artificial Intelligence, 2025*<br>
[[📄 Paper](https://www.arxiv.org/abs/2506.21408)] [[🌐 OpenReview](https://openreview.net/forum?id=neqGuhC3zS)]

```bib
@InProceedings{samplawski2025scalable,
  title={Scalable Bayesian Low-Rank Adaptation of Large Language Models via Stochastic Variational Subspace Inference},
  author={Samplawski, Colin and Cobb, Adam D and Acharya, Manoj and Kaur, Ramneet and Jha, Susmit},
  booktitle={Conference on Uncertainty in Artificial Intelligence},
  year={2025}
}
```






