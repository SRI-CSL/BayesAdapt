# BayesAdapt
BayesAdapt is a library for the Bayesian adaptation of LLMs.

It also acts as the official repo for:<br>
**Scalable Bayesian Low-Rank Adaptation of Large Language Models via Stochastic Variational Subspace Inference**<br>
Colin Samplawski, Adam D. Cobb, Manoj Acharya, Ramneet Kaur, Susmit Jha <br>
*Conference on Uncertainty in Artificial Intelligence, 2025*<br>
[[📄 Paper](https://www.arxiv.org/abs/2506.21408)] [[🌐 OpenReview](https://openreview.net/forum?id=neqGuhC3zS)]

## ⚙️ Installation
BayesAdapt uses uv to manage requirements. Start by installing uv as described by the [official documentation](https://docs.astral.sh/uv/getting-started/installation).

Clone the code by running: ```git clone https://github.com/SRI-CSL/BayesAdapt.git```

Inside the ```BayesAdapt/``` directory run ```uv init``` to build the environment.

Then run ```source .venv/bin/activate``` to load the environment.

To use wandb, make sure the environment variable ```WANDB_ENTITY``` is set to your full wandb username.

## 🔬🧪 Running an experiment
BayesAdapt uses ```hydra``` configuration to define the parameters of an experiment, allowing us to control options from the command-line.
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


## 📚 Citation
```bib
@InProceedings{samplawski2025scalable,
  title={Scalable Bayesian Low-Rank Adaptation of Large Language Models via Stochastic Variational Subspace Inference},
  author={Samplawski, Colin and Cobb, Adam D and Acharya, Manoj and Kaur, Ramneet and Jha, Susmit},
  booktitle={Conference on Uncertainty in Artificial Intelligence},
  year={2025}
}
```



