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

## 🚀 Running the Code
We have provided the command to reproduce the results of in-distribution and out-of-distribution experiment in the `/scripts` folders. 

### To run the in-distribution experiment, use the following script:
```sh
bash scripts/<method_name>/<method_name>-llama-all.sh
```

### To run the out-of-distribution experiment, use the following script:
```sh
bash scripts/<method_name>/<method_name>-llama-ood-all.sh
```
In this script, we also demonstrate how to save and load your trained LoRA adapter. To save a LoRA checkpoint, use flag: ``--checkpoint --checkpoint-name $name``. To load a LoRA checkpoint, use flag: ``--load-lora-path checkpoints/$modelwrapper/<model_of_checkpoint>/<dataset_of_checkpoint>/<your_previous_checkpoint_name>``.


