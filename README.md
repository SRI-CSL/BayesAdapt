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
TODO


## 📚 Citation
```bib
@InProceedings{samplawski2025scalable,
  title={Scalable Bayesian Low-Rank Adaptation of Large Language Models via Stochastic Variational Subspace Inference},
  author={Samplawski, Colin and Cobb, Adam D and Acharya, Manoj and Kaur, Ramneet and Jha, Susmit},
  booktitle={Conference on Uncertainty in Artificial Intelligence},
  year={2025}
}
```

