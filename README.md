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
bash scripts/<method_name>/<method_name>-<model_name>-id.sh
```

### To run the out-of-distribution experiment, use the following script:
```sh
bash scripts/<method_name>/<method_name>-<model_name>-ood.sh
```
In this script, we also demonstrate how to save and load your trained LoRA adapter. To save a LoRA checkpoint, use flag: ``--checkpoint --checkpoint-name $name``. To load a LoRA checkpoint, use flag: ``--load-lora-path checkpoints/$modelwrapper/<model_of_checkpoint>/<dataset_of_checkpoint>/<your_previous_checkpoint_name>``.

## 🔧 Extending the Code
This code is a fork of the [bayesian-peft](https://github.com/Wang-ML-Lab/bayesian-peft/) library which was introduced for the BLoB method. Through this fork we inherit the following baselines:
- MLE / MAP (i.e. base LoRA)
- Deep Ensembles
- Monte Carlo Dropout
- Bayesian LoRA by Backprop (BLoB)


### Overview of the WrapperBase Class
The `WrapperBase` class in `bayesian-peft/modelwrappers/wrapperbase.py` is designed as a flexible base class that integrates with various PEFT frameworks and datasets. Key features include:

* **Evaluation Metrics:** Includes accuracy, calibration error, negative log-likelihood, and Brier score.
* **Adapter Support:** Seamlessly integrates with the PEFT framework for parameter-efficient fine-tuning.
* **Optimizer and Scheduler:** Configurable optimizer and learning rate scheduler.
* **Training Loop:** Handles training and evaluation with built-in logging and metrics.

### Creating a Custom Wrapper
To implement a custom wrapper:

1. **Inherit from `WrapperBase`:** Your custom wrapper should subclass `WrapperBase`.
2. **Override `forward_logits`:** Implement how your model generates logits from input batches.
3. **Add Custom Behavior:** Extend or modify the existing methods to suit your needs.

Below is an example of creating a custom wrapper, CustomWrapper.

#### Step 1: Subclass `WrapperBase`
To create your custom wrapper, first subclass the `WrapperBase` class. This class manages training and evaluation routines, so when you create a custom wrapper, you can extend or modify any of the existing methods.

```python
from wrapperbase import WrapperBase

class CustomWrapper(WrapperBase):
    def __init__(self, model, peft_config, args, accelerator, adapter_name="default"):
        super().__init__(model, peft_config, args, accelerator, adapter_name)
        # Your custom initialization code
```

#### Step 2: Implement the `forward_logits` Method
The `forward_logits` method is used to define the forward pass for your model. It must return the logits (output) of the model, which are used to calculate the loss during training and for evaluation. Note that the `forward_logits` method is not implemented in the `WrapperBase` class; you need to implement it based on your specific requirements.
```python
def forward_logits(self, batch, sample=True, n_samples=1, **kwargs):
    # Custom logic to process the batch and return logits
    output = self.base_model(**batch)
    logits = output.logits
    return logits
```

#### Step 3: Add Custom Training and Evaluation Logic (Optional)
You can customize the training logic by overriding the `fit` method, which manages the training loop. You can modify how gradients are computed, how the model is updated, and how metrics are logged. The `evaluate` method handles the evaluation of your model. You can customize it to calculate additional metrics, apply different evaluation procedures, or modify how results are logged. You can also customize the `fit_evaluate` and `prepare_for_fit_evaluate` method to further control the procedure of training and evaluating.

For more information about the `WrapperBase` class, refer to the code provided in the project.


## 📚 Citation
```bib
@InProceedings{samplawski2025scalable,
  title={Scalable Bayesian Low-Rank Adaptation of Large Language Models via Stochastic Variational Subspace Inference},
  author={Samplawski, Colin and Cobb, Adam D and Acharya, Manoj and Kaur, Ramneet and Jha, Susmit},
  booktitle={Conference on Uncertainty in Artificial Intelligence},
  year={2025}
}
```
