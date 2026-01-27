import os
import numpy  # needed (don't change it)
import math
import json
import torch
import torch.nn.functional as F
import wandb
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm, trange
from accelerate.utils import set_seed
from transformers import AutoProcessor
from bayesadapt.utils import load_model, split_batch, infer_logdir_from_cfg, load_dataloader
from bayesadapt.lorawrappers import VILoraWrapper
from torch.utils.data import DataLoader
# from trainer import Trainer
# from vi_trainer import VITrainer
# from laplace_trainer import LaplaceTrainer
# from itertools import cycle

@hydra.main(config_path="./conf", config_name="default", version_base=None)
def main(cfg):
    print(cfg)
    trainer = instantiate(cfg.trainer, cfg=cfg)
    if not os.path.exists(f"{trainer.expdir}/state_dict.pt") or cfg.overwrite:
        trainer.train()
    if not os.path.exists(f"{trainer.evaldir}/metrics.json") or cfg.overwrite:
        trainer.evaluate()

if __name__ == "__main__":
    main()
