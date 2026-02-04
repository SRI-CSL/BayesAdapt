import math
import json
import torch
import numpy as np
from tqdm import tqdm
from bayesadapt.datasets.mmlu import MMLUPro
from bayesadapt.datasets.collate import instruct_collate_fn
from transformers import AutoProcessor
from functools import partial
from datasets import load_dataset, ClassLabel
import os
import hydra
from hydra.utils import instantiate
from datasets import concatenate_datasets
from bayesadapt.utils import average_log_probs


@hydra.main(config_path="./conf", config_name="default", version_base=None)
def main(cfg):
    print(cfg)
    train_dataset = MMLUPro(split='train')
    pool_dataset = MMLUPro(split='test', split_fname='./splits/mmlupro_pool_ids.json')
    test_dataset = MMLUPro(split='test', split_fname='./splits/mmlupro_test_ids.json')

    num_select = 100
    for i in range(100):
        trainer = instantiate(cfg.trainer, cfg=cfg) #cold start each time
        trainer.update_dataloaders(train_dataset=train_dataset, test_dataset=pool_dataset)
        trainer.train(save=False, use_wandb=False)
        pool_metrics, logits = trainer.evaluate(save=False, verbose=False)
        logprobs = average_log_probs(logits) #B x C
        num_classes = logprobs.shape[1]
        probs = torch.exp(logprobs)
        entropies = -torch.sum(probs * logprobs, dim=1) / math.log(num_classes)

        selected_qids = torch.topk(
            entropies,
            k=num_select,
            largest=True
        ).indices.tolist()
        
        #test on true test set for plots later
        trainer.update_dataloaders(test_dataset=test_dataset)
        test_metrics, _ = trainer.evaluate(save=False)

        # selection_probs = entropies / entropies.sum()
        # selected_qids = torch.multinomial(
            # selection_probs, 
            # num_samples=num_select, 
            # replacement=False
        # ).tolist()
        selected_hf = pool_dataset.data.filter(lambda x: x["question_id"] in selected_qids)
        train_dataset.data = concatenate_datasets([train_dataset.data, selected_hf])
        pool_dataset.data = pool_dataset.data.filter(lambda x: x["question_id"] not in selected_qids)
    import ipdb; ipdb.set_trace() # noqa



if __name__ == "__main__":
    main()
