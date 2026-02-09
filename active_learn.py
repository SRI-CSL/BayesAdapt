import math
import json
import torch
import numpy as np
from tqdm import tqdm
from bayesadapt.datasets.mmlu import MMLUPro
from bayesadapt.datasets.arc import ARC
from bayesadapt.datasets.srqa import SRQA
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
    pool_dataset = SRQA(split='train')
    pool_ids = pool_dataset.data['question_id']
    test_dataset = SRQA(split='test')
    id_key = 'question_id'
    initial_train_size = 10
    initial_train_ids = np.random.choice(
        pool_ids, 
        size=initial_train_size, 
        replace=False
    ).tolist()
    train_dataset = SRQA(split='train')
    train_dataset.data = pool_dataset.data.filter(lambda x: x[id_key] in initial_train_ids)
    pool_dataset.data = pool_dataset.data.filter(lambda x: x[id_key] not in initial_train_ids)


    # train_dataset = MMLUPro(split='train')
    # pool_dataset = MMLUPro(split='test', split_fname='./splits/mmlupro_pool_ids.json')
    # pool_ids = pool_dataset.data['question_id']
    # test_dataset = MMLUPro(split='test', split_fname='./splits/mmlupro_test_ids.json')
    # id_key = 'question_id'
    
    # train_dataset = ARC(split='validation', difficulty='challenge')
    # pool_dataset = ARC(split='train', difficulty='challenge')
    # pool_ids = pool_dataset.data['id']
    # test_dataset = ARC(split='test', difficulty='challenge')
    #id_key = 'id'
    # initial_train_size = 10
    # initial_train_ids = np.random.choice(
        # pool_ids, 
        # size=initial_train_size, 
        # replace=False
    # ).tolist()
    # train_dataset = ARC(split='train', difficulty='challenge')
    # train_dataset.data = pool_dataset.data.filter(lambda x: x[id_key] in initial_train_ids)
    # pool_dataset.data = pool_dataset.data.filter(lambda x: x[id_key] not in initial_train_ids)

    num_select = 10
    for i in range(100):
        trainer = instantiate(cfg.trainer, cfg=cfg) #cold start each time
        trainer.update_dataloaders(train_dataset=train_dataset, test_dataset=pool_dataset)
        trainer.train(save=False, use_wandb=False)
        pool_metrics, logits_dict = trainer.evaluate(save=False, verbose=False)
        
        logit_ids = list(logits_dict.keys())
        logits = torch.stack([logits_dict[qid] for qid in logit_ids]) #B x C

        logprobs = average_log_probs(logits) #B x C
        num_classes = logprobs.shape[1]
        probs = torch.exp(logprobs)
        entropies = -torch.sum(probs * logprobs, dim=1) / math.log(num_classes)

        top_indices = torch.topk(
            entropies,
            k=num_select,
            largest=True
        ).indices.tolist()

        selected_qids = [logit_ids[idx] for idx in top_indices]
        selected_hf = pool_dataset.data.filter(lambda x: x[id_key] in selected_qids)
        
        #test on true test set for plots later
        trainer.update_dataloaders(test_dataset=test_dataset)
        test_metrics, _ = trainer.evaluate(save=False)

        # selection_probs = entropies / entropies.sum()
        # selected_qids = torch.multinomial(
            # selection_probs, 
            # num_samples=num_select, 
            # replacement=False
        # ).tolist()
        # selected_hf = pool_dataset.data.filter(lambda x: x[id_key] in selected_qids)
        train_dataset.data = concatenate_datasets([train_dataset.data, selected_hf])
        pool_dataset.data = pool_dataset.data.filter(lambda x: x[id_key] not in selected_qids)
    import ipdb; ipdb.set_trace() # noqa



if __name__ == "__main__":
    main()
