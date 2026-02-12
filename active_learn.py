import sys
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
from accelerate.utils import set_seed
import torch.nn.functional as F

def bald(logits):
    logprobs = F.log_softmax(logits, dim=-1) #B x N x C
    probs = F.softmax(logits, dim=-1) #B x N x C
    avg_probs = probs.mean(dim=1) #B x C
    log_avg_probs = average_log_probs(logits) #B x C
    entropy_avg = -torch.sum(avg_probs * log_avg_probs, dim=-1) #B
    sample_entropy = -torch.sum(probs * logprobs, dim=-1).mean(dim=1) #B
    bald_score = entropy_avg - sample_entropy
    return bald_score

@hydra.main(config_path="./conf", config_name="default", version_base=None)
def main(cfg):
    print(cfg)
    set_seed(cfg.seed)
    pool_dataset = instantiate(cfg.train_dataset, split='train')
    pool_ids = pool_dataset.data['question_id']
    test_dataset = instantiate(cfg.test_dataset, split='test')
    id_key = 'question_id'
    num_select = 10
    initial_train_ids = np.random.choice(
        pool_ids, 
        size=num_select,
        replace=False
    ).tolist()
    train_dataset = instantiate(cfg.train_dataset, split='train')
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
    history = []
    for i in range(20):
        trainer = instantiate(cfg.trainer, cfg=cfg) #cold start each time
        evaldir = trainer.evaldir.replace('id', 'active_learn')

        json_path = os.path.join(evaldir, 'results.json')
        if os.path.exists(json_path):
            if not cfg.overwrite:
                sys.exit(f"Results already exist at {json_path}, experiment already complete")

    
        
        pool_ids = pool_dataset.data[id_key]
        #randomly sample 1000 from pool for efficiency, can be changed to all if desired
        if len(pool_ids) > 1000:
            sampled_pool_ids = np.random.choice(pool_ids, size=1000, replace=False).tolist()
            sampled_pool_dataset = instantiate(cfg.train_dataset, split='train')
            sampled_pool_dataset.data = pool_dataset.data.filter(lambda x: x[id_key] in sampled_pool_ids)
        else:
            sampled_pool_dataset = pool_dataset

        trainer.update_dataloaders(train_dataset=train_dataset, test_dataset=sampled_pool_dataset)
        trainer.train(save=False, use_wandb=False)
        pool_metrics, logits_dict = trainer.evaluate(save=False, verbose=False)
        
        logit_ids = list(logits_dict.keys())
        logits = torch.stack([logits_dict[qid] for qid in logit_ids]) #B x N x C
        logits = logits.to(torch.float64)
        B, num_samples, num_classes = logits.shape

        if num_samples == 1: #only one sample, probably MLE, so use predictive entropy instead of BALD
            logprobs = average_log_probs(logits) #B x C
            probs = torch.exp(logprobs)
            scores = -torch.sum(probs * logprobs, dim=1) / math.log(num_classes) #entropy normalized by log(num_classes)
        else:
            scores = bald(logits) #B

        top_values, top_indices = torch.topk(
            scores,
            k=num_select,
            largest=True
        )

        selected_qids = [logit_ids[idx] for idx in top_indices.tolist()]
        selected_hf = pool_dataset.data.filter(lambda x: x[id_key] in selected_qids)
        
        #test on true test set for plots later
        trainer.update_dataloaders(test_dataset=test_dataset)
        test_metrics, _ = trainer.evaluate(save=False)

        history.append({
            'selected_qids': selected_qids,
            'pool_metrics': pool_metrics,
            'test_metrics': test_metrics,
            'top_scores': top_values.cpu().tolist(),
        })

        # selection_probs = entropies / entropies.sum()
        # selected_qids = torch.multinomial(
            # selection_probs, 
            # num_samples=num_select, 
            # replacement=False
        # ).tolist()
        # selected_hf = pool_dataset.data.filter(lambda x: x[id_key] in selected_qids)
        train_dataset.data = concatenate_datasets([train_dataset.data, selected_hf])
        pool_dataset.data = pool_dataset.data.filter(lambda x: x[id_key] not in selected_qids)
    
    if not os.path.exists(evaldir):
        os.makedirs(evaldir, exist_ok=True)

    json_path = os.path.join(evaldir, 'results.json')
    with open(json_path, 'w') as f:
        json.dump(history, f)



if __name__ == "__main__":
    main()
