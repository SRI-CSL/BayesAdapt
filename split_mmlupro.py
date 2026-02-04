import json
import torch
import numpy as np
from tqdm import tqdm
from bayesadapt.datasets.mmlu import MMLUPro
from bayesadapt.datasets.collate import instruct_collate_fn
from transformers import AutoProcessor
from functools import partial
from datasets import load_dataset, ClassLabel

dataset = MMLUPro(split='test')

processor = AutoProcessor.from_pretrained(
    'Qwen/Qwen3-8B',
    trust_remote_code=True, 
    padding_side='left'
)
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    collate_fn=partial(instruct_collate_fn, processor)
)

seq_lengths, question_ids = [], []
for batch in tqdm(dataloader):
    input_ids = batch[0]['input_ids']
    seq_lengths.extend([len(ids) for ids in input_ids])
    question_ids.extend(batch[2])

seq_lengths = torch.tensor(seq_lengths).float()
question_ids = torch.tensor(question_ids).long()

too_long = seq_lengths > 512
too_long_ids = question_ids[too_long].tolist()

hf_dataset = dataset.data
too_long_dataset = hf_dataset.filter(lambda x: x['question_id'] in too_long_ids)
short_dataset = hf_dataset.filter(lambda x: x['question_id'] not in too_long_ids)

unique_categories = short_dataset.unique('category')
short_dataset = short_dataset.cast_column('category', ClassLabel(names=unique_categories))
split = short_dataset.train_test_split(test_size=0.5, seed=42, stratify_by_column='category')

pool_ids = sorted(split['train']['question_id'])
test_ids = sorted(split['test']['question_id'])

with open('./splits/mmlupro_pool_ids.json', 'w') as f:
    json.dump(pool_ids, f)

with open('./splits/mmlupro_test_ids.json', 'w') as f:
    json.dump(test_ids, f)
