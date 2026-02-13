from bayesadapt.datasets.obqa import OBQA
from bayesadapt.datasets.circuit_logic import CircuitLogic
from bayesadapt.datasets.collate import instruct_collate_fn
from transformers import AutoProcessor
import torch
from tqdm import tqdm
from functools import partial


for split in ['train', 'validation', 'test']:
    # dataset = OBQA(split=split)
    dataset = CircuitLogic(split=split, representation='expression')

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
    print('Split:', split, 'Max Seq Length:', seq_lengths.max().item(), 'Mean Seq Length:', seq_lengths.mean().item())
import ipdb; ipdb.set_trace() # noqa
question_ids = torch.tensor(question_ids).long()
