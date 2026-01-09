from datasets import load_dataset
from tqdm import tqdm
import torch
from torch.utils.data import Dataset

prompt_template = """Read the passage below and answer the question with the words 'true' or 'false'.
Passage: {passage}
Question: {question}?"""

class BoolQ(Dataset):
    labels = ['false', 'true']
    def __init__(self, split='train'):
        #boolq has no test set
        #so following prior work, we use validation as test
        if split == 'test':
            self.data = load_dataset("boolq")['validation']

        elif split == 'validation':
            import ipdb; ipdb.set_trace() # noqa

        elif split == 'train':
            self.data = load_dataset("boolq")['train']

        else:
            raise ValueError(f"Unknown split: {split}")


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = prompt_template.format(
            passage=item['passage'],
            question=item['question'],
        )
        return {
            'prompt': prompt,
            'label': int(item['answer'])
        }
