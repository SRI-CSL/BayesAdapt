from datasets import load_dataset
from tqdm import tqdm
import torch
import os
# from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader

# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B", trust_remote_code=True, padding_side='left')

def is_boolean(example):
    return example['answer'] in ['Yes', 'No']


# prompt_template = "Answer the following question as Yes or No only.\n{question}"
# class SLAKE(Dataset):
    # labels = ['A', 'B', 'C']
    # def __init__(self, split='train', root='slake_images/imgs/'):
        # if split not in ['train', 'validation', 'test']:
            # raise ValueError(f"Unknown split: {split}")

        # self.root = root

        # self.data = load_dataset("BoKelvin/SLAKE", split=split)
        # self.data = self.data.filter(is_boolean)

    # def __len__(self):
        # return len(self.data)
    
    # def __getitem__(self, idx):
        # item = self.data[idx]
        # img_path = os.path.join(self.root, item['img_name'])
        # prompt = prompt_template.format(question=item['question'])

        # return {
            # 'prompt': prompt.strip(),
            # 'label': self.labels.index(item['answer']),
            # 'image': img_path
        # }

prompt_template = "Answer the following question as Yes or No only.\n{question}"
class SLAKE(Dataset):
    labels = ['Yes', 'No']
    def __init__(self, split='train', root='slake_images/imgs/'):
        if split not in ['train', 'validation', 'test']:
            raise ValueError(f"Unknown split: {split}")

        self.root = root

        self.data = load_dataset("BoKelvin/SLAKE", split=split)
        self.data = self.data.filter(is_boolean)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = os.path.join(self.root, item['img_name'])
        prompt = prompt_template.format(question=item['question'])

        return {
            'prompt': prompt.strip(),
            'label': self.labels.index(item['answer']),
            'image': img_path
        }
