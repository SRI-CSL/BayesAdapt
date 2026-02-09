from datasets import load_dataset, ClassLabel, load_from_disk
from tqdm import tqdm
import torch
import os
# from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader

# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B", trust_remote_code=True, padding_side='left')


# prompt_template = "Answer the following question as Yes or No only.\n{question}"
prompt_template = "For the provided image of a plot, which of following formulas best describes the relationship between the variables? Output the letter of your choice only.\nChoices:\n"
class SRQA(Dataset):
    labels = ['A', 'B', 'C', 'D']
    def __init__(self, split='train'):
        if split not in ['train', 'validation', 'test']:
            raise ValueError(f"Unknown split: {split}")

        self.data = load_from_disk("srqa")[split]


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = prompt_template
        for letter, choice in zip(self.labels, item['choices']):
            prompt += f"{letter}) {choice}\n"

        return {
            'prompt': prompt.strip(),
            'label': self.labels.index(item['label']),
            'image': item['image'],
            'question_id': item['question_id']
        }
