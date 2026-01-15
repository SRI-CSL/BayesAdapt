from datasets import load_dataset, ClassLabel
from tqdm import tqdm
import torch
import os
# from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B", trust_remote_code=True, padding_side='left')


# prompt_template = "Answer the following question as Yes or No only.\n{question}"
prompt_template = "Answer the multiple choice question below. Output the letter of your choice only.\n{question}"
class MathVerse(Dataset):
    labels = ['A', 'B', 'C', 'D']
    def __init__(self, split='train'):
        if split not in ['train', 'validation', 'test']:
            raise ValueError(f"Unknown split: {split}")

        #randomly split the dataset into 70% train, 15% val, 15% test
        #also ensure that the splits remain stratified by l2_category
        dataset = load_dataset("AI4Math/MathVerse", 'testmini')['testmini']
        dataset = dataset.filter(lambda x: x['question_type'] == 'multi-choice' and x['answer'] in self.labels)
        problem_indices = dataset.unique('problem_index')
        train_idx, temp_idx = train_test_split(
            problem_indices, test_size=0.30, random_state=42, shuffle=True
        )
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.50, random_state=42, shuffle=True
        )
        # unique_labels = dataset.unique("l2_category")
        # dataset = dataset.cast_column("l2_category", ClassLabel(names=unique_labels))
        # train_remainder = dataset.train_test_split(test_size=0.3, seed=42, stratify_by_column='l2_category')
        # test_val = train_remainder["test"].train_test_split(test_size=0.5, seed=42, stratify_by_column='l2_category')
        if split == 'train':
            self.data = dataset.filter(lambda x: x['problem_index'] in train_idx)
        elif split == 'validation':
            self.data = dataset.filter(lambda x: x['problem_index'] in val_idx)
        else:  # split == 'test'
            self.data = dataset.filter(lambda x: x['problem_index'] in test_idx)


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = prompt_template.format(question=item['question'])

        return {
            'prompt': prompt.strip(),
            'label': self.labels.index(item['answer']),
            'image': item['image']
        }
