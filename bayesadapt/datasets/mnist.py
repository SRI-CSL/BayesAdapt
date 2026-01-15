from datasets import load_dataset, ClassLabel
from tqdm import tqdm
import torch
import os
# from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from torchvision.datasets import MNIST as TorchMNIST

# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B", trust_remote_code=True, padding_side='left')


# prompt_template = "Answer the following question as Yes or No only.\n{question}"
prompt_template = "Which numerical digit is shown in the image? Output the digit from 0 to 9 only."
class MNIST(Dataset):
    labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    def __init__(self, root='images/mnist', split='train'):
        self.root = root
        if split not in ['train', 'validation', 'test']:
            raise ValueError(f"Unknown split: {split}")
        
        train_dataset = TorchMNIST(root=self.root, train=True, download=True)
        num_train_images = len(train_dataset)
        indices = list(range(num_train_images))
        train_indices, val_indices = train_test_split(indices, test_size=0.1, random_state=42)
        if split == 'train':
            self.data = torch.utils.data.Subset(train_dataset, train_indices)
        elif split == 'validation':
            self.data = torch.utils.data.Subset(train_dataset, val_indices)
        elif split == 'test':
            self.data = TorchMNIST(root=self.root, train=False, download=True)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image, label = self.data[idx]
        return {
            'prompt': prompt_template.strip(),
            'label': label,
            'image': image
        }
