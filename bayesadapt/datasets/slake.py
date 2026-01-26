import os
import zipfile
from PIL import Image
from torch.utils.data import Dataset
from datasets import load_dataset
from huggingface_hub import hf_hub_download

def is_boolean(example):
    return example['answer'] in ['Yes', 'No']

prompt_template = "Answer the following question as Yes or No only.\n{question}"
class SLAKE(Dataset):
    labels = ['Yes', 'No']
    def __init__(self, split='train', root='./images/slake'):
        if split not in ['train', 'validation', 'test']:
            raise ValueError(f"Unknown split: {split}")
        
        self.image_dir = os.path.join(root, 'imgs')

        if not os.path.exists(self.image_dir):
            zip_path = hf_hub_download(repo_id="BoKelvin/SLAKE", filename="imgs.zip", repo_type="dataset")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(root)
        
        self.data = load_dataset("BoKelvin/SLAKE", split=split)
        self.data = self.data.filter(is_boolean)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = os.path.join(self.image_dir, item['img_name'])
        image = Image.open(img_path)
        prompt = prompt_template.format(question=item['question'])

        return {
            'prompt': prompt.strip(),
            'label': self.labels.index(item['answer']),
            'image': image
        }
