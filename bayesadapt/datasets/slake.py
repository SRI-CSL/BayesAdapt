import os
import zipfile
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from datasets import load_dataset
from huggingface_hub import hf_hub_download

def is_boolean(example):
    return example['answer'] in ['Yes', 'No']

def add_gaussian_noise(image, sigma=0.0, per_channel=False, rng=None):
    if sigma <= 0:
        return image
    if rng is None:
        rng = np.random.default_rng()
    arr = np.asarray(image).astype(np.float32)  # (H, W, 3) in 0..255
    if per_channel:
        noise = rng.normal(0.0, sigma, size=arr.shape).astype(np.float32)
    else:
        noise2d = rng.normal(0.0, sigma, size=arr.shape[:2]).astype(np.float32)
        noise = noise2d[..., None]
    noisy_image = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_image)

prompt_template = "Answer the following question as Yes or No only.\n{question}"
class SLAKE(Dataset):
    labels = ['Yes', 'No']
    def __init__(self, split='train', root='./images/slake', noise_std=0.0):
        if split not in ['train', 'validation', 'test']:
            raise ValueError(f"Unknown split: {split}")

        self.noise_std = noise_std
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
        image = add_gaussian_noise(image, sigma=self.noise_std, per_channel=False)
        return {
            'prompt': prompt.strip(),
            'label': self.labels.index(item['answer']),
            'image': image,
            'question_id': item['qid']
        }
