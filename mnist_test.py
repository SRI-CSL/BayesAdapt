import os
import torch
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from transformers import AutoTokenizer
from bayesadapt.utils import load_model, split_batch
from transformers import AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig, AutoProcessor

class MNISTVLMDataset(Dataset):
    def __init__(self, root_dir='./data', split='train', prompt_text="Identify the number in this image. Output only the digit."):
        """
        Args:
            root_dir (str): Directory where MNIST data and extracted images will be stored. split (str): 'train' or 'test'.
            prompt_text (str): The text prompt to accompany the image.
        """
        self.root_dir = root_dir
        self.images_dir = os.path.join(root_dir, 'mnist_images', split)
        self.split = split
        self.prompt_text = prompt_text
        
        # Download and load the standard MNIST dataset
        train = (split == 'train')
        self.mnist_data = datasets.MNIST(root=root_dir, train=train, download=True)
        
        # Ensure image directory exists
        os.makedirs(self.images_dir, exist_ok=True)
        
        # Pre-save images to disk if they don't exist
        print(f"Checking/Saving {split} images to disk at {self.images_dir}...")
        self._save_images_to_disk()

    def _save_images_to_disk(self):
        """
        Iterates through the MNIST dataset and saves images as PNGs 
        so they can be referenced by file path.
        """
        # We check for the existence of the last file to avoid re-processing 
        # (Naive check, but sufficient for this demo)
        last_idx = len(self.mnist_data) - 1
        if os.path.exists(self.get_image_path(last_idx)):
            return

        for idx in tqdm(range(len(self.mnist_data)), desc=f"Saving {self.split} images"):
            img, _ = self.mnist_data[idx]
            # img is already a PIL Image because we didn't pass a transform to datasets.MNIST
            file_path = self.get_image_path(idx)
            img.save(file_path)

    def get_image_path(self, idx):
        return os.path.join(self.images_dir, f"mnist_{idx}.png")

    def __len__(self):
        return len(self.mnist_data)

    def __getitem__(self, idx):
        # Get the label for verification/ground truth (optional, but useful for the classifier)
        _, label = self.mnist_data[idx]
        image_path = self.get_image_path(idx)
        
        # Construct the VLM-specific message format
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a generic visual assistant."}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": os.path.abspath(image_path)},
                    {"type": "text", "text": self.prompt_text}
                ]
            }
        ]
        
        # We return the messages dict and the integer label (for calculating loss later)
        return {
            "messages": messages,
            "label": label, 
            "image_path": image_path 
        }

def vlm_collate_fn(batch):
    """
    Custom collate function.
    Because 'messages' are complex nested dictionaries/lists, default_collate 
    might fail or try to stack them in weird ways. 
    We just return a list of message objects and a tensor of labels.
    """
    messages_batch = [item['messages'] for item in batch]
    labels_batch = torch.tensor([item['label'] for item in batch])
    paths_batch = [item['image_path'] for item in batch]
    
    return {
        "messages": messages_batch,
        "labels": labels_batch,
        "image_paths": paths_batch
    }

# --- Main Execution Block ---
if __name__ == "__main__":

    cifar10_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model_id = 'google/gemma-3-4b-it'
    # tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_id)

    class_ids = processor.tokenizer.convert_tokens_to_ids([str(i) for i in range(10)])
    class_ids = torch.tensor(class_ids).long()
    import ipdb; ipdb.set_trace() # noqa

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=None,
        device_map=device,
        torch_dtype=torch.bfloat16,
        # tie_word_embeddings=False,
    )


    # 1. Setup the Dataset
    # We use a specific prompt designed for classification
    dataset = MNISTVLMDataset(
        root_dir='./mnist_vlm_data', 
        split='test',  # Using test for smaller demo size
        prompt_text="What number is written in this image? Answer with a single digit."
    )

    # 2. Setup the DataLoader
    # batch_size=4 to mimic a small inference batch
    dataloader = DataLoader(
        dataset, 
        batch_size=8, 
        shuffle=True, 
        collate_fn=vlm_collate_fn
    )

    # 3. Simulate an Iteration
    print("\n--- Simulating VLM Input Batch ---")
    batch = next(iter(dataloader))
   
    all_preds, all_labels = [], []
    for batch in tqdm(dataloader, desc="Evaluating on MNIST VLM Dataset"):
        inputs = processor.apply_chat_template(
            batch['messages'], 
            add_generation_prompt=True, 
            tokenize=True,
            return_dict=True, 
            return_tensors="pt"
        ).to(model.device, dtype=torch.bfloat16)
    
        with torch.no_grad():
            output = model(**inputs)

        cls_logits = output.logits[:, -1, class_ids]  # (batch_size, num_classes)
        predictions = torch.argmax(cls_logits, dim=-1)  # (batch_size,)
        all_preds.extend(predictions.cpu().tolist())
        all_labels.extend(batch['labels'].cpu().tolist())

        batch_acc = (predictions.cpu() == batch['labels']).float().mean().item()
        print(f"Batch Accuracy: {batch_acc:.4f}")

    all_preds = torch.tensor(all_preds)
    all_labels = torch.tensor(all_labels)
    accuracy = (all_preds == all_labels).float().mean().item()
        
    import ipdb; ipdb.set_trace() # noqa
    
    # 4. Display structure
    print(f"Batch Size: {len(batch['messages'])}")
    print(f"Ground Truth Labels: {batch['labels']}")
    
    print("\nSample Input Structure (First Item in Batch):")
    import json
    print(json.dumps(batch['messages'][0], indent=2))
    
    print("\n--- How to use this for Classification ---")
    print("1. Pass batch['messages'] into your VLM processor/tokenizer.")
    print("2. Run the model forward pass.")
    print("3. Extract logits for the last token position.")
    print("4. Softmax over the indices corresponding to tokens ['0', '1', ... '9'].")
