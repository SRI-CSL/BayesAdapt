import torch
from PIL import Image

def resize_down_only(image, max_size=224):
    width, height = image.size
    scaling_factor = min(1.0, max_size / width, max_size / height)
    
    # If factor is 1, no resizing is needed
    if scaling_factor == 1.0:
        return image
    
    new_width = int(width * scaling_factor)
    new_height = int(height * scaling_factor)
    return image.resize((new_width, new_height), Image.LANCZOS)

def base_collate_fn(tokenizer, batch):
    labels = torch.tensor([item['label'] for item in batch]).long()
    text_prompts = [item['prompt'] + "\nAnswer:" for item in batch]
    prompts = tokenizer(
        text_prompts,
        padding=True,
        return_tensors="pt",
        add_special_tokens=True,
    )
    return prompts, labels


def vlm_collate_fn(tokenizer, batch):
    labels = torch.tensor([item['label'] for item in batch]).long()
    messages = []
    for item in batch:
        #image = Image.open(item['image']).resize((224, 224))
        image = resize_down_only(item['image'], max_size=224)
        #resize image with aspect ratio preserved
        content = [{'type': 'image', 'image': image},
                   {'type': 'text', 'text': item['prompt']}]
        messages.append([{"role": "user", "content": content}])
    
    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        padding=True,
        return_tensors="pt"
    )
    return inputs, labels

def instruct_collate_fn(tokenizer, batch):
    question_ids = [iten['question_id'] for iten in batch]
    labels = torch.tensor([item['label'] for item in batch]).long()
    messages = [
        [{'role': 'user', 'content': item['prompt']}] 
        for item in batch
    ]
    text_prompts = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False #Qwen3 specific, doesnt seem to break other models
    )
    # text_prompts = [text + '{\n"answer": "' for text in text_prompts]
    prompts = tokenizer(
        text_prompts,
        padding=True,
        return_tensors="pt",
        add_special_tokens=False,
    )
    return prompts, labels, question_ids
