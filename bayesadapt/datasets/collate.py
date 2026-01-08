import torch

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

def instruct_collate_fn(tokenizer, batch):
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
    text_prompts = [text + '{\n"answer": "' for text in text_prompts]
    prompts = tokenizer(
        text_prompts,
        padding=True,
        return_tensors="pt",
        add_special_tokens=False,
    )
    return prompts, labels
