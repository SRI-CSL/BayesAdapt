from datasets import load_dataset
from tqdm import tqdm
import torch
# from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader

# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B", trust_remote_code=True, padding_side='left')

prompt_template = """For the sentence given below, select the answer that best fills in the blank (_) from the given choices.
{sentence}
Choices:
A) {option1}
B) {option2}"""

class Winogrande(Dataset):
    labels = ['A', 'B']
    sizes = ['xs', 's', 'm', 'l', 'xl']
    def __init__(self, size='s', split='train'):
        if size == 'xl':
            raise ValueError("instances unique to winogrande_xl are being used for validation")

        #winogrande has no test set labels
        #so we use validation as test
        if split == 'test':
            self.data = load_dataset("winogrande", f"winogrande_{size}", trust_remote_code=True)['validation']

        #winogrande is made up of a nested training set with 5 sizes
        #we take the elements ONLY in winogrande_xl to be our validation set
        #this ensures no overlap with training data, but prolcudes the use of the full xl set for training
        elif split == 'validation':
            l_train = load_dataset("winogrande", f"winogrande_l", trust_remote_code=True)['train']
            xl_train = load_dataset("winogrande", f"winogrande_xl", trust_remote_code=True)['train']
            l_sentences = set(l_train['sentence'])
            self.data = xl_train.filter(lambda x: x['sentence'] not in l_sentences)

        elif split == 'train':
            self.data = load_dataset("winogrande", f"winogrande_{size}", trust_remote_code=True)['train']

        else:
            raise ValueError(f"Unknown split: {split}")


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = prompt_template.format(
            sentence=item['sentence'],
            option1=item['option1'],
            option2=item['option2']
        )
        return {
            'prompt': prompt,
            'label': int(item['answer']) - 1 #map '1','2' to 0,1
        }


# def base_collate_fn(tokenizer, batch):
    # labels = torch.tensor([item['label'] for item in batch]).long()
    # text_prompts = [item['prompt'] + "\nAnswer:" for item in batch]
    # prompts = tokenizer(
        # text_prompts,
        # padding=True,
        # return_tensors="pt",
        # add_special_tokens=True,
    # )
    # return prompts, labels

# def instruct_collate_fn(tokenizer, batch):
    # labels = torch.tensor([item['label'] for item in batch]).long()
    # messages = [
        # [{'role': 'user', 'content': item['prompt']}] 
        # for item in batch
    # ]
    # text_prompts = tokenizer.apply_chat_template(
        # messages,
        # tokenize=False,
        # add_generation_prompt=True,
        # enable_thinking=False #Qwen3 specific, doesnt seem to break other models
    # )
    # prompts = tokenizer(
        # text_prompts,
        # padding=True,
        # return_tensors="pt",
        # add_special_tokens=False,
    # )
    # return prompts, labels


# ds = Winogrande(size='s', split='validation')
# dataloader = torch.utils.data.DataLoader(
    # ds,
    # batch_size=4,
    # shuffle=False,
    # collate_fn=lambda batch: instruct_collate_fn(tokenizer, batch),
    # num_workers=2,
# )
# for batch in dataloader:
    # import ipdb; ipdb.set_trace() # noqa

