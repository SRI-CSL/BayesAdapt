from torch.utils.data import Dataset
from datasets import load_dataset

prompt_template = "Answer the multiple choice question below. Output the letter of your choice only.\n{question}\nChoices:\n"
class OBQA(Dataset):
    labels = ['A', 'B', 'C', 'D']
    def __init__(self, split='train'):
        if split not in ['train', 'validation', 'test']:
            raise ValueError(f"Unknown split: {split}")
        self.data = load_dataset("openbookqa", "main")[split]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        label = self.labels.index(item['answerKey'])
        
        text_choices = item['choices']['text']
        label_choices = item['choices']['label']

        prompt = prompt_template.format(question=item['question_stem'])
        for letter, choice in zip(label_choices, text_choices):
            prompt += f"{letter}) {choice}\n"

        return {
            'prompt': prompt.strip(),
            'label': label,
            'question_id': item['id']
        }
