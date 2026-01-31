from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

prompt_template = "Answer the multiple choice question below. Output the letter of your choice only.\n{question}\nChoices:\n"
class ARC(Dataset):
    labels = ['A', 'B', 'C', 'D', 'E']
    difficulties = ['easy', 'challenge']
    def __init__(self, difficulty='easy', split='train'):
        if difficulty not in self.difficulties:
            raise ValueError(f"Unknown difficulty: {difficulty}")

        if split not in ['train', 'validation', 'test']:
            raise ValueError(f"Unknown split: {split}")

        if difficulty == 'easy':
            self.data = load_dataset("ai2_arc", 'ARC-Easy')[split]
        else:
            self.data = load_dataset("ai2_arc", 'ARC-Challenge')[split]


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        text_choices = item['choices']['text']
        try:
            label = self.labels.index(item['answerKey'])
            label_choices = item['choices']['label']
        except:
            #for this example the choices are given as ['1', '2', '3', '4'] instead of letters
            #so we need to convert the choices to letters
            label = int(item['answerKey']) - 1  
            label_choices = [self.labels[i] for i in range(len(text_choices))]

        prompt = prompt_template.format(question=item['question'])
        for letter, choice in zip(label_choices, text_choices):
            prompt += f"{letter}) {choice}\n"

        return {
            'prompt': prompt.strip(),
            'label': label,
            'question_id': item['id']
        }
