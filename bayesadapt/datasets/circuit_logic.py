from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import reasoning_gym

prompt_template = "Consider the following logical expression: {expression}\nWhat is the truth value of this expression given the following variable assignments?\n"
class CircuitLogic(Dataset):
    labels = ['0', '1']
    def __init__(self, split='train'):
        if split not in ['train', 'validation', 'test']:
            raise ValueError(f"Unknown split: {split}")

        if split == 'train':
            data = reasoning_gym.create_dataset('circuit_logic', size=10000, seed=42)
        elif split == 'validation':
            data = reasoning_gym.create_dataset('circuit_logic', size=1000, seed=43)
        else:  # test
            data = reasoning_gym.create_dataset('circuit_logic', size=1000, seed=44)


        self.data = []
        for i, x in enumerate(data):
            self.data.append({
                'expression': x['metadata']['expression'],
                'assignments': x['metadata']['assignments'],
                'question_id': f"{split}_{i}",
                'answer': x['answer']
            })




    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]

        prompt = prompt_template.format(expression=item['expression'])
        for var, val in item['assignments'].items():
            prompt += f"{var} = {val}\n"

        prompt += "Output 0 or 1 only."

        return {
            'prompt': prompt.strip(),
            'label': self.labels.index(item['answer']),
            'question_id': item['question_id']
        }
