from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
import reasoning_gym
from datasets import Dataset as HFDataset

prompt_template = "Consider the following logical expression: {expression}\nWhat is the truth value of this expression given the following variable assignments?\n"
class CircuitLogic(Dataset):
    labels = ['0', '1']
    def __init__(self, split='train', representation='expression'):
        if split not in ['train', 'validation', 'test']:
            raise ValueError(f"Unknown split: {split}")

        if split == 'train':
            data = reasoning_gym.create_dataset('circuit_logic', size=10000, seed=42)
        elif split == 'validation':
            data = reasoning_gym.create_dataset('circuit_logic', size=1000, seed=43)
        else:  # test
            data = reasoning_gym.create_dataset('circuit_logic', size=1000, seed=44)

        # expression_legend = {
            # '&': 'AND',
            # '↑': 'NAND',
            # '⊕': 'XOR',
            # "'": 'Negate',
            # "+": 'OR'
        # }
        # expression_legend = "\n".join([f"{k}: {v}" for k, v in expression_legend.items()])
        # expression_legend = f"Legend:\n{expression_legend}".strip()

        self.data = []
        for i, x in enumerate(data):
            question = x['question']
            question_tokens = question.split('\n\n')
            header, gate, legend, assignments, final_question = question_tokens
            legend = legend.replace(' for gates', '').strip()
            if representation == 'expression':
                logic = x['metadata']['expression']

                #use same symbols as in gates, for slightly more fair ood eval
                logic = logic.replace('&', '&&')
                logic = logic.replace('↑', '↑↑')
                logic = logic.replace('⊕', '⊕⊕')
                logic = logic.replace("'", ">o")
                logic = logic.replace('+', '++')
                header = header.replace('circuit', 'expression')
            elif representation == 'circuit':
                logic = gate.strip()
            else:
                raise ValueError(f"Unknown representation: {representation}")
            prompt = f"{header}\n{logic}\n{legend}\n{assignments}\n{final_question}"
            self.data.append({
                'prompt': prompt,
                'question_id': f"{split}_{i}",
                'answer': x['answer']
            })

        self.data = HFDataset.from_list(self.data)



    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]

        # prompt = prompt_template.format(expression=item['expression'])
        # for var, val in item['assignments'].items():
            # prompt += f"{var} = {val}\n"

        # prompt += "Output 0 or 1 only."

        return {
            'prompt': item['prompt'].strip(),
            'label': self.labels.index(item['answer']),
            'question_id': item['question_id']
        }
