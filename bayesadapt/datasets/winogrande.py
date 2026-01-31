from datasets import load_dataset
from torch.utils.data import Dataset

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
        #so following prior work, we use validation as test
        if split == 'test':
            self.data = load_dataset("winogrande", f"winogrande_{size}", trust_remote_code=True)['validation']

        #winogrande is made up of a nested training set with 5 sizes, with a shared test set across all sizes
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
            'label': int(item['answer']) - 1, #map '1','2' to 0,1
            'question_id': idx
        }
