from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

prompt_template = "Answer the multiple choice question below. Output the letter of your choice only.\n{question}"
class MathVerse(Dataset):
    labels = ['A', 'B', 'C', 'D']
    def __init__(self, split='train'):
        if split not in ['train', 'validation', 'test']:
            raise ValueError(f"Unknown split: {split}")

        #randomly split the dataset into 70% train, 15% val, 15% test
        #also ensure that the splits remain stratified by problem_index, so that all questions from the same problem are in the same split
        dataset = load_dataset("AI4Math/MathVerse", 'testmini')['testmini']
        dataset = dataset.filter(lambda x: x['question_type'] == 'multi-choice' and x['answer'] in self.labels)
        problem_indices = dataset.unique('problem_index')
        train_idx, temp_idx = train_test_split(
            problem_indices, test_size=0.30, random_state=42, shuffle=True
        )
        val_idx, test_idx = train_test_split(
            temp_idx, test_size=0.50, random_state=42, shuffle=True
        )
        if split == 'train':
            self.data = dataset.filter(lambda x: x['problem_index'] in train_idx)
        elif split == 'validation':
            self.data = dataset.filter(lambda x: x['problem_index'] in val_idx)
        else:  # split == 'test'
            self.data = dataset.filter(lambda x: x['problem_index'] in test_idx)


    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        prompt = prompt_template.format(question=item['question'])

        return {
            'prompt': prompt.strip(),
            'label': self.labels.index(item['answer']),
            'image': item['image'],
            'question_id': item['sample_index']
        }
