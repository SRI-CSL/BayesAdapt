from torch.utils.data import Dataset
from datasets import load_dataset, concatenate_datasets, get_dataset_config_names

#prompt_template = "Answer the multiple choice question below. Output the letter of your choice only.\n{question}\nChoices:\n"
#Please show your choice in the answer field with only the choice letter, e.g., "answer": "C"."
#prompt_template = """Answer the multiple choice question below. Output your choice in json format with an "answer" field and only the chosen letter, e.g. {{"answer": "C"}}.\n{question}\nChoices:\n"""



#prompt_template = """Answer the multiple choice question below in JSON format. Please show your choice in the answer field with only the choice letter, e.g., "answer": "C"."\n{question}\nChoices:\n"""
prompt_template = "Answer the multiple choice question below. Output the letter of your choice only.\n{question}\nChoices:\n"

class MMLU(Dataset):
    labels = ['A', 'B', 'C', 'D']
    def __init__(self, split='test', topic=None, remove_errors=True):
        if split not in ['test']:
            raise ValueError(f"Unknown split: {split}")        
        
        if topic is None: 
            topics = get_dataset_config_names("edinburgh-dawg/mmlu-redux-2.0")
            self.data = []
            for topic in topics:
                topic_data = load_dataset("edinburgh-dawg/mmlu-redux-2.0", topic)[split]
                self.data.append(topic_data)
            self.data = concatenate_datasets(self.data)
        else:
            self.data = load_dataset("edinburgh-dawg/mmlu-redux-2.0", topic)[split]

        if remove_errors:
            self.data = self.data.filter(lambda x: x['error_type'] == 'ok')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
         
        prompt = prompt_template.format(question=item['question'])
        for letter, choice in zip(self.labels, item['choices']):
            prompt += f"{letter}) {choice}\n"
        return {
            'prompt': prompt.strip(),
            'label': item['answer'],
        }


class MMLUPro(Dataset):
    labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
    def __init__(self, split='test'):
        if split not in ['test', 'validation', 'train']:
            raise ValueError(f"Unknown split: {split}")
        if split == 'train':
            split = 'validation' # MMLUPro does not have a train set so we use validation as train
        self.data = load_dataset("TIGER-Lab/MMLU-Pro")[split]

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        prompt = prompt_template.format(question=item['question'])
        for letter, choice in zip(self.labels, item['options']):
            prompt += f"{letter}) {choice}\n"

        return {
            'prompt': prompt.strip(),
            'label': item['answer_index'],
            'question_id': item['question_id'],
        }
