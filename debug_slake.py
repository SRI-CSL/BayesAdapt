from datasets import load_dataset

def get_binary_questions(split_name="train"):
    # 1. Load the dataset
    dataset = load_dataset("BoKelvin/SLAKE", split=split_name)
    
    # 2. Define the filter logic
    # We look for "yes" or "no" while ignoring case and extra spaces
    def is_yes_no(example):
        answer = str(example['answer'])
        return answer in ['Yes', 'No']

    # 3. Filter the dataset
    filtered_dataset = dataset.filter(is_yes_no)
    
    print(f"--- Split: {split_name} ---")
    print(f"Total questions: {len(dataset)}")
    print(f"Yes/No questions: {len(filtered_dataset)}")
    
    return filtered_dataset


def get_modality_questions(split_name='train'):
    dataset = load_dataset("BoKelvin/SLAKE", split=split_name)

    def is_modality_question(example):
        return example['content_type'] == 'Modality'
    filtered_dataset = dataset.filter(is_modality_question)
    print(f"--- Split: {split_name} ---")
    print(f"Total questions: {len(dataset)}")
    print(f"Modality questions: {len(filtered_dataset)}")
    return filtered_dataset

def is_chinese(example):
    return example['q_lang'] == 'zh'

def is_english(example):
    return example['q_lang'] == 'en'

dataset = load_dataset("BoKelvin/SLAKE", split='train')
english_dataset = dataset.filter(is_english)

# Execution
split = "train" # Change to "test" or "validation" as needed
binary_data = get_binary_questions('train')
binary_data = get_binary_questions('validation')
binary_data = get_binary_questions('test')
import ipdb; ipdb.set_trace() # noqa

binary_data = get_modality_questions('train')

# Display the first 5 examples
for i in range(min(5, len(binary_data))):
    print(f"\nQ: {binary_data[i]['question']}")
    print(f"A: {binary_data[i]['answer']}")
    print(f"Img: {binary_data[i]['img_name']}")


import ipdb; ipdb.set_trace() # noqa
