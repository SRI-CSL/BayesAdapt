import json
import time
from openai import OpenAI
from pydantic import BaseModel, ValidationError
from bayesadapt.datasets.mmlu import MMLUPro
from tqdm import trange, tqdm
from typing import Literal

class MultipleChoiceAnswer(BaseModel):
    answer: Literal['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']

def ask_question(client, prompt, num_retries=3):
    for attempt in range(num_retries):
        try:
            response = client.responses.parse(
                model='gpt-5.2',
                input=prompt,
                text_format=MultipleChoiceAnswer,
                reasoning={'effort': 'medium'},
                tools=[],
            )
            return response
        except ValidationError as e:
            print(f"Validation error on attempt {attempt + 1}/{num_retries}: {e}")
    return None


dataset = MMLUPro(split='test')

results_fname = "mmlupro_gpt52_medium.jsonl"

num_correct = 0
num_failed = 0
client = OpenAI()
pbar = trange(len(dataset))
for i in pbar:
    prompt = dataset[i]['prompt']
    label_idx = dataset[i]['label']
    question_id = dataset[i]['question_id']
    start = time.time()
    response = ask_question(client, prompt)
    end = time.time()

    if response is not None:
        predicted_answer = response.output_parsed.answer
        is_correct = predicted_answer == dataset.labels[label_idx]
        num_correct += int(is_correct)
    else:
        num_failed += 1
    
    acc = num_correct / (i + 1)

    pbar.set_description(f"Acc: {acc:.4f}, Failed: {num_failed}")
    
    result = {
        'question_id': question_id,
        'response': response.model_dump() if response else None,
        'time_taken': end - start,
    }
    with open(results_fname, 'a') as f:
        f.write(json.dumps(result) + '\n')
