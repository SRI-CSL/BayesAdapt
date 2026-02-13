from transformers import AutoModelForCausalLM, AutoTokenizer
from bayesadapt.datasets.mmlu import MMLU
from bayesadapt.datasets.collate import instruct_collate_fn
import torch
import json
from tqdm import tqdm

model_name = "Qwen/Qwen3-235B-A22B-Instruct-2507-FP8"
# model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"
#dataset = MMLU(split='test', topic='machine_learning', remove_errors=True)
dataset = MMLU(split='test', topic=None, remove_errors=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.padding_side = "left"
item = dataset[0]

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    num_workers=0,
    collate_fn=lambda batch: instruct_collate_fn(tokenizer, batch)
)


# load the tokenizer and the model
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto",
    device_map="auto",
    low_cpu_mem_usage=True
)

target_ids = tokenizer.convert_tokens_to_ids(dataset.labels)
num_correct = 0
num_seen = 0
for batch in tqdm(dataloader):
    inputs, labels = batch
    inputs = inputs.to(model.device)

    # generated_ids = model.generate(
        # **inputs,
        # max_new_tokens=1,
        # do_sample=True,
        # temperature=0.7,
        # top_p=0.8,
        # top_k=20,
        # min_p=0.0,
        # num_return_sequences=10,
    # )
    # print(tokenizer.batch_decode(generated_ids[:,-1]))
    # new_ids = generated_ids[:, inputs['input_ids'].shape[1]:]
    # generated_texts = tokenizer.batch_decode(new_ids, skip_special_tokens=True)
    # try:
        # answer_dicts = [json.loads(text) for text in generated_texts]
    # except json.JSONDecodeError:
        # pass
    # import ipdb; ipdb.set_trace() # noqa

    with torch.no_grad():
        output = model(**inputs)
    class_logits = output.logits[:, -1, target_ids]  # (batch_size, num_classes)
    preds = torch.argmax(class_logits, dim=-1)
    is_correct = (preds == labels.to(model.device)).sum().item()
    num_correct += is_correct
    num_seen += len(labels)
    # print('Pred:', dataset.labels[preds])
    # print('GT:', dataset.labels[labels])
    print(f"Current accuracy: {num_correct / num_seen:.4f} ({num_correct}/{num_seen})")
import ipdb; ipdb.set_trace() # noqa

# prepare the model input
prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=100
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

content = tokenizer.decode(output_ids, skip_special_tokens=True)

import ipdb; ipdb.set_trace() # noqa
