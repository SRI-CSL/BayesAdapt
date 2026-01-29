import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from bayesadapt.datasets.mmlu import MMLUPro

id2gt = {}
dataset = MMLUPro(split='test')
for item in dataset:
    id2gt[item['question_id']] = item['label']

results = []
json_fname = 'mmlupro_gpt52_medium.jsonl'
with open(json_fname, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        result = json.loads(line)
        results.append(result)

id2pred = {}
for result in results:
    pred_letter = result['response']['output'][-1]['content'][0]['parsed']['answer']
    pred_label = dataset.labels.index(pred_letter)
    id2pred[result['question_id']] = pred_label


question_ids = list(id2gt.keys())
pred_labels = torch.tensor([id2pred[qid] for qid in question_ids])
gt_labels = torch.tensor([id2gt[qid] for qid in question_ids])
full_acc = (pred_labels == gt_labels).float().mean().item()

def test_random_subset(pred_labels, gt_labels, subset_size=1000, n_trials=100):
    num_samples = pred_labels.size(0)
    accs = []
    for i in range(n_trials):
        indices = torch.randperm(num_samples)[0:subset_size]
        subset_pred = pred_labels[indices]
        subset_gt = gt_labels[indices]
        acc_i = (subset_pred == subset_gt).float().mean().item()
        acc_i = np.abs(acc_i - full_acc)
        accs.append(acc_i)
    accs = torch.tensor(accs)
    mean, std = accs.mean().item(), accs.std().item()
    return mean, std

sizes = np.arange(1,5001,1)
y_vals, y_errs = [], []
for size in tqdm(sizes):
    mean, std = test_random_subset(pred_labels, gt_labels, subset_size=size.item(), n_trials=100)
    y_vals.append(mean)
    y_errs.append(std)
y_vals = np.array(y_vals)
y_errs = np.array(y_errs)

percents = sizes / len(question_ids) 

plt.plot(percents, y_vals)
# plt.fill_between(sizes, y_vals - y_errs, y_vals + y_errs, alpha=0.2)
# plt.axhline(full_acc, color='red', linestyle='--', label='Full Set Accuracy')
plt.xlabel('Subset Size (% of Full Set)')
plt.ylabel('Absolute Error From Full Set Accuracy')
# plt.title('MMLUPro GPT-5.2 Medium Accuracy vs Subset Size')
plt.grid()
plt.savefig('test_fig.png')
