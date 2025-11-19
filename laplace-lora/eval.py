import json
import torch
import torch.nn.functional as F
from ece  import expected_calibration_error
import argparse
import torchmetrics
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("json_fnames", nargs='+', help="json file containing logits and labels")
args = parser.parse_args()

# json_fname ="outputs/winogrande_s/meta-llama/Llama-2-7b-chat-hf_lora_lmhead_16_0.1_5e-05_21/step_4999/eval_res_la_kron_last_layer_homo_mc_corr_100.json"

def parse_json(json_fname):
    with open(json_fname, 'r') as f:
        data = f.read().splitlines()
        data = [eval(line) for line in data]

    probs = torch.tensor([x['logits'] for x in data]) #"logits" already have softmax on them!
    # probs = torch.softmax(logits, dim=-1)
    logits = torch.log(probs)
    labels = torch.tensor([x['true'] for x in data])

    preds = torch.argmax(probs, dim=1)
    acc = (preds == labels).float().mean().item()

    num_classes = probs.shape[-1]
    ece = torchmetrics.classification.MulticlassCalibrationError(num_classes)(probs, labels).item()

    #nll = F.cross_entropy(logits, labels)
    nll = F.nll_loss(logits, labels).item()
    return {'acc': acc*100, 'ece': ece*100, 'nll': nll}

results = []
for json_fname in args.json_fnames:
    try:
        result = parse_json(json_fname)
        results.append(result)
    except:
        print(json_fname, 'failed')

results = pd.DataFrame.from_dict(results)
print(results)
print(results.mean())
print(results.std())
