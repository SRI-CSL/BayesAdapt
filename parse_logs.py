import re
import glob
import pandas as pd
import argparse

DATASETS = ['ARC-Challenge', 'ARC-Easy', 'boolq', 'obqa', 'winogrande_m', 'winogrande_s']

parser = argparse.ArgumentParser()
parser.add_argument('root', type=str, default='checkpoints_final/')
args = parser.parse_args()


def extract_metrics(log_fname):
    with open(log_fname, 'r') as f:
        log_line = f.readlines()[0].strip()
    pattern = r'(\w+): ([\d\.]+)' #chatGPT
    matches = re.findall(pattern, log_line)
    result = {name: float(value) for name, value in matches}
    result['NLL'] = result['val_nll']
    result['ECE'] = result['val_ece'] * 100
    result['ACC'] = result['val_acc'] * 100
    result = {k: v for k, v in result.items() if 'val' not in k}
    return result


methods = glob.glob(f'{args.root}/*')
methods = [method.split('/')[-1] for method in methods]
for method in methods:
    print(method)
    for dataset in DATASETS:
        print(dataset)
        log_fnames = glob.glob(f'{args.root}/{method}/meta-llama/Llama-2-7b-hf/{dataset}/*/log.txt')
        metrics = [extract_metrics(log_fname) for log_fname in log_fnames]
        df = pd.DataFrame(metrics)
        means = df.mean().to_dict()
        stds = df.std().to_dict()
        for metric in means.keys():
            mean_value = means[metric]
            std_value = stds[metric]
            formatted_value = f"${mean_value:.2f}_{{\pm {std_value:.2f}}}$"
            print(formatted_value)
