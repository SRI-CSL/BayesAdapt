import json
import glob
import pandas as pd
import numpy as np

CLS_METRICS = ['ACC', 'ECE', 'NLL', 'Brier']
RUNTIME_METRICS = ['peak_memory', 'latency']
METRICS = CLS_METRICS + RUNTIME_METRICS

PARAM_KEYS = ['num_base', 'num_trainable_params', 'num_total_params']
MODEL_KEYS = ['model', 'quant', 'wrapper', 'rank', 'prompt_type', 'dataset'] 
EXP_KEYS = MODEL_KEYS + PARAM_KEYS

def load_json(fname):
    try:
        with open(fname, 'r') as f:
            data = json.load(f)
        return data
    except:
        return []

def find_expdirs(root, mode='id'):
    if mode == 'id':
        json_fnames = glob.glob(f'{root}/**/id/metrics.json', recursive=True)
    elif mode == 'ood':
        json_fnames = glob.glob(f'{root}/**/ood/**/metrics.json', recursive=True)
    elif mode == 'active_learn':
        json_fnames = glob.glob(f'{root}/**/active_learn/results.json', recursive=True)
    else:
        pass
        
    expdirs = []
    for fname in json_fnames:
        tokens = fname.split('/')
        edir = '/'.join(tokens[0:-1])
        expdirs.append(edir)
    expdirs = list(set(expdirs))
    return expdirs

def reduce_vectors(series: pd.Series) -> pd.Series:
    arrs = [np.asarray(v, dtype=float) for v in series]
    lengths = {a.shape for a in arrs}
    if len(lengths) != 1:
        raise ValueError(f"Vector shapes differ within group: {lengths}")
    stacked = np.stack(arrs, axis=0)          # (n_seeds, T)
    return pd.Series({
        "mean": stacked.mean(axis=0),
        "std":  stacked.std(axis=0, ddof=1) if stacked.shape[0] > 1 else np.zeros(stacked.shape[1]),
    })
    
def reduce_seeds(df, mode='id'):
    if mode == 'id' or mode == 'ood':
        df_exploded = df.explode('results').reset_index(drop=True)
        metrics_df = pd.json_normalize(df_exploded['results']).drop(columns=['seed'])
        df_seeds = pd.concat([df_exploded.drop(columns=['results']), metrics_df], axis=1)
        df = df_seeds.groupby(EXP_KEYS)[METRICS].agg(['mean', 'std'])
    elif mode == 'active_learn':
        agg_parts = []
        for m in METRICS:
            tmp_df = df.groupby(EXP_KEYS)[m].apply(reduce_vectors).unstack()
            tmp_df.columns = [f"{m}_{c}" for c in tmp_df.columns]
            agg_parts.append(tmp_df)
        df = pd.concat(agg_parts, axis=1).reset_index()
        df = df.set_index(EXP_KEYS)
    return df

def load_df(root, mode='id', reduce=True):
    expdirs = find_expdirs(root, mode=mode)
    df = []
    for edir in expdirs:
        tokens = edir.replace(root, '').split('/')
        if mode == 'id' or mode == 'active_learn':
            keys = ['model', 'quant', 'wrapper', 'rank', 'prompt_type', 'seed', 'dataset']
            row = dict(zip(keys, tokens[1:]))
        elif mode == 'ood':
            keys = ['model', 'quant', 'wrapper', 'rank', 'prompt_type', 'seed']
            row = dict(zip(keys, tokens[1:-2]))
            trainset = tokens[-4] #name of orinigal id trainset
            testset = tokens[-1] #name of ood testset
            row['dataset'] = f"{trainset}/{testset}"
        row['rank'] = int(tokens[4].replace('rank', ''))
        row['seed'] = int(tokens[6][-1])
        if mode == 'id' or mode == 'ood':
            data = load_json(f'{edir}/metrics.json')
            row['results'] = data
        elif mode == 'active_learn':
            data = load_json(f'{edir}/results.json')
            for metric in METRICS:
                row[metric] = [item['test_metrics'][0][metric] for item in data]
            for param_key in PARAM_KEYS:
                row[param_key] = data[0]['test_metrics'][0][param_key]
        df.append(row)
    df = pd.DataFrame(df)
    if reduce:
        df = reduce_seeds(df, mode=mode)
    return df
    
def query(df, model=None, dataset=None, wrapper=None, prompt_type='instruct', quant='16bit', rank=8):
    query_str = f"prompt_type == '{prompt_type}' and quant == '{quant}' and rank == {rank}"
    if model is not None:
        query_str += f" and model == '{model}'"
    if dataset is not None:
        query_str += f" and dataset == '{dataset}'"
    if wrapper is not None:
        query_str += f" and wrapper == '{wrapper}'"
    q = df.query(query_str).reset_index()
    return q
