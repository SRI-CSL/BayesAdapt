import re
import glob
import pandas as pd
import argparse



parser = argparse.ArgumentParser()
parser.add_argument('root', type=str, default='checkpoints')
parser.add_argument('--ood', action='store_true')
args = parser.parse_args()

if args.ood:
    DATASETS = ['num_params','obqa', 'ARC-Challenge', 'ARC-Easy', 'MMLU-chem', 'MMLU-phy']
    latex_table = (
        "\\begin{table*}[h!]\n"
        "\\centering\n"
        "\\caption{Performance Comparison}\n"
        "\\begin{tabular}{@{}cccccccc@{}}\n"
        "\\toprule\n"
        "\\textbf{Metric} & \\textbf{Method} & \\textbf{Params (M)} & \\textbf{OBQA} & \\textbf{ARC-C} & \\textbf{ARC-E} & "
        "\\textbf{Chemistry} & \\textbf{Physics} \\\\\n"
        "\\midrule\n"
    )
else:
    DATASETS = ['num_params', 'winogrande_s', 'ARC-Challenge', 'ARC-Easy', 'winogrande_m', 'obqa', 'boolq']
    latex_table = (
        "\\begin{table*}[h!]\n"
        "\\centering\n"
        "\\caption{Performance Comparison}\n"
        "\\begin{tabular}{@{}ccccccccc@{}}\n"
        "\\toprule\n"
        "\\textbf{Metric} & \\textbf{Method} & \\textbf{Params (M)} & \\textbf{WG-S} & \\textbf{ARC-C} & \\textbf{ARC-E} & "
        "\\textbf{WG-M} & \\textbf{OBQA} & \\textbf{BoolQ} \\\\\n"
        "\\midrule\n"
    )


def extract_metrics(log_fname):
    with open(log_fname, 'r') as f:
        print(log_fname)
        try:
            log_line = f.readlines()[0].strip()
        except:
            print(f"Error reading {log_fname}")
            return {}
    pattern = r'(\w+): ([\d\.]+)'
    matches = re.findall(pattern, log_line)
    result = {name: float(value) for name, value in matches}
    result['NLL'] = result['val_nll']
    result['ECE'] = result['val_ece'] * 100
    result['ACC'] = result['val_acc'] * 100
    result = {k: v for k, v in result.items() if 'val' not in k}
    return result


#qwen2.5-7B
num_lora_param = 3768320 / 10**6
num_params_ours = 3769232 / 10**6
num_params_blob = 5402624 / 10**6
# num_params_ours_fullcov = 3776528 / 10**6
#3,769,232, no U
#2,134,928, no U random
#3772880 #full rank cov, no U

#ours lora 4 1884616
#ours lora 16 7538464 
#ours lora 32 15076928


#blob lora 16 10805248

#qwen2.5-32B
# num_lora_param = 9646080 / 10**6
# num_params_ours = 9656400 / 10**6
# num_params_ours = 9648144 / 10**6
# num_params_blob = 14929920 / 10**6



#llama2-7B
# num_lora_param = 4483072 / 10**6
# num_params_ours = 7538464 / 10**6
# num_params_blob = 10805248 / 10**6
# 4,484,112

#num_params_ours = 4488272 / 10**6 with U
# Initialize a structure to hold the LaTeX table data

methods_map = {'MLE': 'mle', 'MAP': 'map', 'MC-Dropout': 'mcdropout',
               'Ensemble': 'deepensemble', 'Laplace': 'lap', 'BLoB': 'blob', 'ScalaBL (ours)': 'scalabl'}

method2params = {'MLE': num_lora_param, 'MAP': num_lora_param, 'MC-Dropout': num_lora_param,
                 'Ensemble': 3*num_lora_param, 'Laplace': num_lora_param, 'BLoB': num_params_blob,
                 'ScalaBL (ours)': num_params_ours}

methods = glob.glob(f'{args.root}/*')
methods = [method.split('/')[-1] for method in methods]
table_data = {metric: {method: {dataset: "TBD" for dataset in DATASETS} for method in methods_map.keys()} for metric in ['ACC', 'ECE', 'NLL']}
print(methods)

for table_name in methods_map.keys():
    method = methods_map[table_name]
    for dataset in DATASETS:
        # log_fnames = glob.glob(f'{args.root}/{method}/meta-llama/Llama-2-7b-hf/{dataset}/*/log.txt')
        log_fnames = glob.glob(f'{args.root}/{method}/Qwen/Qwen2.5-7B/{dataset}/*/log.txt')
        metrics = []
        for log_fname in log_fnames:
            if 'sample5' in log_fname:
                continue
            if 'samples' in log_fname:
                continue
            if 'fullcov' in log_fname:
                continue
            if 'random' in log_fname:
                continue
            if 'ood' in log_fname and not args.ood:
                continue
            if not 'ood' in log_fname and args.ood:
                continue
            metrics.append(extract_metrics(log_fname))
        if metrics:
            df = pd.DataFrame(metrics)
            print(df.shape)
            means = df.mean().to_dict()
            stds = df.std().to_dict()
            for metric in ['ACC', 'ECE', 'NLL']:
                if metric in means:
                    mean_value = means[metric]
                    std_value = stds[metric]
                    # Corrected f-string with properly escaped backslashes
                    formatted_value = f"${mean_value:.2f}_{{\\pm {std_value:.1f}}}$"
                    table_data[metric][table_name][dataset] = formatted_value
                    
                    num_params = method2params[table_name]
                    formatted_value = f"${num_params:.3f}$"
                    table_data[metric][table_name]['num_params'] = formatted_value

# Generate the LaTeX table
for metric in table_data.keys():
    arrow = "\\uparrow" if metric == "ACC" else "\\downarrow"
    latex_table += "\\multirow{" + str(len(methods)) + "}{*}{\\textbf{" + metric + " ($" + arrow + "$)}}\n"
    for method in methods_map.keys():
        row = "& " + method + " "
        for dataset in DATASETS:
            value = table_data[metric][method].get(dataset, "TBD")
            row += "& " + value + " "
        row += "\\\\\n"  # End the row
        latex_table += row
    latex_table += "\\midrule\n"

latex_table += (
    "\\bottomrule\n"
    "\\end{tabular}\n"
    "\\end{table*}"
)


# Print the generated LaTeX table
print("\nGenerated LaTeX Table:")
print(latex_table)

# Optionally, save the LaTeX table to a file
if args.ood:
    name = "ood_metrics_table.tex"
else:
    name = "metrics_table.tex"
with open(name, "w") as f:
    f.write(latex_table)

