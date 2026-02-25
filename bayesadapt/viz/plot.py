import math
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from . import CLS_METRICS, query, RUNTIME_METRICS
from .style import metric2arrow, style_dict, wrapper2label, dataset2label
import numpy as np
import glob
import torch
from bayesadapt.datasets.slake import SLAKE
from bayesadapt.utils import average_log_probs
plt.rcParams["font.weight"] = "bold"
plt.rcParams["axes.labelweight"] = "bold"
plt.rcParams["figure.titleweight"] = "bold"
plt.rcParams["font.size"] = 14

def asym_err(center, lo, hi):
    return np.array([[center - lo], [hi - center]])  # shape (2,1) for matplotlib

def plot_with_err(x, y_mean, y_std, label=None, plot_kwargs=None, ax=None):
    sort_idx = np.argsort(x)
    x_sorted = np.array(x)[sort_idx]
    y_mean_sorted = np.array(y_mean)[sort_idx]
    ax.plot(x_sorted, y_mean_sorted, label=label, **plot_kwargs)
    if y_std is not None:
        y_std_sorted = np.array(y_std)[sort_idx]
        y_upper = y_mean_sorted + y_std_sorted
        y_lower = y_mean_sorted - y_std_sorted
        ax.fill_between(
            x_sorted,
            y_lower,
            y_upper,
            alpha=0.1,
            color=plot_kwargs['color']
        )
    return ax

def plot_winogrande(id_df, 
                    model='Qwen3-8B',
                    prompt_type='instruct', 
                    rank=8, 
                    quant='16bit',
                    save_path='plots/id_plot.png'):
    fig, axes = plt.subplots(1, len(CLS_METRICS), figsize=(25, 5), sharey=False)
    metrics = ['ACC', 'ECE', 'NLL', 'Brier']

    dataset_sizes = ['xs','s','m','l']
    x_vals = [160,640,2558,10234]
    model = 'Qwen3-8B'
    rank = 8

    # base_query_str = f"model == '{model}' and prompt_type == '{prompt_type}' and quant == '{quant}' and rank == {rank}"

    for ax, metric in zip(axes, CLS_METRICS):
        arrow = metric2arrow[metric]
        #for wrapper in ['mle', 'scalabl', 'blob', 'mcdropout', 'laplace','tfb', 'deepensemble', 'map', 'tempscale']:
        for wrapper in ['mle', 'map', 'tempscale', 'mcdropout','deepensemble','laplace','blob','scalabl','tfb']:
            label = wrapper2label[wrapper]
            y_mean, y_std = [], []
            for size in dataset_sizes:
                # query_str = base_query_str + f" and wrapper == '{wrapper}' and dataset == 'winogrande_{size}'"
                #metric_df = id_df.groupby(exp_keys)[metric].agg(['mean', 'std'])
                # q = id_df.query(query_str).reset_index()
                qdf = query(
                    id_df,
                    prompt_type=prompt_type,
                    wrapper=wrapper,
                    dataset=f'winogrande_{size}',
                    quant=quant,
                    rank=rank,
                    model=model
                )
                try:
                    y_mean.append(qdf[(metric, 'mean')].item())
                    y_std.append(qdf[(metric, 'std')].item())
                except:
                    continue
            ax = plot_with_err(x_vals[0:len(y_mean)], y_mean, None, plot_kwargs=style_dict[wrapper], label=label, ax=ax)
        ax.grid()
        ax.set_ylabel(f"{metric} ({arrow})")
        ax.set_xlabel('Training Set Size (# of Instances)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.05),
        ncols=9,          # adjust for readability
        frameon=True
    )
    fig.suptitle(f'Effect of Trianing Set Size using {model} and Winogrande',y=0.95)
    fig.subplots_adjust(bottom=0.15)
    for ax in axes:
         ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    fig.savefig(save_path, bbox_inches='tight', dpi=300)



def plot_resource(id_df, dataset, prompt_type='instruct', 
            rank=8, 
            quant='16bit',
            save_path='plots/id_plot.png'):
    fig, axes = plt.subplots(1, len(RUNTIME_METRICS), figsize=(10,5), sharey=False)#, gridspec_kw={"hspace": 0.05})

    xticks = [0, 2, 4, 8, 14]
    for ax, metric in zip(axes, RUNTIME_METRICS):
        arrow = metric2arrow[metric]
        for wrapper in ['mle','map', 'tempscale', 'mcdropout','deepensemble','laplace','blob','scalabl','tfb']:
            label = wrapper2label[wrapper]
            q = query(
                id_df, 
                prompt_type=prompt_type, 
                wrapper=wrapper, 
                dataset=dataset
            )
            x = q['num_base'] / 10**9
            ax = plot_with_err(
                x,
                q[(metric, 'mean')], 
                None, #std
                plot_kwargs=style_dict[wrapper], 
                label=label, 
                ax=ax
            )
        #ax.set_xticks(x)
        if metric == 'latency':
            ax.set_ylabel(f"Inference Time (seconds)")
        elif metric == 'peak_memory':
            ax.set_ylabel(f"Peak Memory (GB)")
        ax.set_xlabel('# Base Parameters (billions)')
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(t) for t in xticks])
        ax.grid()

    # for ax in axes[:-1]:
        # ax.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    # axes[-1].set_xlabel("# Base Parameters (billions)")


    # axes[-1].set_xticks(xticks)
    # axes[-1].set_xticklabels([str(t) for t in xticks])
    
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.05),
        ncols=5,
        frameon=True
    )
    # dataset_label = dataset2label.get(dataset, dataset)
    # if prompt_type == 'vlm':
        # family = 'Qwen3-VL'
    # else:
        # family = 'Qwen3'
    fig.suptitle(f'Resource Usage Comparsion',y=0.95)
    fig.subplots_adjust(bottom=0.21)
    # for ax in axes:
        # ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    fig.savefig(save_path, bbox_inches='tight', dpi=300)

def plot_id(id_df, dataset, prompt_type='instruct', 
            rank=8, 
            quant='16bit',
            save_path='plots/id_plot.png'):
    fig, axes = plt.subplots(1, len(CLS_METRICS), figsize=(25, 5), sharey=False)

    for ax, metric in zip(axes, CLS_METRICS):
        arrow = metric2arrow[metric]
        for wrapper in ['mle','map', 'tempscale', 'blob','tfb']:
            label = wrapper2label[wrapper]
            q = query(
                id_df, 
                prompt_type=prompt_type, 
                wrapper=wrapper, 
                dataset=dataset
            )
            x = q['num_base'] / 10**9
            ax = plot_with_err(
                x,
                q[(metric, 'mean')], 
                None, #std
                plot_kwargs=style_dict[wrapper], 
                label=label, 
                ax=ax
            )
        #ax.set_xticks(x)
        ax.set_xlabel('# Base Parameters (billions)')
        ax.set_ylabel(f"{metric} ({arrow})")
        ax.grid()

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.05),
        ncols=9,          # adjust for readability
        frameon=True
    )
    dataset_label = dataset2label.get(dataset, dataset)
    if prompt_type == 'vlm':
        family = 'Qwen3-VL'
    else:
        family = 'Qwen3'
    fig.suptitle(f'{family} Family on In-Distribution {dataset_label}',y=0.95)
    fig.subplots_adjust(bottom=0.15)
    for ax in axes:
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    fig.savefig(save_path, bbox_inches='tight', dpi=300)


def plot_active_learn(active_df,
                      dataset='srqa',
                      prompt_type='vlm',
                      rank=8,
                      quant='16bit',
                      model='Qwen3-VL-8B-Instruct',
                      save_path='plots/active_learn.png'):

    fig, axes = plt.subplots(1, len(CLS_METRICS), figsize=(25, 5), sharey=False)
    # plt.rcParams.update({'font.size': 12})

    for ax, metric in zip(axes, CLS_METRICS):
        arrow = metric2arrow[metric]
        # for wrapper in ['mle', 'scalabl', 'mcdropout','map','blob','laplace','tfb','deepensemble','tempscale']:
        for wrapper in ['mle', 'map', 'tempscale', 'mcdropout','deepensemble','laplace','blob','scalabl','tfb']:
            label = wrapper2label[wrapper]
            qdf = query(
                active_df, 
                prompt_type=prompt_type, 
                wrapper=wrapper, 
                dataset=dataset,
                model=model
            )
            try:
                y_mean = qdf[f'{metric}_mean'][0]
                y_std = qdf[f'{metric}_std'][0]
                y_std=None
            except:
                continue
            x = np.arange(len(y_mean)) + 1
            x *= 10
            ax = plot_with_err(
                x, y_mean, y_std, 
                plot_kwargs=style_dict[wrapper], 
                label=label, 
                ax=ax
            )
        ax.set_xlabel('# Labels Acquired')
        ax.set_ylabel(f"{metric} ({arrow})")
        ax.grid()

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.05),
        ncols=9,          # adjust for readability
        frameon=True
    )

    dataset_label = dataset2label.get(dataset, dataset)
    fig.suptitle(f'Active Learning using {model} on {dataset_label}',y=0.95)
    fig.subplots_adjust(bottom=0.15)
    for ax in axes:
       ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    fig.savefig(save_path, bbox_inches='tight', dpi=300)


def plot_noisy_slake(id_df, ood_df, 
    model='Qwen3-VL-8B-Instruct',
    rank=8, 
    quant='16bit',
    save_path='plots/noisy_slake.png'):

    fig, axes = plt.subplots(1, 3, figsize=(20, 5), sharey=False, gridspec_kw={"wspace": 0.45})
    noise_stds = [0,1,2,4,8,16,32,64,128]
    x = np.arange(len(noise_stds))
    dataset = 'slake'
    
    for ax, metric in zip(axes[0:2], ['ACC', 'NLL']):
        arrow = metric2arrow[metric]
        
        for wrapper in ['mle', 'map','tempscale','mcdropout','deepensemble','laplace','blob','scalabl','tfb']:
            label = wrapper2label[wrapper]
            y_mean, y_std = [], []
            for std in noise_stds:
                if std == 0:
                    dataset = 'slake'
                    metric_df = id_df
                else:
                    dataset = f'slake/noisy_slake{std}'
                    metric_df = ood_df
                    
                qdf = query(metric_df, dataset=dataset, prompt_type='vlm', wrapper=wrapper, quant=quant, rank=rank, model=model)
                try:
                    y_mean.append(qdf[(metric, 'mean')].item())
                    y_std.append(qdf[(metric, 'std')].item())
                except:
                    continue
            ax = plot_with_err(x[0:len(y_mean)], y_mean, None, plot_kwargs=style_dict[wrapper], label=label, ax=ax)
            ax.set_xlabel('Noise STD')
            
        ax.set_ylabel(f"{metric} ({arrow})")
        
        #ax.set_xscale('log', base=2)
        ax.set_xticks(x)
        ax.set_xticklabels(noise_stds)
        ax.grid()

    ds = SLAKE(split='test')
    id2label = {}
    for item in ds:
        qid = item['question_id']
        label = item['label']
        id2label[qid] = label

    qids = list(id2label.keys())
    labels = torch.tensor([id2label[qid] for qid in qids])

    root = f'/project/synthesis/bayesadapt/logs/Qwen/{model}/16bit/'
    for wrapper in ['mle', 'map','tempscale','mcdropout','deepensemble','laplace','blob','scalabl','tfb']:
        label = wrapper2label[wrapper]
        xs, ys = [], []

        for seed in [0,1,2,3]:
            path = f'{root}/{wrapper}/rank8/vlm/seed{seed}/slake/results/ood/noisy_slake128/'
            pt_files = glob.glob(f"{path}/*.pt")

            for pt_file in pt_files:
                logits_dict = torch.load(pt_file)
                logits = torch.stack([logits_dict[qid] for qid in qids]).to(torch.float64)

                logprobs = average_log_probs(logits)
                probs = torch.exp(logprobs)
                preds = probs.argmax(dim=-1)

                H = -torch.sum(probs * logprobs, dim=1) / math.log(2)
                is_wrong = preds != labels

                acc = (preds == labels).float().mean().item()
                if is_wrong.any():
                    H_mis = H[is_wrong].mean().item()
                    xs.append(H_mis)
                    ys.append(acc)

        print(wrapper, len(xs), len(ys))
        xs = np.array(xs)
        ys = np.array(ys)
        x_c = np.mean(xs)
        x_lo, x_hi = np.quantile(xs, [0.25, 0.75])
        y_c = np.mean(ys)
        y_lo, y_hi = np.quantile(ys, [0.025, 0.975])

        eb = axes[-1].errorbar(
            x_c, y_c,
            xerr=asym_err(x_c, x_lo, x_hi),
            yerr=asym_err(y_c, y_lo, y_hi),
            **style_dict[wrapper],
            # fmt='o', capsize=0, elinewidth=2, label=wrapper
        )
        for line in eb[2]:
            line.set_linestyle(style_dict[wrapper]['linestyle'])

    axes[-1].set_xlabel('Predictive Entropy of\nMisclassified Test Instances')
    axes[-1].set_ylabel('Test Set Acc (All Instances)')
    axes[-1].grid(True)

    # plt.xlabel('Predictive Entropy of Misclassified Test Instances')
    # plt.ylabel('Test Set Acc (All Instances)')
    # plt.grid(True)
    # plt.legend(loc='lower center', ncols=3, bbox_to_anchor=(0.5, -0.4))
        
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc='lower center',
        bbox_to_anchor=(0.5, -0.05),
        ncols=9,          # adjust for readability
        frameon=True
    )
    fig.suptitle(f'{model} on Noisy SLAKE OOD',y=0.95)
    fig.subplots_adjust(bottom=0.2)
    for ax in axes[0:2]:
       ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    fig.savefig(save_path, bbox_inches='tight', dpi=300)
