from bayesadapt.viz.latex import make_latex_table, make_latex_ood_table
from bayesadapt.viz import load_df, reduce_seeds, load_json, query, CLS_METRICS
from bayesadapt.viz.style import style_dict, metric2arrow, wrapper2label
from bayesadapt.viz import plot

root = '/project/synthesis/bayesadapt/logs/'
id_df = load_df(root=root, mode='id')
ood_df = load_df(root=root, mode='ood')
active_df = load_df(root=root, mode='active_learn')


sizes = [0.6, 1.7, 4, 8, 14]

plot.plot_resource(
    id_df, 
    dataset='winogrande_s',
    prompt_type='instruct',
    save_path=f'figs/resource_usage.png'
)

plot.plot_id(
    id_df, 
    dataset='obqa',
    prompt_type='instruct',
    save_path=f'figs/obqa_id.png'
)


sizes = [2, 4, 8]
for size in sizes:
    model = f'Qwen3-VL-{size}B-Instruct'
    plot.plot_noisy_slake(
        id_df, 
        ood_df, 
        model=model,
        save_path=f'figs/noisy_slake_{model}.png'
    )

    plot.plot_active_learn(
        active_df,
        dataset='srqa',
        prompt_type='vlm',
        model=model,
        save_path=f'figs/srqa_active_learn_{model}.png'
    )
