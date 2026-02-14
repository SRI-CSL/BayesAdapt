from bayesadapt.viz.latex import make_latex_table, make_latex_ood_table
from bayesadapt.viz import load_df, reduce_seeds, load_json, query, CLS_METRICS
from bayesadapt.viz.style import style_dict, metric2arrow, wrapper2label

root = '/project/synthesis/bayesadapt/logs/'


ood_df = load_df(root=root, mode='ood')

id_df = load_df(root=root, mode='id')

latex = make_latex_ood_table(
    id_df, 
    ood_df, 
    model="Qwen3-VL-8B-Instruct",
    rank=8, 
    prompt_type="vlm",
    quant="16bit", 
    metrics=["ACC", "ECE", "NLL", "Brier"], 
    #train_datasets=["circuit_logic", "expression_logic"],
    train_datasets=["slake"],
    header_layout='grouped_ood'
)

fname = f"tables/noisy_slake_Qwen3-VL-8B-Instruct.tex"
with open(fname, 'w') as f:
    f.write(latex)


# fname = f"tables/logic_Qwen3-8B.tex"
# with open(fname, 'w') as f:
    # f.write(latex)



sizes = [0.6, 1.7, 4, 8, 14]
for size in sizes:
    model = f"Qwen3-{size}B"
    latex = make_latex_table(
        id_df,
        model=model,
        rank=8,
        datasets=["ARC-Easy", "ARC-Challenge", "obqa"],
        prompt_type="instruct",
        quant="16bit",
        metrics=["ACC", "ECE", "NLL", "Brier"],
    )

    fname = f"tables/arc_id_{model}.tex"
    with open(fname, 'w') as f:
        f.write(latex)

for size in sizes:
    model = f"Qwen3-{size}B"
    latex = make_latex_table(
        id_df,
        model=model,
        rank=8,
        datasets=["winogrande_xs", "winogrande_s", "winogrande_m", "winogrande_l"],
        prompt_type="instruct",
        quant="16bit",
        metrics=["ACC", "ECE", "NLL", "Brier"],
    )

    fname = f"tables/winogrande_id_{model}.tex"
    with open(fname, 'w') as f:
        f.write(latex)


sizes = [2,4,8]
for size in sizes:
    model = f"Qwen3-VL-{size}B-Instruct"
    latex = make_latex_table(
        id_df,
        model=model,
        rank=8,
        datasets=["slake", "mmstar", "MathVerse"],
        prompt_type="vlm",
        quant="16bit",
        metrics=["ACC", "ECE", "NLL", "Brier"],
    )

    fname = f"tables/vlm_id_{model}.tex"
    with open(fname, 'w') as f:
        f.write(latex)

for size in sizes:
    model = f"Qwen3-VL-{size}B-Instruct"
    latex = make_latex_table(
        id_df,
        model=model,
        rank=8,
        datasets=["srqa"],
        prompt_type="vlm",
        quant="16bit",
        metrics=["ACC", "ECE", "NLL", "Brier"],
    )

    print(latex)


