import pandas as pd
from .style import wrapper2label, label2wrapper, metric2arrow
from . import query

def latex_escape(s: str) -> str:
    # at minimum underscore; plus a few common LaTeX specials
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    out = []
    for ch in str(s):
        out.append(repl.get(ch, ch))
    return "".join(out)

# def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    # df = df.copy()
    # if isinstance(df.columns, pd.MultiIndex):
        # flat = []
        # for a, b in df.columns.to_flat_index():
            # if b is None or b == "":
                # flat.append(str(a))
            # else:
                # flat.append(f"{a}_{b}")
        # df.columns = flat
    # return df

# def _prep_df(id_df: pd.DataFrame) -> pd.DataFrame:
    # df = id_df.reset_index() if isinstance(id_df.index, pd.MultiIndex) else id_df.copy()
    # return _flatten_columns(df)

def _format_pm(mean: float, std: float, metric: str) -> str:
    # If ACC/ECE look like fractions, auto-convert to percent
    if metric in ("ACC", "ECE") and mean <= 1.0:
        mean *= 100.0
        std *= 100.0

    # per-metric formatting (tweak if you want)
    if metric in ("ACC", "ECE"):
        return f"${mean:.2f}_{{\\pm {std:.1f}}}$"
    else:
        return f"${mean:.3f}_{{\\pm {std:.3f}}}$"


def make_latex_table(
    id_df: pd.DataFrame,
    model: str,
    rank: int,
    datasets: list[str],
    *,
    prompt_type: str = "instruct",
    quant: str = "16bit",
    metrics: list[str] = ("ACC", "ECE", "NLL"),
    methods_map: dict[str, str] = label2wrapper,
    caption: str | None = None,
) -> str:
    # df = _prep_df(id_df)

    # sanity: required columns
    # required = {"model", "rank", "prompt_type", "quant", "dataset", "wrapper"}
    # missing = required - set(df.columns)
    # if missing:
        # raise ValueError(f"id_df is missing required columns after prep: {sorted(missing)}")

    ncols = 2 + len(datasets)
    col_spec = "@{}" + ("c" * ncols) + "@{}"

    ds_headers = [latex_escape(d) for d in datasets]
    header = (
        "\\textbf{Metric} & \\textbf{Method} & "
        + " & ".join([f"\\textbf{{{h}}}" for h in ds_headers])
        + " \\\\\n"
    )

    if caption is None:
        caption = f"Performance Comparison ({latex_escape(model)}, rank {rank})"

    lines = []
    lines.append("\\begin{table*}[h!]")
    lines.append("\\centering")
    lines.append(f"\\caption{{{caption}}}")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")
    lines.append(header.rstrip("\n"))
    lines.append("\\midrule")
    
    method_display_names = list(methods_map.keys())

    for mi, metric in enumerate(metrics):
        arrow = "\\uparrow" if metric2arrow[metric] == '↑' else "\\downarrow"
        lines.append(f"\\multirow{{{len(method_display_names)}}}{{*}}{{\\textbf{{{latex_escape(metric)} ($${arrow}$$)}}}}".replace("$$", "$"))

        for display_name, wrapper_code in methods_map.items():
            row = [f"& {latex_escape(display_name)}"]

            for ds in datasets:
                # sub = df[
                    # (df["model"] == model)
                    # & (df["rank"] == rank)
                    # & (df["prompt_type"] == prompt_type)
                    # & (df["quant"] == quant)
                    # & (df["dataset"] == ds)
                    # & (df["wrapper"] == wrapper_code)
                # ]

                qdf = query(id_df, model=model, rank=rank, prompt_type=prompt_type, quant=quant, dataset=ds, wrapper=wrapper_code)

                #if sub.empty:
                if qdf.empty:
                    row.append("& TBD")
                    continue

                # if multiple rows match (e.g., because num_* params differ), pick a stable choice
                #if len(sub) > 1:
                if len(qdf) > 1:
                    import ipdb; ipdb.set_trace() # noqa
                    pick_col = "num_trainable_params" if "num_trainable_params" in sub.columns else None
                    if pick_col:
                        sub = sub.sort_values(pick_col, ascending=False)
                    sub = sub.iloc[[0]]

                # mean_col = f"{metric}_mean"
                # std_col = f"{metric}_std"
                # if mean_col not in sub.columns or std_col not in sub.columns:
                    # row.append("& TBD")
                    # continue

                # mean = float(sub[mean_col].iloc[0])
                # std = float(sub[std_col].iloc[0])
                mean = qdf[(metric, "mean")].item()
                std = qdf[(metric, "std")].item()
                row.append(f"& {_format_pm(mean, std, metric)}")

            lines.append(" ".join(row) + " \\\\")

        if mi != len(metrics) - 1:
            lines.append("\\midrule")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")
    return "\n".join(lines)
