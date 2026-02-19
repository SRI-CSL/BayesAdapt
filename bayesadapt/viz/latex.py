import pandas as pd
from .style import wrapper2label, label2wrapper, metric2arrow
from . import query

def latex_escape(s: str) -> str:
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
    return "".join(repl.get(ch, ch) for ch in str(s))


def _dataset_values(df: pd.DataFrame) -> list[str]:
    if "dataset" in df.columns:
        return df["dataset"].astype(str).tolist()
    if df.index.names and "dataset" in df.index.names:
        return df.index.get_level_values("dataset").astype(str).tolist()
    raise KeyError("Expected dataset in df['dataset'] or an index level named 'dataset'.")


def _ood_tests_for_train(ood_df: pd.DataFrame, train_ds: str) -> list[str]:
    seen = set()
    out: list[str] = []
    for ds in _dataset_values(ood_df):
        if "/" not in ds:
            continue
        tr, te = ds.split("/", 1)
        if tr != train_ds:
            continue
        if te == train_ds:
            continue  # ID comes from id_df
        if te not in seen:
            seen.add(te)
            out.append(te)
    return out


def _pick_single_row(qdf: pd.DataFrame) -> pd.DataFrame:
    if len(qdf) <= 1:
        return qdf
    pick_col = "num_trainable_params" if "num_trainable_params" in qdf.columns else None
    if pick_col:
        qdf = qdf.sort_values(pick_col, ascending=False)
    return qdf.iloc[[0]]


def make_latex_ood_table(
    id_df: pd.DataFrame,
    ood_df: pd.DataFrame,
    model: str,
    rank: int,
    train_datasets: list[str],
    *,
    prompt_type: str = "instruct",
    quant: str = "16bit",
    metrics: list[str] = ("ACC", "ECE", "NLL"),
    methods_map: dict[str, str] = label2wrapper,
    caption: str | None = None,
    na_cell: str = r"\textemdash{}",
    header_layout: str = "compact",  # "compact" or "grouped_ood"
    in_dist_label: str = "In-dist",
    ood_suffix: str = " (OOD)",
    ood_group_label: str = "OOD (test set)") -> str:
    """
    Wide format:
      Metric | Method | [Train on ds1: ...] | [Train on ds2: ...] | ...

    header_layout:
      - "compact": subcolumns are "In-dist" and "<test> (OOD)".
      - "grouped_ood": adds an extra header row, with OOD columns grouped under "OOD (test set)".
    """
    if header_layout not in {"compact", "grouped_ood"}:
        raise ValueError("header_layout must be one of {'compact', 'grouped_ood'}")

    groups: list[tuple[str, list[str]]] = [(tr, _ood_tests_for_train(ood_df, tr)) for tr in train_datasets]
    group_widths = [1 + len(tests) for _, tests in groups]  # 1 for In-dist + OOD tests
    total_eval_cols = sum(group_widths)

    ncols = 2 + total_eval_cols
    col_spec = "@{}" + ("c" * ncols) + "@{}"

    if caption is None:
        caption = f"OOD Performance Comparison ({latex_escape(model)}, rank {rank})"

    lines: list[str] = []
    lines.append(r"\begin{table*}[h!]")
    lines.append(r"\centering")
    lines.append(f"\\caption{{{latex_escape(caption)}}}")
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append(r"\toprule")

    # Header row 1: Train-on groups
    hdr1 = [r"\textbf{Metric}", r"\textbf{Method}"]
    for (tr, _tests), w in zip(groups, group_widths):
        hdr1.append(f"\\multicolumn{{{w}}}{{c}}{{\\textbf{{Train on {latex_escape(tr)}}}}}")
    lines.append(" & ".join(hdr1) + r" \\")

    # cmidrules under each group
    start = 3
    cmids = []
    for w in group_widths:
        end = start + w - 1
        cmids.append(f"\\cmidrule(lr){{{start}-{end}}}")
        start = end + 1
    lines.append(" ".join(cmids))

    # Sub-headers
    if header_layout == "compact":
        hdr2 = ["", ""]
        for _tr, tests in groups:
            hdr2.append(f"\\textbf{{{latex_escape(in_dist_label)}}}")
            for te in tests:
                hdr2.append(f"\\textbf{{{latex_escape(te)}{latex_escape(ood_suffix)}}}")
        lines.append(" & ".join(hdr2) + r" \\")
        lines.append(r"\midrule")

    else:  # grouped_ood
        # Row 2: In-dist + OOD super-header
        hdr2 = ["", ""]
        for _tr, tests in groups:
            hdr2.append(f"\\textbf{{{latex_escape(in_dist_label)}}}")
            if tests:
                hdr2.append(f"\\multicolumn{{{len(tests)}}}{{c}}{{\\textbf{{{latex_escape(ood_group_label)}}}}}")
        lines.append(" & ".join(hdr2) + r" \\")

        # Row 3: (blank under In-dist) + OOD dataset names
        hdr3 = ["", ""]
        for _tr, tests in groups:
            hdr3.append("")  # under In-dist
            for te in tests:
                hdr3.append(f"\\textbf{{{latex_escape(te)}}}")
        lines.append(" & ".join(hdr3) + r" \\")
        lines.append(r"\midrule")

    methods = list(methods_map.items())

    for mi, metric in enumerate(metrics):
        arrow_cmd = "uparrow" if metric2arrow[metric] == "↑" else "downarrow"
        metric_cell = f"\\textbf{{{latex_escape(metric)} ($\\{arrow_cmd}$)}}"

        for meth_i, (display_name, wrapper) in enumerate(methods):
            row_cells: list[str] = []

            # Put multirow on the SAME line as the first method (fixes alignment)
            if meth_i == 0:
                row_cells.append(f"\\multirow{{{len(methods)}}}{{*}}{{{metric_cell}}}")
            else:
                row_cells.append("")

            row_cells.append(latex_escape(display_name))

            for tr, tests in groups:
                # In-dist cell (from id_df where dataset == tr)
                q_id = query(
                    id_df,
                    model=model,
                    rank=rank,
                    prompt_type=prompt_type,
                    quant=quant,
                    dataset=tr,
                    wrapper=wrapper,
                )
                if q_id.empty:
                    row_cells.append("TBD")
                else:
                    q_id = _pick_single_row(q_id)
                    mean = q_id[(metric, "mean")].item()
                    std = q_id[(metric, "std")].item()
                    row_cells.append(f"{mean:.3f}$_{{\\pm {std:.3f}}}$")

                # OOD cells (from ood_df where dataset == f"{tr}/{te}")
                for te in tests:
                    q_ood = query(
                        ood_df,
                        model=model,
                        rank=rank,
                        prompt_type=prompt_type,
                        quant=quant,
                        dataset=f"{tr}/{te}",
                        wrapper=wrapper,
                    )
                    if q_ood.empty:
                        row_cells.append("TBD")
                    else:
                        q_ood = _pick_single_row(q_ood)
                        mean = q_ood[(metric, "mean")].item()
                        std = q_ood[(metric, "std")].item()
                        
                        if pd.isna(std):
                            row_cells.append(f"{mean:.3f}")
                        else:
                            row_cells.append(f"{mean:.3f}$_{{\\pm {std:.3f}}}$")

            lines.append(" & ".join(row_cells) + r" \\")

        if mi != len(metrics) - 1:
            lines.append(r"\midrule")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)




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
    wrappers=None,
) -> str:
    ncols = 2 + len(datasets)
    col_spec = "@{}" + ("c" * ncols) + "@{}"

    if wrappers is None:
        wrappers = methods_map.values()


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
    
    #method_display_names = list(methods_map.keys())
    method_display_names = [wrapper2label.get(w, w) for w in wrappers]

    for mi, metric in enumerate(metrics):
        arrow = "\\uparrow" if metric2arrow[metric] == '↑' else "\\downarrow"
        lines.append(f"\\multirow{{{len(method_display_names)}}}{{*}}{{\\textbf{{{latex_escape(metric)} ($${arrow}$$)}}}}".replace("$$", "$"))
        
        # for display_name, wrapper in methods_map.items():
        for wrapper in wrappers:
            display_name = wrapper2label.get(wrapper, wrapper)
            row = [f"& {latex_escape(display_name)}"]


            for dataset in datasets:
                qdf = query(
                    id_df, 
                    model=model, 
                    rank=rank if wrapper != "zeroshot" else 0,
                    prompt_type=prompt_type,
                    quant=quant, 
                    dataset=dataset,
                    wrapper=wrapper
                )

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

                mean = qdf[(metric, "mean")].item()
                std = qdf[(metric, "std")].item()

                if pd.isna(std):
                    row.append(f"& {mean:.3f}")
                else:
                    row.append(f"& {mean:.3f}$_{{\\pm {std:.3f}}}$")

            lines.append(" ".join(row) + " \\\\")

        if mi != len(metrics) - 1:
            lines.append("\\midrule")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")
    return "\n".join(lines)
