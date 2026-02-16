dataset2label = {
    'obqa': 'OpenBookQA',
    'ARC-Easy': 'ARC-Easy',
    'ARC-Challenge': 'ARC-Challenge',
    'srqa': 'SymbolicRegressionQA',
    'boolq': 'BoolQ',
}

metric2arrow = {
    'ACC': '↑',
    'ECE': '↓',
    'NLL': '↓',
    'Brier': '↓',
    'peak_memory': '↓',
    'latency': '↓',
}

wrapper2label = {
    'mle': 'MLE',
    'map': 'MAP',
    'tempscale': 'TempScale',
    'mcdropout': 'MCDropout',
    'deepensemble': 'Ensemble',
    'laplace': 'Laplace',
    'blob': 'BLoB',
    'scalabl': 'ScalaBL',
    'tfb': 'TFB',
}

label2wrapper = {v: k for k, v in wrapper2label.items()}

style_dict = {
    "mle": {
        "color": "#000000",
        "linestyle": (0, (5, 2)),   # dashed
        "marker": "o",
        "linewidth": 2.4,
        "markersize": 6.5,
        "markerfacecolor": "none",
        "markeredgewidth": 1.6,
    },
    "laplace": {
        "color": "#D55E00",
        "linestyle": "solid",
        "marker": "s",
        "linewidth": 2.6,
        "markersize": 6.2,
        "markerfacecolor": "#D55E00",
        "markeredgecolor": "white",
        "markeredgewidth": 0.8,
    },
    "map": {
        "color": "#7F7F7F",
        "linestyle": (0, (1, 1)),   # dotted
        "marker": "D",
        "linewidth": 2.2,
        "markersize": 6.0,
        "markerfacecolor": "none",
        "markeredgewidth": 1.4,
    },
    "tempscale": {
        "color": "#0072B2",
        "linestyle": "dashdot",
        "marker": "^",
        "linewidth": 2.4,
        "markersize": 6.8,
        "markerfacecolor": "#0072B2",
        "markeredgecolor": "white",
        "markeredgewidth": 0.8,
    },
    "blob": {
        #"color": "#CC79A7",
        "color": "#bf033c",
        "linestyle": (0, (3, 1, 1, 1)),  # short dash-dot pattern
        "marker": "P",                   # plus-filled
        "linewidth": 2.2,
        "markersize": 7.0,
        #"markerfacecolor": "#CC79A7",
        "markerfacecolor": "#bf033c",
        "markeredgecolor": "white",
        "markeredgewidth": 0.8,
    },
    "scalabl": {
        "color": "#009E73",
        "linestyle": (0, (7, 2)),   # long dashed
        "marker": "v",
        "linewidth": 2.6,
        "markersize": 6.8,
        "markerfacecolor": "none",
        "markeredgewidth": 1.6,
    },
    "tfb": {
        "color": "#56B4E9",
        "linestyle": (0, (2, 2)),   # evenly dashed
        "marker": "X",
        "linewidth": 2.3,
        "markersize": 6.8,
        "markerfacecolor": "#56B4E9",
        "markeredgecolor": "black",
        "markeredgewidth": 0.6,
    },
    "mcdropout": {
        "color": "#E69F00",
        "linestyle": (0, (1, 2)),   # spaced dots
        "marker": "<",
        "linewidth": 2.4,
        "markersize": 6.8,
        "markerfacecolor": "#E69F00",
        "markeredgecolor": "white",
        "markeredgewidth": 0.8,
    },
    "deepensemble": {
        "color": "#332288",
        "linestyle": (0, (4, 1, 1, 1, 1, 1)),  # dash-dot-dot
        "marker": ">",
        "linewidth": 2.4,
        "markersize": 6.8,
        "markerfacecolor": "none",
        "markeredgewidth": 1.6,
    },
}

style_dict_old = {
    'laplace': {'color': 'black', 'linestyle': '--', 'marker': '.'},
    'mle': {'color': 'red', 'linestyle': ':', 'marker': 'v'},
    'map': {'color': 'grey', 'linestyle': ':', 'marker': 'v'},
    'tempscale': {'color': 'blue', 'linestyle': 'dashdot', 'marker': 'o'},
    'blob': {'color': 'purple', 'linestyle': '--', 'marker': 's'},
    'scalabl': {'color': 'green', 'linestyle': 'solid', 'marker': '^'},
    'tfb': {'color': 'blue', 'linestyle': 'dashdot', 'marker': '^'},
    'mcdropout': {'color': 'orange', 'linestyle': 'dashdot', 'marker': 'v'},
    'deepensemble': {'color': 'teal', 'linestyle': 'dashdot', 'marker': 'v'},
}
