from utils.data_utils import DATA_NAME_MAPPING
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
pd.set_option("display.max_rows", 200,
              "display.max_columns", 10)
import numpy as np
from argparse import ArgumentParser
import json
from utils.train_utils import get_filename
from utils.eval_utils import METRIC_NAME_MAPPING
from models.model_utils import MODEL_NAME_MAPPING
import os

parser = ArgumentParser()
parser.add_argument(
    "config_path",
    help="path to the experiment config file for this test script"
)

parser.add_argument(
    "num_runs",
    type=int,
    help="number of independent runs to average over"
)

parser.add_argument(
    "--results_path",
    type=str,
    default=None,
    help=(
        "directory where result .csv files are kept," 
        "deduced from config by default"
    )
)

parser.add_argument(
    "--seeds",
    default=None,
    type=str,
    help="string containing random seeds, overrides default 1 to num_runs."
)

parser.add_argument(
    "--latex",
    type=int,
    default=1,
    help=(
        "whether to print datframe directly or to latex"
    )
)

parser.add_argument(
    "--std",
    type=int,
    default=1,
    help=(
        "whether to show standard deviations or not in latex table"
    )
)


args = parser.parse_args()


metrics = [" ROC", " FPR@95"] # space needed

# list of seeds
seeds = [i for i in range(1, args.num_runs + 1)] if (
    args.seeds is None
) else list(args.seeds)

EVAL_MAPPING = {
    "errROC": r"AUROC$\uparrow$",
    " ROC": r"AUROC$\uparrow$",
    "errFPR@95": r"FPR@95$\downarrow$",
    " FPR@95": r"FPR@95$\downarrow$"
}
higher = [True, False]

# load config
config = open(args.config_path)
config = json.load(config)

# list of seeds


# results path generated as results_savedir/arch_dataset
if args.results_path is not None:
    results_path = args.results_path
else:
    results_path = os.path.join(
        config["test_params"]["results_savedir"], 
        get_filename(config, seed=None)
    )


# metrics we care about
ens_metrics_of_interest = [
    "confidence",
    "ens_entropy",
    "DU",
    "KU",
    "av_energy"
]

sing_metrics_of_interest = [
    "confidence",
    "entropy",
    "energy"
]


# reformatting function
def rearrange_df(df, cols_to_drop, datasets_to_drop=[]):

    # get rid of specified colums
    df.drop(cols_to_drop, axis=1, inplace=True, errors="ignore")

    # get rid of certain data
    data_cols_to_drop = [
        col for col in df.columns
        if
        any(data_name in col for data_name in datasets_to_drop)
        and "fix" not in col
        or "-c" in col
        or "-r" in col
        
    ]
    df.drop(data_cols_to_drop, axis=1, inplace=True, errors="ignore")

    # drop shifted rows
    df.dropna(axis=1, inplace=True)
    
    df = df.transpose().reset_index(level=0)
    df.columns = ["data-method", "performance"]
    df[["data", "method"]] = df["data-method"].str.rsplit(
        " ", 1, expand=True
    )

    df.drop("data-method", axis=1, inplace=True, errors="ignore")
    df = df[["data", "method", "performance"]]

    def clean_data_name(name: str):
        for pattern in [
            " PR", " ROC", "OOD ", " FPR@95", " errROC", " errFPR@95", "err@95"
        ]:
            name = name.replace(pattern, "")

        return name

    # retain order of datasets

    df["data"] = df["data"].apply(clean_data_name)
    df["data"] = pd.Categorical(
        df["data"],
        categories=df["data"].unique(),
        ordered=True
    )
    df = df.pivot(
        index="method", columns="data", values="performance"
    )
    # nice names for the datasets
    df.columns = [
        col
        if col != config["id_dataset"]["name"] else "ID\\xmark"
        for col in df.columns
    ]
    df.columns = [
        DATA_NAME_MAPPING[col]
        if col in DATA_NAME_MAPPING else col.replace("_", " ")
        for col in df.columns
    ]

    return df

# ENSEMBLE csv
# get only FP row, outputs as a series so need to do a bit of messing around

ens_df = pd.read_csv(
    os.path.join(
        results_path, get_filename(config) + f"_ens.csv"
    ),  # results_savedir/arch_dataset/arch_dataset_ens.csv
    index_col=0
).iloc[0].drop(["dataset", "precision"], axis=0, errors="ignore")
ens_df = pd.DataFrame(ens_df).transpose()
ens_df = ens_df.astype(float)

# INDIVIDUAL CSV
dfs = [
    pd.DataFrame(
        pd.read_csv(
            os.path.join(
                results_path, get_filename(config, seed=seed) + ".csv"
            ),  # results_savedir/arch_dataset/arch_dataset_seed.csv
            index_col=0
        ).iloc[0].drop(
            ["weights", "activations", "dataset", "precision"],
            axis=0,
            errors="ignore"
        )
    ).transpose()
    for seed in seeds
]

# concatenate into a big boi
df = pd.concat(dfs)
# get mean and standard deviation over runs
df = df.astype(float)
mean = df.groupby(df.index).mean()
std = df.groupby(df.index).std()
id_mean = mean[["top1", "top5", "nll", "ece"]]
id_std = std[["top1", "top5", "nll", "ece"]]


mean_std = pd.concat([mean, std])
# combine into pretty tex


# format
def mean_std_format(data):
    """Take array [mean, std] and return formatted string."""
    data = np.array(data)
    if args.std and args.latex and len(data) > 1:
        return f"{data[0]:.2f} \scriptsize ±{2*data[1]:.1f}"
    elif args.std and len(data) > 1:
        return f"{data[0]:.2f} ± {2*data[1]:.2f}"
    else:
        return f"{data[0]:.2f}"



dfs = []
for i in range(2):
    mean = df.groupby(df.index).mean()
    ens_res = pd.DataFrame(ens_df.apply(mean_std_format, axis=0)).transpose()
    single_res = pd.DataFrame(mean_std.apply(mean_std_format, axis=0)).transpose()

    print("="*80)

    ens_cols_to_drop = [
        col for col in ens_res.columns 
        if metrics[i] not in col
    ] + ["seed"] 

    sing_cols_to_drop = [
        col for col in single_res.columns 
        if metrics[i] not in col
    ] + ["seed"] 

    data_to_drop = [
        
    ]

    # one is strings the other floats
    ens_res = rearrange_df(ens_res, ens_cols_to_drop, datasets_to_drop=data_to_drop)
    ens_data = rearrange_df(ens_df.copy(deep=True), ens_cols_to_drop, datasets_to_drop=data_to_drop)
    # one is strings the other floats
    single_res = rearrange_df(single_res, sing_cols_to_drop,
                           datasets_to_drop=data_to_drop)
    single_data = rearrange_df(mean.copy(deep=True),
                            sing_cols_to_drop, datasets_to_drop=data_to_drop)

    ens_res = ens_res.loc[ens_metrics_of_interest]
    single_res = single_res.loc[sing_metrics_of_interest]
    ens_data = ens_data.loc[ens_metrics_of_interest]
    single_data = single_data.loc[sing_metrics_of_interest]

    keys = [
        "\\begin{sideways}\\textbf{Single}\\end{sideways}",
        "\\begin{sideways}\\textbf{Ens.}\\end{sideways}"
    ]
    res = pd.concat(
        [single_res, ens_res], 
        keys=keys
    )
    mean = pd.concat(
        [single_data, ens_data], 
        keys=keys
    )

    mean = mean.mean(axis=1)

    mean = mean.apply(lambda x: f"{x:.2f}")
    res.insert(0,"OOD mean",mean)
    high = higher[i]

    def bold_max(data):
        data = list(data)
        means = [float(value.split(" ", maxsplit=1)[0]) for value in data]
        means = np.array(means)
        ids = np.argsort(means)
        if high:
            idx1 = ids[-1]
            idx2 = ids[-2]
            idx3 = ids[-3]
        else:
            idx1 = ids[0]
            idx2 = ids[1]
            idx3 = ids[2]
        data[idx1] = "\\textbf{" + data[idx1] + "}"
        data[idx2] = "\\underline{" + data[idx2] + "}"
        data[idx3] = "\\underline{" + data[idx3] + "}"
        return data

    # save for plot later
    if i == 1:
        plot_vals = res[
            ["iNaturalist", "Openimage-O", "Textures"]
        ].applymap(lambda x:x.split(" ")[0])
    
    # bold best, underline 2nd and 3rd best
    if args.latex:
        res = res.apply(bold_max, axis=0)
    res = res.transpose()

    # make text nicer
    res.columns = pd.MultiIndex.from_tuples([
        (colname[0], METRIC_NAME_MAPPING[colname[1]]) 
        if colname[1] in METRIC_NAME_MAPPING 
        else (colname[0], colname[1].replace("_", " "))
        for colname in res.columns
    ])

    dfs.append(res)
idx = res.index
comb = pd.concat(dfs, keys=[EVAL_MAPPING[metric] for metric in metrics])
comb = comb.swaplevel().reindex(idx, level=0).transpose()
def tidy_idx_cls(df):
    df.columns = pd.MultiIndex.from_tuples(
        [("\\textbf{" + x[0] + "}", x[1]) for x in df.columns]
    )
    df.index.names = [None, "\\textbf{Method}"]
    df_idx = df.index.to_frame()
    df_idx.insert(
        0, "\\textbf{Model}", 
        [
            f'\\begin{{sideways}}\\textbf{{{MODEL_NAME_MAPPING[config["model"]["model_type"]]}}}\\end{{sideways}}' 
            for metric in df_idx.iloc[:,0]
        ]
    )
    df.index = pd.MultiIndex.from_frame(df_idx)

    
if (
    config["id_dataset"]["name"] in ["imagenet200"]  and args.latex 
):
    data_names1 = [
        "OOD mean", 'Near-ImageNet-200', 
        'Caltech-45', 'Openimage-O', "iNaturalist"
    ]
    data_names2 = [
        "Textures", 'Colonoscopy', 
        'Colorectal', 'Noise', 'ImageNet-O'
    ]
    comb1 = comb[data_names1]
    comb2 = comb[data_names2]
    tidy_idx_cls(comb1)
    tidy_idx_cls(comb2)
    print(comb1.style.to_latex(hrules=True, multicol_align="c"))
    print("\n", 90*"=", "\n")
    print(comb2.style.to_latex(hrules=True, multicol_align="c"))
    print("\n", 90*"=", "\n")
    print(comb1.iloc[:,:4].style.to_latex(hrules=True, multicol_align="c"))

    print("\n", 90*"=", "\n")
    tidy_idx_cls(comb)
    print(comb.style.to_latex(hrules=True, multicol_align="c"))
else:
    # print entire dataframe
    with pd.option_context(
        'display.max_rows', None, 'display.max_columns', None
    ):
        print(comb)    



# plot one comparison on a bar chart
# comparing single model entropy with ensemble measures
sns.set_theme()
fig, ax = plt.subplots(1,1, figsize=(6,5))
x_labs = ["single $\mathcal{H}$", "ens. $\mathcal{H}$", "av. $\mathcal{H}$", "MI"]
plot_vals = plot_vals.iloc[[1,4,5,6],:]
plot_vals.index = x_labs
plot_vals["method"] = x_labs
plot_vals = plot_vals.melt(
    id_vars=["method"], 
    value_vars=["iNaturalist", "Openimage-O", "Textures"]
)
plot_vals["value"] = pd.to_numeric(plot_vals["value"])
plot_vals["OOD data"] = plot_vals["variable"]

sns.barplot(
    ax=ax, data=plot_vals, hue="OOD data", x="method", y="value",
    alpha=0.7, palette="mako"
)
ax.set_xlabel("method")
ax.set_ylabel("%FPR@95$\leftarrow$")
ax.get_xaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
ax.get_yaxis().set_minor_locator(mpl.ticker.AutoMinorLocator())
ax.grid(axis="y", which='major', color='w', linewidth=1.0)
ax.grid(axis="y", which='minor', color='w', linewidth=0.5)
ax.annotate(
    'ensemble',
    xy=(0.62, 1.01),
    xytext=(0.62, 1.06),
    xycoords='axes fraction',
    ha='center', va='bottom',
    arrowprops=dict(
        arrowstyle='-[, widthB=11, lengthB=1.3',
        lw=3,
        color="slategrey"
    ),
    color="slategrey"

)
ax.set_ylim(ymax=100)
fig.tight_layout()
spec = get_filename(config, seed=None)
save_dir = os.path.join(config["test_params"]["results_savedir"], spec)

# suffix is there for custom filename
filename = get_filename(config, seed=config["seed"]) +  \
    "_MI_illust.pdf"
path = os.path.join(save_dir, filename)
fig.savefig(path)
print(f"figure saved to:\n{path}")
