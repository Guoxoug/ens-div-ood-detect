"""Illustrative plot to show difference between different combination methods.
Tuned for paper presentation, so quite a specific script.
"""
from utils.plot_utils import (
    plot_uncs_conditional,
    plot_uncs_conditional_together
)
from utils.train_utils import get_filename
import torch
import os
import json
import numpy as np
import seaborn as sns
from utils.eval_utils import uncertainties
from argparse import ArgumentParser
from utils.data_utils import get_preprocessing_transforms, Data




sns.set_theme()

parser = ArgumentParser()
parser.add_argument(
    "config_path",
    help="path to the experiment config file for this test script"
)
parser.add_argument(
    "--num_runs",
    default=5,
    type=int
)
parser.add_argument(
    "--logits_path",
    type=str,
    default=None,
    help=(
        "directory where result logit files are kept,"
        "deduced from config by default"
    )
)

parser.add_argument(
    "--seeds",
    default=None,
    type=str,
    help="random seed, used to load model outputs."
)


parser.add_argument(
    "--suffix",
    type=str,
    default="",
    help="filename suffix to make file unique if needs be"
)


args = parser.parse_args()

# load config
config = open(args.config_path)
config = json.load(config)

# load logits -----------------------------------------------------------------
# list of seeds
seeds = [i for i in range(1, args.num_runs + 1)] if (
    args.seeds is None
) else list(args.seeds)

# results path generated as results_savedir/arch_dataset
results_path = os.path.join(
    config["test_params"]["results_savedir"],
    get_filename(config, seed=None)
)
logits_paths = [
    os.path.join(
        results_path, get_filename(config, seed=seed) + f"_logits.pth"
    )  # results_savedir/arch_dataset/arch_dataset_seed_logits.pth
    for seed in seeds
]

# these are actually dictionaries
# containing many difference quantization levels
print("Loading logits")
logits = [
    torch.load(path) for path in logits_paths
]
print("Loading complete")
print(logits[3]["afp, wfp"].keys())
# we want the ensemble dimension to be moved into the tensors
# ensemble member; precision; dataset
ensemble_logit_dict = {}
# precision
for precision in logits[0]:
    ensemble_logit_dict[precision] = {}
    # dataset
    for dataname in logits[0][precision]:
        # member
        ensemble_list = []
        for logit_precision_dict in logits:
            ensemble_list.append(
                logit_precision_dict[precision][dataname]
            )

            # place ensemble dimension after batch
            ensemble_logits = torch.stack(ensemble_list, dim=0).transpose(0, 1)
            ensemble_logit_dict[precision][dataname] = ensemble_logits



id_data = Data(
    **config["id_dataset"],
    test_only=False,
    transforms=get_preprocessing_transforms(config["id_dataset"]["name"]),
    fast=False
)

labels = torch.tensor(id_data.test_set.targets)

# just do one for now
# just do floating point
logits = ensemble_logit_dict["afp, wfp"]


# quick fix to filter out corrupted datasets
logits = {k: v for k, v in logits.items() if "-c" not in k}


S1_name = "DU"
S2_name = "KU"
# logits and features

ood_data_names = [
    "inaturalist",
    "openimage-o",
    "textures",
    "near-imagenet200",
    "caltech256",
    "imagenet-o",
    "colonoscopy",
    "colorectal",
    "imagenet-noise"
]

id_logits = logits[id_data.name]
# ensemble take mean of probimagenet-os,

preds = id_logits.softmax(dim=-1).mean(dim=-2).max(dim=-1).indices
correct_logits = id_logits[preds==labels]
incorrect_logits = id_logits[preds!=labels]
ood_logits = [ 
    logits[ood_data_name] for ood_data_name in ood_data_names
]


# uncertainties
correct_uncs = uncertainties(
    correct_logits,  ensemble=True
)
incorrect_uncs = uncertainties(
    incorrect_logits,  ensemble=True
)
ood_uncs = [uncertainties(
    logits, ensemble=True
) for logits in ood_logits]

id_uncs = uncertainties(
    id_logits, ensemble=True
)

all_uncs = [id_uncs] + ood_uncs[:3]
all_names = ["ImageNet-200 (ID)", "iNaturalist", "Openimages-O","Textures"]
unc_names = ["DU", "KU"]
unc_range = [
    [0, np.log(id_logits.shape[-1])],
    [0, 1]
]

plot_uncs_conditional_together(
    all_uncs,
    all_names,
    unc_names,
    config,
    unc_range=unc_range,
    suffix="OOD"
)


all_uncs = [id_uncs] + ood_uncs[3:6]
all_names = ["ImageNet-200 (ID)",    "Near-Imagenet-200",
             "Caltech-45",
             "ImageNet-O",
]
plot_uncs_conditional_together(
    all_uncs,
    all_names,
    unc_names,
    config,
    unc_range=unc_range,
    suffix="OOD_extra1"
)

all_uncs = [id_uncs] + ood_uncs[6:9]
all_names = ["ImageNet-200 (ID)",              
            "Colonoscopy",
             "Colorectal",
             "Noise"]
plot_uncs_conditional_together(
    all_uncs,
    all_names,
    unc_names,
    config,
    unc_range=unc_range,
    suffix="OOD_extra2"
)
