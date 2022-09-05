import torch
import torch.nn as nn
import os
import json 
import pandas as pd
import seaborn as sns
sns.set_theme()

from utils.eval_utils import (
    ECELoss, TopKError, print_results,
    rejection_ratio_results,
)
from utils.eval_utils import (
    ood_detect_results,
    uncertainties,
)
from utils.data_utils import (
    Data,
    get_preprocessing_transforms,
)

from argparse import ArgumentParser

from utils.train_utils import get_filename

# argument parsing
parser = ArgumentParser()

parser.add_argument(
    "config_path",
    help="path to the experiment config file for this test script"
)

parser.add_argument(
    "--logits_path",
    default=None,
    type=str,
    help="path to directory containing logits"
)

parser.add_argument(
    "--features_path",
    default=None,
    type=str,
    help="path to directory containing features"
)

parser.add_argument(
    "--seed",
    default=None,
    type=int,
    help="random seed, can be specified as an arg or in the config."
)


parser.add_argument(
    "--gpu",
    type=int,
    default=None,
    help="gpu override for debugging to set the gpu to use."
)

parser.add_argument(
    "--suffix",
    type=str,
    default="",
    help="added to end of filenames to differentiate them if needs be"
)



args = parser.parse_args()

# load config
config = open(args.config_path)
config = json.load(config)

# set random seed
# prioritize arg seed
if args.seed is not None:
    torch.manual_seed(args.seed)
    # add seed into config dictionary
    config["seed"] = args.seed
elif "seed" in config and type(config["seed"]) == int:
    torch.manual_seed(config['seed'])
else:
    torch.manual_seed(0)
    config["seed"] = 0


# set gpu
# bit of a hack to get around converting json syntax to bash
# deals with a list of integer ids
if args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
else:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(
        config["gpu_id"]
    ).replace("[", "").replace("]", "")
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"using device: {dev}")

# ood data truncation
if "ood_truncate" not in config["test_params"]:
    config["test_params"]["ood_truncate"] = False
ood_truncate = config["test_params"]["ood_truncate"]


# data-------------------------------------------------------------------------

id_data = Data(
    **config["id_dataset"],
    test_only=False,
    transforms=get_preprocessing_transforms(config["id_dataset"]["name"]),
    fast=False
)

test_loader = id_data.test_loader
train_loader = id_data.train_loader 




# ood_data
# get id dataset normalisation values
if "ood_datasets" in config:
    ood_data = [
        Data(
            **ood_config,
            transforms=get_preprocessing_transforms(
                ood_config["name"],
                id_dataset_name=config["id_dataset"]["name"]
            )
        )
        for ood_config in config["ood_datasets"]
    ]
else:
    ood_data = None


# print transforms
print("="*80)
print(id_data.name)
try:
    print(id_data.test_set.dataset.transforms)
except:
    print(id_data.test_set.transforms)
print("="*80)

for data in ood_data:
    print("="*80)
    print(data.name)
    try:
        print(data.test_set.dataset.transforms)
    except:
        print(data.test_set.transforms)
    print("="*80)




# helper functions ------------------------------------------------------------


def evaluate(
    logits_dict, id_data, precision,
    features_dict=None, ood_data=None, 
    feature_threshold=torch.inf, 
):
    """Evaluate the logits topk error rate and ECE."""
    ece = ECELoss()
    top1 = TopKError(k=1, percent=True)
    top5 = TopKError(k=5, percent=True)
    nll = nn.CrossEntropyLoss()
    print(f"eval on: {id_data.name}")

    labels = torch.tensor(id_data.test_set.targets)
    # we are assuming the same as in test.py
    # that this is at a single precision
    logits = logits_dict[id_data.name]
    if features_dict is not None:
        features = torch.clamp(
            features_dict[id_data.name], max=feature_threshold
        )
    else:
        features =  None
    


    results = {}
    results["dataset"] = id_data.name
    results["top1"] = top1(labels, logits)
    results["top5"] = top5(labels, logits)
    results["nll"] = nll(logits, labels).item() # backwards
    results["ece"] = ece(labels, logits)
   

    # average uncertainties
    print("calculating uncertainties on ID test set")
    metrics = uncertainties(
        logits, 
        features=features, 
    )
    res = {
        f"{id_data.name} {k}": v.mean().item()
        for k, v in metrics.items()
    }
    results.update(res)

    # misclassification detection
    rej_ratio_res = rejection_ratio_results(labels, logits, metrics)
    rej_ratio_res = {
        "PRR " + k: v for k, v in rej_ratio_res.items()
    }
    
    results.update(rej_ratio_res)


    if ood_data is not None and config["test_params"]["ood_data"]:
        ood_results = {}
        for data in ood_data:
            print(f"eval on: {data.name}")
            ood_logits = logits_dict[data.name]
            if features_dict is not None:
                ood_features = features_dict[data.name]
            else:
                ood_features = None


            # balance the #samples between OOD and ID data
            # unless OOD dataset is smaller than ID, then it will stay smaller
            if ood_truncate:
                ood_logits = ood_logits[:len(logits)]

            combined_logits = torch.cat([logits, ood_logits])
            
            # ID 0, OOD 1
            domain_labels = torch.cat(
                [torch.zeros(len(logits)), torch.ones(len(ood_logits))]
            )

            # optional features
            if ood_features is not None:
                if ood_truncate:
                   ood_features = ood_features[:len(features)]
                combined_features = torch.cat([features, ood_features])
            else: 
                combined_features = None
            # gets different uncertainty metrics for combined ID and OOD
            metrics = uncertainties(
                combined_logits, 
                features=combined_features
            )

            # average uncertainties
            res = {
                f"{data.name} {k}": v.mean().item()
                for k, v in metrics.items()
            }
            ood_results.update(res)

            # OOD detection
            res = ood_detect_results(
                domain_labels, metrics, mode="PR"
            )

            res = {
                f"OOD {data.name} PR " + k: v 
                for k, v in res.items() 
                if k != "mode"
            }
            ood_results.update(res)

            res = ood_detect_results(
                domain_labels, metrics, mode="ROC"
            )

            res = {
                f"OOD {data.name} ROC " + k: v 
                for k, v in res.items() 
                if k != "mode"
            }
            ood_results.update(res)

            res = ood_detect_results(
                domain_labels, metrics, mode="FPR@95"
            )

            res = {
                f"OOD {data.name} FPR@95 " + k: v 
                for k, v in res.items() 
                if k != "mode"
            }
            ood_results.update(res)

        results.update(ood_results)

    results["precision"] = precision
    return results




# evaluation-------------------------------------------------------------------

# load logits
results_path = os.path.join(
    config["test_params"]["results_savedir"],
    get_filename(config, seed=None)
)

if args.logits_path is None:
    logits_path = os.path.join(
            results_path, 
            get_filename(config, seed=config["seed"]) + "_logits.pth"
        )  # results_savedir/arch_dataset/arch_dataset_seed_logits.pth
        
else:
    logits_path = args.logits_path


# these are actually dictionaries
# containing many difference quantization levels
print("Loading logits")
logits_dict = torch.load(logits_path)
print(f"logit precisions: {logits_dict.keys()}")
print("Loading complete")

if config["test_params"]["features"]:
    if args.features_path is None:
        features_path = os.path.join(
            results_path,
            get_filename(config, seed=config["seed"]) + "_features.pth"
        )  # results_savedir/arch_dataset/arch_dataset_seed_logits.pth

    else:
        features_path = args.features_path


    # these are actually dictionaries
    # containing many difference quantization levels
    print("Loading logits")
    features_dict = torch.load(features_path)
    print("Loading complete")

    print(f"feature precisions: {features_dict.keys()}")
else:
    features_dict = {k: None for k in logits_dict.keys()}

# early exit
# NB quantization is NOT supported
if (
    "early_exit_params" in config["model"]
    and
    config["model"]["early_exit_params"]
):

    multihead = True
    print("evaluating multiheaded network")

else:
    multihead = False
    print("evaluating single network")


# list of results dictionaries
result_rows = []


# eval floating point model
fp_results = evaluate(
    logits_dict["afp, wfp"], id_data, "afp, wfp",
    features_dict=features_dict["afp, wfp"], ood_data=ood_data
)

fp_results["seed"] = config["seed"]
fp_results["activations"] = "fp"
fp_results["weights"] = "fp"
print("floating point" + 80*"=")
print_results(fp_results)
result_rows.append(fp_results)

print(f"datasets: {logits_dict['afp, wfp'].keys()}")


# results into DataFrame
result_df = pd.DataFrame(result_rows)

# save to subfolder with dataset and architecture in name
# filename will have seed 
if config["test_params"]["results_save"]:
    spec = get_filename(config, seed=None)
    filename = get_filename(config, seed=config["seed"])
    save_dir = os.path.join(config["test_params"]["results_savedir"], spec)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    savepath = os.path.join(save_dir, f"{filename}{args.suffix}.csv")

    # just overwrite what's there
    result_df.to_csv(savepath, mode="w", header=True)
    print(f"results saved to \n{savepath}")



    

