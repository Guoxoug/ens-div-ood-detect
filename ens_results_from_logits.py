"""Given the single models have already run, use their saved logits to 
extract results for the whole ensemble."""

import torch
import os
import pandas as pd
import json

from utils.eval_utils import (
    ood_detect_results,
    uncertainties,
    ECELoss, TopKError,
    rejection_ratio_results,
)
from utils.data_utils import (
    Data,
    get_preprocessing_transforms,
)
from argparse import ArgumentParser
from utils.train_utils import get_filename


# argument parsing ------------------------------------------------------------
parser = ArgumentParser()
parser.add_argument(
    "config_path",
    help="path to the experiment config file for this test script"
)

parser.add_argument(
    "--num_runs",
    default=5,
    type=int,
    help="number of members in ensemble"
)

parser.add_argument(
    "--seeds",
    default=None,
    type=str,
    help="string containing random seeds, overrides default 1 to num_runs."
)


parser.add_argument(
    "--gpu",
    type=int,
    default=None,
    help="gpu override for debugging to set the gpu to use."
)

parser.add_argument(
    "--suffix",
    default="",
    help="a suffix to differentiate a file"
)


args = parser.parse_args()

# load config
config = open(args.config_path)
config = json.load(config)

# ood data truncation
if "ood_truncate" not in config["test_params"]:
    config["test_params"]["ood_truncate"] = False
ood_truncate = config["test_params"]["ood_truncate"]

# determinism in testing
torch.backends.cudnn.benchmark = False


# evaluation functions --------------------------------------------------------


def evaluate_ens(
    logits_dict, labelled_data, precision, 
    ood_data=None, id_data_name=None, data_name=None
):
    """Evaluate the ensemble's performance."""

    if id_data_name is None:
        id_data_name = labelled_data.name

    data_name = labelled_data.name if data_name is None else data_name
    ece = ECELoss()
    top1 = TopKError(k=1, percent=True)
    top5 = TopKError(k=5, percent=True)
    nll = torch.nn.CrossEntropyLoss()

    print(f"eval on: {data_name}")
    try:
        labels = torch.tensor(labelled_data.test_set.targets)
    except:
        labels = torch.tensor(id_data.test_set.targets)
    # we are assuming the same as in test.py
    # that this is at a single precision
    logits = logits_dict[data_name]
    log_av_probs = logits.softmax(dim=-1).mean(dim=-2).log()
    # get log probs of average predictive distribution
    results = {}
    results["dataset"] = data_name
    results["top1"] = top1(labels, log_av_probs)
    results["top5"] = top5(labels, log_av_probs)
    results["nll"] = nll(log_av_probs, labels).item()  # backwards
    results["ece"] = ece(labels, log_av_probs)

    # average uncertainties
    metrics = uncertainties(logits, ensemble=True)
    res = {
        f"{data_name} {k}": v.mean().item()
        for k, v in metrics.items()
    }
    results.update(res)

    # misclassification detection
    rej_ratio_res = rejection_ratio_results(labels, log_av_probs, metrics)
    rej_ratio_res = {
        "PRR " + k: v for k, v in rej_ratio_res.items()
    }

    results.update(rej_ratio_res)

    # OOD data detection
    if ood_data is not None and config["test_params"]["ood_data"]:
        ood_results = {}
        id_logits = logits_dict[id_data_name]
        for data in ood_data:
            
            ood_data_name = data.name
            print(f"eval on: {ood_data_name}")
            ood_logits = logits_dict[ood_data_name]
            # balance the #samples between OOD and ID data
            # unless OOD dataset is smaller than ID, then it will stay smaller
            if ood_truncate:
                ood_logits = ood_logits[:len(logits)]

            combined_logits = torch.cat([id_logits, ood_logits])
            # ID 0, OOD 1
            domain_labels = torch.cat(
                [torch.zeros(len(id_logits)), torch.ones(len(ood_logits))]
            )

            # gets different uncertainty metrics for combined ID and OOD
            metrics = uncertainties(combined_logits, ensemble=True)

            # average uncertainties
            res = {
                f"{ood_data_name} {k}": v.mean().item()
                for k, v in metrics.items()
            }
            ood_results.update(res)

            # OOD detection
            res = ood_detect_results(
                domain_labels, metrics, mode="PR"
            )

            res = {
                f"OOD {ood_data_name} PR " + k: v
                for k, v in res.items()
                if k != "mode"
            }
            ood_results.update(res)

            res = ood_detect_results(
                domain_labels, metrics, mode="FPR@95"
            )

            res = {
                f"OOD {ood_data_name} FPR@95 " + k: v
                for k, v in res.items()
                if k != "mode"
            }
            ood_results.update(res)

            res = ood_detect_results(
                domain_labels, metrics, mode="ROC"
            )

            res = {
                f"OOD {ood_data_name} ROC " + k: v
                for k, v in res.items()
                if k != "mode"
            }
            ood_results.update(res)

        results.update(ood_results)

    results["precision"] = precision

    return results

if __name__ == "__main__":

# data-------------------------------------------------------------------------

    id_data = Data(
        **config["id_dataset"],
        test_only=False,
        transforms=get_preprocessing_transforms(config["id_dataset"]["name"]),
        fast=False
    )

    test_loader = id_data.test_loader
    train_loader = id_data.train_loader  # for ptq calibration

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
    print(logits[0]["afp, wfp"].keys())
    

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
            ensemble_logits = torch.stack(ensemble_list, dim=0).transpose(0,1)
            ensemble_logit_dict[precision][dataname] = ensemble_logits


    # now evaluate on logits
    result_rows = []
    print(ensemble_logit_dict["afp, wfp"].keys())
    for precision in ensemble_logit_dict:
        result_rows.append(
            evaluate_ens(
                ensemble_logit_dict[precision], 
                id_data,
                precision,
                ood_data=ood_data,
                id_data_name=id_data.name
            )
        )
        break 
    

    # results into DataFrame
    result_df = pd.DataFrame(result_rows)

    # save to subfolder with dataset and architecture in name
    # filename will have seed
    if config["test_params"]["results_save"]:
        spec = get_filename(config, seed=None)
        filename = get_filename(config) + "_ens"
        save_dir = os.path.join(config["test_params"]["results_savedir"], spec)
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        savepath = os.path.join(save_dir, f"{filename}{args.suffix}.csv")

        # just overwrite what's there
        result_df.to_csv(savepath, mode="w", header=True)
