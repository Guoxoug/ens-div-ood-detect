import numpy as np
from sklearn.metrics import roc_auc_score 
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import average_precision_score
import torch

import torch.nn as nn

DOUBLE_INFO = torch.finfo(torch.double)
JITTERS = [0, DOUBLE_INFO.tiny] + [10 ** exp for exp in range(-308, 0, 1)]
METRIC_NAME_MAPPING = {
    "confidence": "MSP",
    "entropy": "$\\mathcal{H}$",
    "ens_entropy": "$Ens. \\mathcal{H}$",
    "energy": "Energy",
    "KU": "MI",
    "DU": "Av. $\\mathcal{H}$",
    "av_energy": "Av. Energy"
}
def get_metric_name(unc):
    if unc in METRIC_NAME_MAPPING:
        return METRIC_NAME_MAPPING[unc]
    else:
        return unc


def entropy(probs: torch.Tensor, dim=-1):
    "Calcuate the entropy of a categorical probability distribution."
    log_probs = probs.log()
    ent = (-probs*log_probs).sum(dim=dim)
    return ent


def print_results(results: dict):
    """Print the results in a results dictionary."""
    print("="*80)
    for k, v in results.items():
        if type(v) == float:
            print(f"{k}: {v:.2f}")
        else:
            print(f"{k}: {v}")
    print("="*80)


class ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """

    def __init__(self, n_bins=15, percent=True):
        """
        n_bins (int): number of confidence interval bins
        """
        super().__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]
        self.percent = percent

    def forward(self, labels, logits):
        softmaxes = logits.softmax(dim=-1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * \
                confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin -
                                 accuracy_in_bin) * prop_in_bin

        return 100 * ece.item() if self.percent else ece.item()


class TopKError(nn.Module):
    """
    Calculate the top-k error rate of a model. 
    """

    def __init__(self, k=1, percent=True):
        super().__init__()
        self.k = k
        self.percent = percent

    def forward(self, labels, outputs):
        # get rid of empty dimensions
        if type(labels) == np.ndarray:
            labels = torch.tensor(labels)
        if type(outputs) == np.ndarray:
            outputs = torch.tensor(outputs)
        labels, outputs = labels.squeeze(), outputs.squeeze()
        _, topk = outputs.topk(self.k, dim=-1)
        # same shape as topk with repeated values in class dim
        labels = labels.unsqueeze(-1).expand_as(topk)
        acc = torch.eq(labels, topk).float().sum(dim=-1).mean()
        err = 1 - acc
        err = 100 * err if self.percent else err
        return err.item()


def uncertainties(
    logits: torch.Tensor, features=None,
    ensemble=False,
) -> dict:
    """Calculate uncertainty measures from categorical output."""

    # increase precision
    logits = logits.type(torch.DoubleTensor)

    if ensemble:
        probs = logits.softmax(dim=-1)
        av_probs = probs.mean(dim=1)
        ent = entropy(av_probs)
        conf = av_probs.max(dim=-1).values

        # average over ensemble dim
        av_ent = entropy(probs, dim=-1).mean(dim=1)
        mutual_information = ent - av_ent      
        av_energy = -torch.logsumexp(logits, dim=-1).mean(dim=1)
        av_max_logit = -(logits.max(dim=-1).values.mean(dim=1))

        uncertainty = {
            "confidence": conf,
            "ens_entropy": ent,
            "DU": av_ent,
            "KU": mutual_information,
            "av_energy": av_energy,
            "av_max_logit": av_max_logit,
        }

    
    else:
        probs = logits.softmax(dim=-1)
        max_logit = -logits.max(dim=-1).values
        conf = probs.max(dim=-1).values
        ent = entropy(probs, dim=-1)
        energy = -torch.logsumexp(logits, dim=-1)


        uncertainty = {
            'confidence': conf,
            'entropy': ent, 
            "max_logit": max_logit,
            "energy": energy,
        }


    return uncertainty



def get_ood_metrics_from_combined(metrics, domain_labels):
    """Extract metrics only related to OOD data from combined data."""
    OOD_metrics = {}
    for key, metric in metrics.items():
        OOD_metrics[key] = metric[domain_labels == 1]

    return OOD_metrics


# code adapted from
# https://github.com/KaosEngineer/PriorNetworks/blob/master/prior_networks/assessment/misc_detection.py
# https://github.com/KaosEngineer/PriorNetworks/blob/master/prior_networks/assessment/rejection.py
# use of numpy and scipy due to use of sklearn in order to calculate AUC

# OOD detection ---------------------------------------------------------------

def fpr_at_recall(labels, scores, recall_level):
    """Get the false positive rate at a specific recall."""

    # positive is ID now
    labels = ~labels.astype(bool)
    scores = -scores
    precision, recall, thresholds = precision_recall_curve(
            labels, scores
    )

    # postive if >= threshold, recall and precision have an extra value
    # for 0 recall (all data classified as negative) at the very end
    # get threshold closest to specified (e.g.95%) recall
    cut_off = np.argmin(np.abs(recall-recall_level))
    t = thresholds[cut_off]


    negatives = ~labels 

    # get positively classified samples and filter
    fps = np.sum(negatives * (scores >= t))

    return fps/np.sum(negatives)


def ood_detect_results(
    domain_labels,
    metrics,
    mode="ROC",
    classes_flipped=None,
):
    """Evaluate OOD data detection using different uncertainty metrics."""

    # iterate over different metrics (e.g. mutual information)
    assert mode in ["PR", "ROC", "FPR@95"]
    domain_labels = np.asarray(domain_labels)
    results = {"mode": mode}
    for key in metrics.keys():
        pos_label = 1
        if key == 'confidence':
            pos_label = 0

        results[key] = ood_detect(
            domain_labels,
            metrics[key],
            mode=mode,
            pos_label=pos_label
        )

    return results


def ood_detect(
    domain_labels,
    metric,
    mode,
    pos_label=1,
    labels=None,
    preds=None
):
    """Calculate the AUPR or AUROC for OOD detection (binary)."""
    scores = metric
    scores = np.asarray(scores, dtype=np.float128)
    if pos_label != 1:
        scores *= -1.0
        
    # if there is overflow just clip to highest float
    scores = np.nan_to_num(scores) 
    # precision recall
    if mode == 'PR':
        n_positive = np.sum(domain_labels == 1)
        # precision, recall, thresholds = precision_recall_curve(
        #     domain_labels, scores
        # )
        aupr = average_precision_score(domain_labels, scores)

        # percent ABOVE RANDOm)
        return (aupr - n_positive/len(domain_labels)) * 100

    # receiver operating characteristic
    elif mode == 'ROC':
        # fpr, tpr, thresholds = roc_curve(domain_labels, scores)
        # symmetric so don't care 
        try:
            roc_auc = roc_auc_score(domain_labels, scores)

        # exception occurs when model is wrong on everything (e.g. ImageNet-A)
        except:
            roc_auc = 0

        # percent
        return roc_auc * 100

    elif mode == "FPR@95":
        recall_level = 0.95
        # note that the labels are reversed
        # OOD is positive for PR
        fpr = fpr_at_recall(domain_labels, scores, recall_level)
        # percent
        return fpr * 100
    


def ood_detect_early_exit(
    domain_labels,
    head_metrics,
    mode,
    pos_label=1
):
    """Evaluate the ood detection performance of an early exit network.
    As the threshold changes for what is and isn't OOD, so does which exit 
    the OOD detection may occur at. 
    """

    scale = -1 if pos_label != 1 else 1
    scores = {
        head: np.asarray(scores, dtype=np.float128) * scale
        for head, scores in 
        head_metrics.items()
    }


    # precision recall
    if mode == 'PR':
        n_positive = np.sum(domain_labels == 1)
        precision, recall, thresholds = precision_recall_curve(
            domain_labels, scores
        )
        aupr = auc(recall, precision)

        # percent ABOVE RANDOm)
        return (aupr - n_positive/len(domain_labels)) * 100

    # receiver operating characteristic
    elif mode == 'ROC':
        # fpr, tpr, thresholds = roc_curve(domain_labels, scores)
        roc_auc = roc_auc_score(domain_labels, scores)

        # percent
        return roc_auc * 100

def get_ee_pr(
    domain_labels, combined_uncs
):
    """Calculate PR and thresholds for multiple uncertainties."""
    unc_data = {}
    for unc in combined_uncs:
        data = {}
        scores = combined_uncs[unc]
        if unc == "confidence":
            scores = scores * -1
        precision, recall, thresholds = precision_recall_curve(
            domain_labels, scores
        )
        data["precision"] = precision
        data["recall"] = recall

        # NB positive is if value >= threshold
        data["thresholds"] = thresholds
        unc_data[unc] = data
    
    return unc_data


            

# Missclassification detection ------------------------------------------------

def err_at_recall_results(labels, logits, metrics):
    """Eval classification acc over different ood detection metrics."""
    results = {}
    for key in metrics.keys():
        rev = False
        if key == 'confidence':
            rev = True

        results[key] = err_at_recall(
            labels, logits, metrics[key], 
            rev=rev, recall=0.95 # going to stick to 95% 
        )

    return results



def err_at_recall(labels, logits, metric, rev=False, recall=0.95):
    """Evaluate the classification accuracy at a certain OOD detection recall."""
    n_samples = len(metric)
    preds = logits.argmax(dim=-1)

    # sort in terms of ascending uncertainty
    sorted_idx = metric.argsort(descending=rev)

    # extract subset of ID dataset, remove most uncertain
    lim = int(n_samples * recall)
    selected_preds = preds[sorted_idx][:lim]
    selected_labels = labels[sorted_idx][:lim]

    # get error rate on subset selected as ID
    err = (selected_preds!=selected_labels).sum()/n_samples * 100
    return err.item()


def rejection_ratio_results(labels, logits, metrics):
    """Evaluate missclassification detection using different metrics."""
    results = {}
    for key in metrics.keys():
        rev = False
        if key == 'confidence':
            rev = True

        results[key] = rejection_ratio(labels, logits, metrics[key], rev=rev)

    return results


def rejection_ratio(
        labels,
        logits,
        metric,
        rev: bool,
):
    """Calculate the PRR for a given metric."""
    # Get predictions from probabilities
    if type(logits) == torch.Tensor:
        logits = logits.to("cpu")
    if type(labels) == torch.Tensor:
        labels = labels.to("cpu")
    logits = np.asarray(logits)
    labels = np.asarray(labels)
    metric = np.asarray(metric)
    preds = np.argmax(logits, axis=-1)

    # rank results by metric
    if rev:
        inds = np.argsort(metric)[::-1]
    else:
        inds = np.argsort(metric)

    total_data = np.float(preds.shape[0])
    errors, percentages = [], []

    # get errors/misclassifications
    for i in range(preds.shape[0]):
        errors.append(
            np.sum(
                np.asarray(
                    labels[inds[:i]] != preds[inds[:i]], dtype=np.float32
                )
            ) * 100.0 / total_data
        )
        percentages.append(float(i + 1) / total_data * 100.0)

    errors, percentages = (
        np.asarray(errors)[:, np.newaxis],
        np.asarray(percentages)
    )

    base_error = errors[-1]
    n_items = errors.shape[0]
    auc_uns = 1.0 - auc(percentages / 100.0, errors[::-1] / 100.0)

    random_rejection = np.asarray(
        [
            base_error*(1.0 - float(i) / float(n_items))
            for i in
            range(n_items)
        ],
        dtype=np.float32
    )

    auc_rnd = 1.0 - auc(percentages / 100.0, random_rejection / 100.0)
    orc_rejection = np.asarray(
        [
            base_error*(1.0 - float(i) / float(base_error / 100.0*n_items))
            for i in
            range(int(base_error / 100.0 * n_items))
        ],
        dtype=np.float32
    )
    orc = np.zeros_like(errors)
    orc[0:orc_rejection.shape[0]] = orc_rejection
    auc_orc = 1.0 - auc(percentages / 100.0, orc / 100.0)

    errors = np.squeeze(errors)
    rejection_ratio = (auc_uns - auc_rnd) / (auc_orc - auc_rnd) * 100.0
    return rejection_ratio


