# debugging
if __name__ == "__main__":
    from resnet import ResNet
else:
    from models.resnet import ResNet


import re
from tqdm import tqdm

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from collections import OrderedDict
from typing import Dict, Optional

MODEL_TYPES = [
    "resnet", 
    "resnet50"
]
MODEL_TYPE_MAPPINGS = {
    "resnet":ResNet,
    "resnet50": ResNet,

}
MODEL_NAME_MAPPING = {
    "resnet50": "ResNet-50",
}


def model_generator(model_type:str, **model_params) -> torch.nn.Module:
    """Construct a model following the supplied parameters."""
    assert model_type in MODEL_TYPES, (
        f"model type not supported"
        f"needs to be in {MODEL_TYPES}"    
    )

    # select model class
    Model = MODEL_TYPE_MAPPINGS[model_type]

    # override with proper values

    if model_type == "resnet50":
        model_params["layers"] = [3, 4, 6, 3]
        model_params["block"] = "bottleneck"



    # generic unpacking of pararmeters, need to match config file with 
    # model definition
    model = Model(**model_params)

    return model

def load_weights_from_file(
    model, weights_path, dev="cuda", keep_last_layer=True, 
):
    """Load parameters from a path of a file of a state_dict."""

    state_dict = torch.load(weights_path, map_location=dev)

   

    new_state_dict = OrderedDict()

    # data parallel trained models have module in state dict
    # prune this out of keys
    for k, v in state_dict.items():
        name = k.replace("module.", "")
        new_state_dict[name] = v
    # load params
    state_dict = new_state_dict
    if not keep_last_layer:

        # filter out final linear layer weights
        state_dict = {
            key: params for (key, params) in state_dict.items()
            if "classifier" not in key and "fc" not in key
        }
        model.load_state_dict(state_dict, strict=False)
    else:
        print("loading weights")
        model.load_state_dict(state_dict, strict=True)
       