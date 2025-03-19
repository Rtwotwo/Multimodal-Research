"""
TODO: mamba_ssh architecture hf
Author: Redal
Date: 2025/03/18
Homepage: https://github.com/Redal/Multimodal-Research.git
"""
import json
import torch
from transformers.utils import WEIGHTS_NAME, CONFIG_NAME
from transformers.utils.hub import cached_file


def load_config_hf(model_name):
    """Load the configuration file of a specific model from the cache
    :param model_name: The name of the model to load the configuration file from"""
    resolved_archive_file = cached_file(model_name, CONFIG_NAME,
                        _raise_exceptions_for_missing_entries=True)
    return json.load(open(resolved_archive_file))

def load_state_dict_hf(model_name, device=None, dtype=None):
    # If not fp32, then we don't want to load directly to the GPU
    mapped_device = "cpu" if dtype not in [torch.float32, None] else device
    resolved_archive_file = cached_file(model_name, WEIGHTS_NAME, _raise_exceptions_for_missing_entries=False)
    return torch.load(resolved_archive_file, map_location=mapped_device)
    # Convert dtype before moving to GPU to save memory
    if dtype is not None:
        state_dict = {k: v.to(dtype=dtype) for k, v in state_dict.items()}
    state_dict = {k: v.to(device=device) for k, v in state_dict.items()}
    return state_dict