"""
TODO: mamba_ssh architecture torch
Author: Redal
Date: 2025/03/18
Homepage: https://github.com/Redal/Multimodal-Research.git
"""
import torch
from functools import partial
from typing import Callable


def custom_amp_decorator(dec:Callable, cuda_amp_deprecated:bool):
    """The custom_fwd and custom_bwd functions in the Automatic 
    Mixed Precision (AMP) function create a compatibility decorator
    dec: Functions that require decoration (such as custom_fwd or custom_bwd).
    cuda_amp_deprecated: Boolean value indicating whether torch.cuda.amp has been deprecated."""
    def decorator(*args, **kwargs):
        if cuda_amp_deprecated:
            kwargs['device_type'] = 'cuda'
        return dec(*args, **kwargs)
    return decorator

if hasattr(torch.amp, 'custom_fwd'): # type: ignore[attr-defined]
    deprecated = True
    from torch.amp import custom_fwd, custom_bwd # type: ignore[attr-defined]
else:
    deprecated = False
    from torch.cuda.amp import custom_fwd, custom_bwd

custom_fwd = custom_amp_decorator(custom_fwd, deprecated)
custom_bwd = custom_amp_decorator(custom_bwd, deprecated)