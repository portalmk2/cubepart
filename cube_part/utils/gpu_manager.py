"""Lightweight GPU memory manager for phased model loading.

Provides a context-manager helper that loads a sub-module to GPU before a
computation block and unloads it back to CPU afterwards, keeping peak VRAM
low when multiple large models share a single GPU.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Generator, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@contextmanager
def on_device(
    module: nn.Module,
    device: torch.device,
) -> Generator[nn.Module, None, None]:
    """Context manager: move ``module`` to ``device``, yield, then move back
    to its original device and free GPU memory.

    Usage::

        with on_device(self.system.shape_model, self.device) as m:
            result = m.encode(surface)
        # shape_model is back on its original device, GPU memory freed.
    """
    prior_device = _guess_device(module)
    if prior_device == device:
        yield module
        return

    module = module.to(device)

    try:
        yield module
    finally:
        module = module.to(prior_device)
        torch.cuda.empty_cache()


def _guess_device(module: nn.Module) -> torch.device:
    """Return the first parameter's device as a reasonable guess for where the
    module currently lives.  Falls back to CPU when the module has no
    parameters (e.g. a scheduler).
    """
    for p in module.parameters():
        return p.device
    for b in module.buffers():
        return b.device
    return torch.device("cpu")