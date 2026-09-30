import importlib
from functools import wraps
from typing import Any

from cosmos_rl.utils.logging import logger

if importlib.util.find_spec("xformers") is not None:
    # Fix xformers non-compatible, for environment with xformers installed
    # Silently make it unfindable when using importlib.util.find_spec
    # TODO (yy): may find a better solution
    _orig_find_spec = importlib.util.find_spec

    @wraps(_orig_find_spec)
    def blocked_find_spec(name, *args, **kwargs):
        if name == "xformers" or name.startswith("xformers."):
            return None
        return _orig_find_spec(name, *args, **kwargs)

    importlib.util.find_spec = blocked_find_spec
    logger.warning("xformers is not compatible with Cosmos-RL now, please uninstall it")

def _diffusion_pipeline():
    """Import optional diffusion dependencies only for diffusion workloads.

    Text/VLM policy training does not use diffusers.  Eagerly importing it can
    nevertheless pull in optional torchao kernels whose version requirements
    differ from the CUDA-compatible PyTorch installed for the policy worker.
    """

    from diffusers import DiffusionPipeline as pipeline

    return pipeline


def diffusers_config_fn(*args: Any, **kwargs: Any):
    return _diffusion_pipeline().load_config(*args, **kwargs)


def __getattr__(name: str):
    if name == "DiffusionPipeline":
        return _diffusion_pipeline()
    raise AttributeError(name)

__all__ = ["diffusers_config_fn", "DiffusionPipeline"]
