"""Device setup shared by sampling and chain-count benchmarks."""

import torch


def device_for_config(cfg):
    device = torch.device(cfg["compute"]["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("a CUDA device was requested but torch.cuda.is_available() is false")
    if cfg["compute"]["deterministic"]:
        torch.use_deterministic_algorithms(True)
    return device
