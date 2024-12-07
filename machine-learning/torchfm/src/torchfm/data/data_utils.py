import torch
from torch import Tensor

def encode(input: list[str], features: dict[str, int]) -> Tensor:
    if len(input) > len(features):
        raise ValueError("Length of `input` must be less than `features`.")
    output_dim = len(features)
    encoded = torch.zeros((output_dim), dtype=torch.int8)
    for t in input:
        encoded[features[t]] = 1
    return encoded