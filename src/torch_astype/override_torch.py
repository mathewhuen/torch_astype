import logging


logger = logging.getLogger(__name__)


import torch
from torch_astype.parse import parse_dtype


def astype(tensor, dtype):
    if isinstance(dtype, torch.dtype):
        return tensor.to(dtype=dtype)
    if isinstance(dtype, str):
        return tensor.to(dtype=parse_dtype(dtype))
    raise ValueError()


torch.Tensor.astype = astype
