r"""
Add the :meth:`astype` method to PyTorch Tensors.
"""


import logging


logger = logging.getLogger(__name__)


import torch
from typing import Union
from torch_astype.parse import parse_dtype, DTYPE_STRINGS


def astype(tensor: torch.Tensor, dtype: Union[str, torch.dtype]):
    r"""
    Return a copy of tensor with the specified dtype.

    Args:
        tensor (torch.Tensor): The tensor to convert to the target dtype.
        dtype (str, torch.dtype): The target dtype.

    Note:
        For a list of dtypes supported by PyTorch, see their documentation
        at `https://pytorch.org/docs/stable/tensor_attributes.html`\.
        The full list of dtype strings that torch_astype supports is:
        'int', 'torch.uint8', 'torch.int8', 'torch.int16', 'torch.short',
        'torch.int32', 'torch.int', 'torch.int64', 'torch.long', 'float',
        'torch.float16', 'binary16', 'torch.half', 'torch.float32',
        'torch.float', 'torch.float64', 'torch.double', 'complex',
        'torch.complex64', 'torch.cfloat', 'torch.complex128',
        'torch.cdouble', 'torch.bool', 'torch.bfloat16'.
        This list can be accessed at :const:`torch_astype.DTYPE_STRINGS`\.
    """
    if isinstance(dtype, torch.dtype):
        return tensor.to(dtype=dtype)
    if isinstance(dtype, str):
        return tensor.to(dtype=parse_dtype(dtype))
    raise ValueError()


torch.Tensor.astype = astype
