r"""
Parse PyTorch dtypes from strings.

Use :func:`parse_dtype` to parse str-valued dtypes.

Use :class:`DTypeParseError` catch attempts to parse unsupported dtypes.

See :const:`DTYPE_STRINGS` for a list of supported dtypes (which is dependent on
the installed PyTorch version).
"""


import logging


logger = logging.getLogger(__name__)


import torch
from packaging import version


class DTypeParseError(Exception):
    r"""
    Exception raised when parsing unsupported str-valued dtypes.
    """
    def __init__(
        self,
        custom_message: str = None,
    ):
        r"""
        Initialize :class:`DTypeParseError`\.

        Args:
            custom_message (str, optional): If not None, is appended to the
                primary "Unrecognized dtype" message.
                Defaults to None
        """
        message = "Unrecognized dtype"
        if custom_message is not None:
            message = f"{message}: {custom_message}"
        super().__init__(message)


def parse_dtype(dtype: str):
    r"""
    Parse str-valued dtypes into PyTorch dtypes.

    Args:
        dtype (str): The str-valued dtype to parse.

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
    parsed_dtype = _DTYPE_MAP.get(dtype)
    if parsed_dtype is None:
        raise DTypeParseError(dtype)
    return parsed_dtype


def match_version():
    r"""
    Update the list of supported dtypes depending on the installed PyTorch version.
    """
    torch_version = version.parse(torch.__version__)
    if torch_version < version.parse("1.3.0"):
        message = (
            f"Using PyTorch version '{torch_version}'. Note that torch.bool "
            "and torch.bfloat16 dtypes were added in version '1.3.0'. "
            "To use Boolean or BFloat16 dtypes, please upgrade PyTorch."
        )
        logger.warning(message)
    else:
        _DTYPE_MAP.update(_DTYPE_from_v1_3_0)


_DTYPE_MAP = {
    "int": int,
    "torch.uint8": torch.uint8,
    "torch.int8": torch.int8,
    "torch.int16": torch.int16,
    "torch.short": torch.int16,
    "torch.int32": torch.int32,
    "torch.int": torch.int32,
    "torch.int64": torch.int64,
    "torch.long": torch.int64,

    "float": float,
    "torch.float16": torch.float16,
    "binary16": torch.float16,
    "torch.half": torch.float16,
    "torch.float32": torch.float32,
    "torch.float": torch.float32,
    "torch.float64": torch.float64,
    "torch.double": torch.float64,

    "complex": complex,
    "torch.complex64": torch.complex64,
    "torch.cfloat": torch.complex64,
    "torch.complex128": torch.complex128,
    "torch.cdouble": torch.complex128,
}
_DTYPE_from_v1_3_0 = {
    "torch.bool": torch.bool,
    "torch.bfloat16": torch.bfloat16,
}


match_version()


DTYPE_STRINGS = list(_DTYPE_MAP.keys())
