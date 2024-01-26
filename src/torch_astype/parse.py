import torch


_DTYPE_MAP = {
    "torch.bool": torch.bool,

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

    "torch.bfloat16": torch.bfloat16,

    "complex": complex,
    "torch.complex64": torch.complex64,
    "torch.cfloat": torch.complex64,
    "torch.complex128": torch.complex128,
    "torch.cdouble": torch.complex128,
}


class DTypeParseError(Exception):
    def __init__(
        self,
        custom_message: str = None,
    ):
        message = "Unrecognized dtype"
        if custom_message is not None:
            message = f"{message}: {custom_message}"
        super().__init__(message)


def parse_dtype(dtype: str):
    parsed_dtype = _DTYPE_MAP.get(dtype)
    if parsed_dtype is None:
        raise DTypeParseError(dtype)
    return parsed_dtype
