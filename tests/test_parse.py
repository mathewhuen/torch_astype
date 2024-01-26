import torch
import pytest
import warnings
from torch_astype import (
    parse_dtype,
    DTypeParseError,
)


class TestParse():
    known_dtypes = {
        "torch.bool": torch.bool,

        "int": int,
        "torch.uint8": torch.uint8,
        "torch.int8": torch.int8,
        "torch.int16": torch.int16,
        "torch.short": torch.short,
        "torch.int32": torch.int32,
        "torch.int": torch.int,
        "torch.int64": torch.int64,
        "torch.long": torch.long,

        "float": float,
        "torch.float16": torch.float16,
        "binary16": torch.float16,
        "torch.half": torch.half,
        "torch.float32": torch.float32,
        "torch.float": torch.float,
        "torch.float64": torch.float64,
        "torch.double": torch.double,

        "torch.bfloat16": torch.bfloat16,

        "complex": complex,
        "torch.complex64": torch.complex64,
        "torch.cfloat": torch.cfloat,
        "torch.complex128": torch.complex128,
        "torch.cdouble": torch.cdouble,
    }

    @pytest.mark.parametrize(
        "string_dtype,target_dtype",
        known_dtypes.items(),
    )
    def test_parse_success(self, string_dtype, target_dtype):
        assert parse_dtype(string_dtype) == target_dtype

    def test_error(self):
        with pytest.raises(DTypeParseError):
            parse_dtype("np.int")
