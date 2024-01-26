import torch
import pytest
import warnings
import torch_astype


class TestAstype():

    tensor_data = (
        (torch.tensor([1.1]), "torch.bool", torch.tensor([True], dtype=torch.bool)),

        (torch.tensor([1.1]), "torch.uint8", torch.tensor([1], dtype=torch.uint8)),
        (torch.tensor([1.1]), "torch.int8", torch.tensor([1], dtype=torch.int8)),
        (torch.tensor([1.1]), "torch.int16", torch.tensor([1], dtype=torch.int16)),
        (torch.tensor([1.1]), "torch.short", torch.tensor([1], dtype=torch.short)),
        (torch.tensor([1.1]), "torch.int32", torch.tensor([1], dtype=torch.int32)),
        (torch.tensor([1.1]), "torch.int", torch.tensor([1], dtype=torch.int)),
        (torch.tensor([1.1]), "torch.int64", torch.tensor([1], dtype=torch.int64)),
        (torch.tensor([1.1]), "torch.long", torch.tensor([1], dtype=torch.long)),

        (torch.tensor([1.1]), "torch.float16", torch.tensor([1.1], dtype=torch.float16)),
        (torch.tensor([1.1]), "binary16", torch.tensor([1.1], dtype=torch.float16)),
        (torch.tensor([1.1]), "torch.half", torch.tensor([1.1], dtype=torch.half)),
        (torch.tensor([1.1]), "torch.float32", torch.tensor([1.1], dtype=torch.float32)),
        (torch.tensor([1.1]), "torch.float", torch.tensor([1.1], dtype=torch.float)),
        (torch.tensor([1.1]), "torch.float64", torch.tensor([1.1], dtype=torch.float64)),
        (torch.tensor([1.1]), "torch.double", torch.tensor([1.1], dtype=torch.double)),

        (torch.tensor([1.1]), "torch.bfloat16", torch.tensor([1.101], dtype=torch.bfloat16)),

        (torch.tensor([1.1]), "torch.complex64", torch.tensor([1.1+0.j], dtype=torch.complex64)),
        (torch.tensor([1.1]), "torch.cfloat", torch.tensor([1.1+0.j], dtype=torch.cfloat)),
        (torch.tensor([1.1]), "torch.complex128", torch.tensor([1.1+0.j], dtype=torch.complex128)),
        (torch.tensor([1.1]), "torch.cdouble", torch.tensor([1.1+0.j], dtype=torch.cdouble)),
    )
    @pytest.mark.parametrize(
        "source,dtype,target",
        tensor_data,
    )
    def test_conversion(self, source, dtype, target):
        tensor = source.astype(dtype)
        assert torch.isclose(tensor, target)
        assert tensor.dtype == target.dtype

    @pytest.mark.parametrize(
        "source,dtype,target",
        tensor_data,
    )
    def test_conversion_cuda(self, source, dtype, target):
        if torch.cuda.is_available():
            source = source.to("cuda")
            target = target.to("cuda")
            tensor = source.astype(dtype)
            assert torch.isclose(tensor.data, target.data)
            assert tensor.dtype == target.dtype
            assert tensor.device == target.device
        else:
            warnings.warn("Cuda is not available. All cuda tests will be skipped")

    def test_error(self):
        with pytest.raises(torch_astype.DTypeParseError):
            torch.tensor([1]).astype("np.int")
