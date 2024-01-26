import logging


logger = logging.getLogger(__name__)


from torch_astype import parse_dtype


def _astype(tensor, dtype):
    if isinstance(dtype, torch.dtype):
        return tensor.to(dtype=dtype)
    if isinstance(dtype, str):
        return tensor.to(dtype=parse_dtype(dtype))
    raise ValueError()


try:
    import torch
    torch.Tensor.astype = _astype
except:
    message = (
        "Attempted to add the astype method to torch.Tensor, but `import "
        "torch` failed. Running without astype."
    )
    logger.info(message)
