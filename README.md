# Torch AsType


A hacky fix to make PyTorch tensors a little more like NumPy arrays by adding
an .astype method that accepts str-valued dtypes.

## Installation

```shell
git clone https://github.com/mathewhuen/torch_astype.git
cd torch_astype
pip install .
```

## Quick Start

Either import torch_astype and use the torch.Tensor.astype method right away:
```python
import torch
import torch_astype


tensor = torch.tensor([1,2,3.5])
print(tensor.astype('int'))
```

Or import the dtype parser for general string-to-dtype conversion:
```python
import json
from torch_astype.parse import parse_dtype


with open("config.json", "rb") as f:
    config = json.load(f)
test = torch.randn(2, 3, dtype=parse_dtype(config["dtype"]))
```

Unrecognized dtypes will raise a DTypeParseError. Use this to catch conversions
that we do not yet support:
```python
from torch_astype import parse_dtype, DTypeParseError


dtype = "fancyqbit"
try:
    torch_dtype = parse_dtype(dtype)
except DTypeParseError:
    print(f"torch_dtype does not yet support the specified type: {dtype}")
```

If there is a missing type that should be included, please submit an issue!
