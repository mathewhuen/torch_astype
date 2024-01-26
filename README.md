# Torch AsType


A hacky fix to make PyTorch tensors a little more like NumPy arrays by adding a
.astype method that accepts str-valued dtypes.

## Quick Start

Either import torch_astype and use the new torch.Tensor.astype right away:
```python
import torch
import torch_astype


tensor = torch.tensor([1,2,3.5])
print(tensor.astype('int'))
```

Or import the dtype parser directly for general string-to-dtype conversion:
```python
import json
from torch_astype.parse import parse_dtype


with open("config.json", "rb") as f:
    config = json.load(f)
test = torch.randn(2, 3, dtype=parse_dtype(config["dtype"]))
```
