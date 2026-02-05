---
title: Example Title
draft: false
tags:
  - example-tag
---


> [!Quote]
> Stay Hungry. Stay Foolish.

测试

```python
import torch
import torch.linalg as LA
from utils import set_seed

set_seed(42)

dim1, dim2 = 1792, 1792
dtype = torch.float32
tensors = [torch.randn((dim1, dim2), dtype=dtype).cuda() for _ in range(3)]


def compute_l2_norm_for_tensors(tensors):
    individual_norms = [LA.vector_norm(t.detach()) for t in tensors]
    return LA.vector_norm(torch.stack(individual_norms))


orig_weight_norm = compute_l2_norm_for_tensors(tensors)
print('orig weight norm: ', orig_weight_norm)


def fro_normalize(x):
    return x / (LA.norm(x, ord='fro', dim=(-2, -1)) + 1e-7)


max_violate = 0
for i in range(1000):
    for p in tensors:
        p.data = LA.norm(p) * fro_normalize(p + 8e-4)
    updated_weight_norm = compute_l2_norm_for_tensors(tensors)
    max_violate = max(max_violate, abs(orig_weight_norm - updated_weight_norm))

print(dtype, ', max_violate: ', max_violate.item())
```

先算 $\boldsymbol{y}_{i}$ 的期望：

$$
\mathbb{E}\left[\sum_{j=1}^{d_{in}}\boldsymbol{W}_{ij}\boldsymbol{x}_{j}\right] = \sum_{j=1}^{d_{in}} \mathbb{E}[\boldsymbol{W}_{ij}\boldsymbol{x}_{j}] = \sum_{j=1}^{d_{in}} \mathbb{E}[\boldsymbol{W}_{ij}] \, \mathbb{E}[\boldsymbol{x}_{j}] = 0
$$

$\boldsymbol{y}_{i}$ 的方差为：

$$
\begin{align}
\mathbb{V}\left[\sum_{j=1}^{d_{in}}\boldsymbol{W}_{ij}\boldsymbol{x}_{j}\right]  & = \sum_{j=1}^{d_{in}} \mathbb{V}[\boldsymbol{W}_{ij}\boldsymbol{x}_{j}] = \sum_{j=1}^{d_{in}} \mathbb{E}[\boldsymbol{W}_{ij}^{2}\boldsymbol{x}_{j}^{2}] - \mathbb{E}[\boldsymbol{W}_{ij}\boldsymbol{x}_{j}]^{2} \\
 & = \sum_{j=1}^{d_{in}} \mathbb{E}[\boldsymbol{W}_{ij}^{2}] \, \mathbb{E}[\boldsymbol{x}_{j}^{2}] - \mathbb{E}[\boldsymbol{W}_{ij}]^{2}\,\mathbb{E}[\boldsymbol{x}_{j}]^{2} \\
 & = \sum_{j=1}^{d_{in}} \mathbb{E}[(\boldsymbol{W}_{ij}-\mu_{\boldsymbol{W}})^{2}] \, \mathbb{E}[(\boldsymbol{x}_{j}-\mu_{\boldsymbol{x}})^{2}]  \\
 & = {d_{in}} \, \sigma_{\boldsymbol{W}}^{2} \, \sigma_{\boldsymbol{x}}^{2}
\end{align}
$$
