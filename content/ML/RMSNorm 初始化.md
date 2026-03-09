---
tags:
  - Normalization
---

## 定义

传统的 RMSNorm 都会将 $\gamma$ 初始化为 $1$，而如果采用零初始化，则计算变为：

$$
\boldsymbol{x}_{\text{norm}} = (1 + \gamma) \cdot\frac{\boldsymbol{x}}{\sqrt{ \frac{1}{n} \sum_{i}^{n} x_{i}^{2}+\epsilon}}
$$

```python
class RMSNorm(nn.Module):

    def __init__(self, dim: int, zero_init: bool = False, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.zero_init = zero_init
        self.gamma = nn.Parameter(torch.zeros(dim) if zero_init else torch.ones(dim))

    def forward(self, x: Tensor):
        x_norm = self._norm(x.float())
        gamma = self.gamma + 1 if self.zero_init else self.gamma
        return (gamma * x_norm).type_as(x)

    def _norm(self, x: Tensor):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
```

## 影响

首先在 weight decay 的影响下：`p.mul_(1 - lr * weight_decay)`，会使得参数逐渐趋近于 $0$，规定 $w$ 指最终乘上的系数：

- $\gamma$ 初始化为 $1$ 时，最终 $w$ 的值会趋近 $0$
- $\gamma$ 初始化为 $0$ 时，最终 $w$ 的值会趋近 $1$

换言之，不同的初始化会改变最终乘上的系数。那么新问题是，什么情况下需要将 $\gamma$ 初始化为 $0$？

- 有更好的性能，实测会略差，当然也有人测出差不多的情况
- 控制系数的值域而不让其一直小到 $0$[^1]
- 进行低精度训练时，如果 $\gamma$ 初始化为 $1$ 且用 _bfloat16_ 训练，由于[[不同精度的表示#舍入误差]]会更新不动；抑或是在 FP8 下进行一些量化[^2]

[^1]: _Qwen3-Next: Open-Source Long-Context AI Model for Reasoning & Code_. (不详). 取读于 2025 年 11 月 13 日, 从 [https://qwen3next.org/](https://qwen3next.org/)
[^2]: https://ceramic.ai/blog/zerocentered
