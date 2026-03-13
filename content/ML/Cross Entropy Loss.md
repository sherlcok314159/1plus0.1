---
tags:
  - 指标
  - 信息论
---

## 基础知识

先举个例子，记扔骰子为 $X$，该事件可能的结果 $x=0, 1, \dots$

在信息学中用 0 和 1 的序列来编码信息，比如 00 代表天晴，01 代表下雨，编码长度就是两位比特，在传输中我们希望用「最短编码」来表达某个结果。设计最短编码的原则是：对于发生概率大的结果，给它更短的序列；而发生概率小的结果，给它长点的序列也没关系，因为用得不多，即

$$
L(x) = - \log_{2} P(x)
$$

天晴和下雨的概率都是 $0.5$，那么 $L(x) = 1$，用 0 代表天晴，1 代表下雨，一位比特就够了，同时后文默认 $\log$ 以 $2$ 为底

当事件有多种结果时，可以用[[大数定律和中心极限定理#大数定律]]来算出编码全部结果所需的平均编码长度：

$$
\frac{{\sum_{i} L(x_{i})}}{n} = \mathbb{E}[L(x)] = \sum_{i} P(x_{i}) (-\log P(x_{i}))
$$

上式即为「熵」的定义，由于用的是最短编码，熵也就是理论上「平均最少」需要多少位比特来编码某个事件的全部结果，记为 $H(P)$

$$
H(P) =  -\sum_{i} P(x_{i}) \log P(x_{i}) = \mathbb{E}_{x \sim P}[-\log P(x)]
$$

假如在扔骰子，骰子五个面是 0，一个面是 1, 那么熵为：

$$
H(P) = -\left( \frac{5}{6} \log \frac{5}{6} + \frac{1}{6}\log \frac{1}{6}\right) \approx 0.65
$$

但每次不是最少得有一位嘛，$0.65 < 1$，这里可以用「分块传输」的特点：假如一共扔了两次，一共有这几种可能，设计时需满足[[前缀码原则]]：

- 00：0
- 01：10
- 10：110
- 11：111

接着算一下平均编码长度：

$$
L = \frac{5}{6}\times \frac{5}{6} \times 1 + \frac{5}{6} \times \frac{1}{6} \times 2 + \frac{1}{6} \times \frac{5}{6} \times 3 + \frac{1}{6} \times \frac{1}{6} \times 3 = 1.47
$$

平均到每次即为 $0.74$

当熵越大，即需要更多的位数来进行编码，说明不确定性越大，系统越不稳定

## 交叉熵

交叉熵 $H(P, Q)$ 指的是当我用错误的分布 $Q$ 去编码时的平均编码长度

$$
H(P, Q) = \mathbb{E}_{x \sim P}[-\log Q(x)] \geq H(P)
$$

当我们在最小化交叉熵时，其实是让分布 $Q$ 和 $P$ 尽可能地接近，这样才是最短编码，KL 散度指用错误分布 $Q$ 编码时比用 $P$ 编码时多浪费的比特数

$$
D_{\text{KL}}(P\|Q) = H(P, Q) - H(P)
$$

不难看出，最小化交叉熵和 KL 散度目的是一样的：使得分布 $Q$ 和 $P$ 尽可能地接近，在单样本多分类任务中用「交叉熵损失」来让数据分布和模型分布尽可能地接近：

$$
H(\mathcal{D}, p_{\text{model}}) = \mathbb{E}_{\boldsymbol{x},y \sim\mathcal{ D}}[-\log p_{\text{model}}(y|\boldsymbol{x})]
$$

如果用「KL 散度」来优化会得到一样的结果，原因是 $H(P)=0$，每个样本的标签都是硬标签，即 $p(y_{i}|\boldsymbol{x}_{i}) = 1$

## 实践

规定 $\boldsymbol{x}$ 是 `logits`，所以先用 `softmax` 让其变成分布；另外每个样本是「等概率出现」的，所以求期望是直接 `mean()`

```python
import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss

def cross_entropy_loss(x: Tensor, y: Tensor):
    y = y.reshape(y.size(0), -1)
    probs = x.softmax(dim=-1)
    p_model = probs.take_along_dim(y, dim=1)
    return -p_model.log().mean()

x = torch.randn((3, 2))
y = torch.tensor([0, 1, 1])
loss_fct = CrossEntropyLoss()
torch.testing.assert_close(loss_fct(x, y), cross_entropy_loss(x, y))
```
