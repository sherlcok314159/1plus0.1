---
tags:
  - 训练动态
  - lm-head
---

## 问题

由 Cross Entropy Loss 的[[Cross Entropy Loss#实践]]可知：

$$
\begin{align*}
\boldsymbol{z} &= \boldsymbol{Wx}\\
\boldsymbol{a} &= \text{softmax}(\boldsymbol{z})\\
\mathcal{L} &= -\sum_{i} y_{i}\log {a}_{i}
\end{align*}
$$

$$
\frac{{\partial \mathcal{L}}}{\partial \boldsymbol{x}} = \boldsymbol{W^{\top}}(\boldsymbol{a} - \boldsymbol{y})\rule{0pt}{1.4em}
$$

联系[[矩阵的基本性质#零空间]]可知，由于 $\boldsymbol{W^{\top}} \in \mathbb{R}^{d \times V}, \boldsymbol{g}=\boldsymbol{a} - \boldsymbol{y} \in \mathbb{R}^{V}$，会有至少 $V - d$ 个方向被零空间 $\mathrm{ker}(\boldsymbol{W}^{\top})$ 湮灭，而

$$
\mathrm{ker}(\boldsymbol{W^{\top}})^{\perp} = \text{col}(\boldsymbol{W})
$$

将 $\boldsymbol{g}$ 拆分成两个分量：

$$
\boldsymbol{g} = \boldsymbol{g}_{\text{col}(\boldsymbol{W})} + \boldsymbol{g}_{\mathrm{ker}(\boldsymbol{W^{\top}})}
$$

联系投影的[[投影#实践]]，$\boldsymbol{g}_{\text{col}(\boldsymbol{W})}$ 可以通过投影来得出，

$$
r = \frac{\|P_{\text{col}(\boldsymbol{W})}\boldsymbol{g}\|_{\text{F}}}{\|\boldsymbol{g}\|_{\text{F}}} = \frac{\|\boldsymbol{W}(\boldsymbol{W^{\top}W})^{-1}\boldsymbol{W^{\top}}\boldsymbol{g}\|_{\text{F}}}{\|\boldsymbol{g}\|_{\text{F}}}
$$

然后可以统计 $1-r$ 来看梯度被湮灭的比例，$1-r$ 越高，被湮灭的越多

同时考虑到给 $\boldsymbol{W^{\top}W}$ 求逆，需要保证其特征值均不为 0，这里利用「特征值平移定理」，来规避：

```python
WtW.diagonal().add_(1e-6 * WtW.diagonal().mean())
```

实验设置如下：

```yaml
 dim: [1792, 2048, 3072]
 vocab_size: 50257
 num_layers: 16
 num_heads: 16
 lr: 8e-4
```

<div align="center">

![[W&B Chart 2026_3_23 14_46_31.png|600]]

</div>

可以发现：

- 整体梯度回传「湮灭比例」较高，最低的也有 70+
- 增大 $d$ 可以减少湮灭的比例

这说明目前「lm-head 还有很大潜力未被激发出来」，community 的目光都集中在 hidden layers

## 初探湮灭

解决湮灭问题有两种思路，一种是让 $\boldsymbol{W}$ 更接近满秩，这样 $D$ 维空间更满；另一种是换个架构去设计

对于第一种思路：

- 可以使用 Muon 来更新 $\boldsymbol{W}$，但由于词频不同，整体做正交化已知会掉点，故作为观测手段
- 将 $\mathcal{L}_{\text{aux}} = \lambda \|\boldsymbol{W^{\top}W} - \boldsymbol{I}\|_{2}$ 加入 loss
