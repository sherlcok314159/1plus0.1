---
tags:
  - 初始化
---

## 前传和反传

考虑参数矩阵 $\boldsymbol{W} \in \mathbb{R}^{m \times n}$ 和输入 $\boldsymbol{x} \in \mathbb{R}^{n}$，两者相互独立，同时期望均为 0，$\boldsymbol{y} = \boldsymbol{Wx}$

$$
\begin{align*}
\boldsymbol{y}_{i} &= \sum_{j}^{n} \boldsymbol{W}_{ij}\boldsymbol{x}_{j}\\
\mathbb{E}[\boldsymbol{y}_{i}] &= \sum_{j}^{n}\mathbb{E}[\boldsymbol{W}_{ij}\boldsymbol{x}_{j}] = \sum_{j}^{n} \mathbb{E}[\boldsymbol{W}_{ij}]\mathbb{E}[\boldsymbol{x}_{j}] = 0\\
\mathbb{V}[\boldsymbol{y}_{i}] &= \sum_{j}^{n} \mathbb{V}[\boldsymbol{W}_{ij}\boldsymbol{x}_{j}] = \sum_{j}^{n} \mathbb{E}[\boldsymbol{W}_{ij}^{2}\boldsymbol{x}_{j}^{2}] - \mathbb{E}[\boldsymbol{W}_{ij}\boldsymbol{x}_{j}]^{2}\\
&= \sum_{j}^{n} \mathbb{E}[\boldsymbol{W}_{ij}^{2}]\mathbb{E}[\boldsymbol{x}_{j}^{2}]
= \sum_{j}^{n} (\mathbb{E}[\boldsymbol{W}_{ij}^{2}] - \mathbb{E}[\boldsymbol{W}_{ij}]^{2})(\mathbb{E}[x_{j}^{2}] - \mathbb{E}[\boldsymbol{x}_{j}]^{2})\\
&= \sum_{j}^{n} \mathbb{V}[\boldsymbol{W}_{ij}]\mathbb{V}[\boldsymbol{x}_{j}] = n \mathbb{V}[\boldsymbol{W}_{ij}]\mathbb{V}[\boldsymbol{x}_{j}]
\end{align*}
$$

如果要控制 activation $\boldsymbol{y}_{i}$ 的方差不随着 $n$ 发生变动，令 $\mathbb{V}[\boldsymbol{W}_{ij}] = 1/n$

接着考虑反向传播，因为要看 $\boldsymbol{W}$ 的方差，所以就看 $\nabla_{\boldsymbol{x}}\mathcal{L}$

$$
\mathrm{d} \mathcal{L} = \mathrm{Tr}((\nabla_{\boldsymbol{y}} \mathcal{L})^{\top}\mathrm{d}\boldsymbol{y})= \mathrm{Tr}((\nabla_{\boldsymbol{y}}\mathcal{L})^{\top}\boldsymbol{W}\mathrm{d}\boldsymbol{x})
$$

得：

$$
\begin{align*}
\nabla_{\boldsymbol{x}}\mathcal{L} &= \boldsymbol{W^{\top}}\nabla_{\boldsymbol{y}}\mathcal{L}\\
(\nabla_{\boldsymbol{x}}\mathcal{L})_{i} &= \sum_{j} \boldsymbol{W}_{ji}(\nabla_{\boldsymbol{y}}\mathcal{L})_{j}\\
\mathbb{E}[(\nabla_{\boldsymbol{x}}\mathcal{L})_{i}] &= \sum_{j}^{m} \mathbb{E}[\boldsymbol{W}_{ji}(\nabla_{\boldsymbol{y}}\mathcal{L})_{j}] = \sum_{j} \mathbb{E}[\boldsymbol{W}_{ji}] \mathbb{E}[(\nabla_{\boldsymbol{y}}\mathcal{L})_{j}] = 0\\
\mathbb{V}[(\nabla_{\boldsymbol{x}}\mathcal{L})_{i}] &= \sum_{j}^{m} \mathbb{V}[\boldsymbol{W}_{ji}(\nabla_{\boldsymbol{y}}\mathcal{L})_{j}] = \sum_{j}^{m} \mathbb{E}[\boldsymbol{W}_{ji}^{2}(\nabla_{\boldsymbol{y}}\mathcal{L})_{j}^{2}] - \mathbb{E}[\boldsymbol{W}_{ji}(\nabla_{\boldsymbol{y}}\mathcal{L})_{j}]^{2}\\
&= m \mathbb{V}[\boldsymbol{W}_{ji}] \mathbb{V}[(\nabla_{\boldsymbol{y}}\mathcal{L})_{j}]
\end{align*}
$$

如果想控制梯度反传 $\nabla_{\boldsymbol{x}}\mathcal{L}$ 的方差不随着 $m$ 发生变动，令 $\mathbb{V}[\boldsymbol{W}_{ij}] = 1/m$

## 调和两者

那么如果 $m\neq n$，则使用两者的 [[各种平均方式#调和平均]]，即：

$$
\frac{2}{\frac{1}{\frac{1}{m}} + \frac{1}{\frac{1}{n}}} = \frac{2}{m+n}
\rule{0pt}{1.4em}
$$

这就是 [Xavier Normal](https://docs.pytorch.org/docs/stable/nn.init.html#torch.nn.init.xavier_normal_) 的方差值[^1]，也被成为 Glorot 初始化，如果是 uniform 方式，需要做下转换：

$\mathcal{U}(-a, a)$ 的方差是 $a^{2}/3$，那么：

$$
\frac{a^{2}}{3} =\frac{2}{m+n} \implies a = \sqrt{ \frac{6}{m+n} }
$$

[^1]: Glorot, X. &amp; Bengio, Y.. (2010). Understanding the difficulty of training deep feedforward neural networks. <i>Proceedings of the Thirteenth International Conference on Artificial Intelligence and Statistics</i>, in <i>Proceedings of Machine Learning Research</i> 9:249-256 Available from https://proceedings.mlr.press/v9/glorot10a.html.
