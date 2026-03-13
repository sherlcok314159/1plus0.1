---
tags:
  - 求导
---

手动 autograd 是通过调 `autograd.Function` 接口来实现，有 forward 和 backward 两个静态方法，使用时有两个要点：

1. forward 中会通过 `ctx.save_for_backward` 来保留反传需要的张量，这也是有些张量「被原地修改」报错 RuntimeError: a leaf Variable that requires grad is being used in an「in-place」operation 的原因，因为反传时找不到张量原始的值
2. backward 的输入是上游对 apply 对象的梯度，求梯度时用 `ctx.saved_tensors` 读取需要的张量，返回的梯度和输入是「一一对应」的关系

## 全连接层

将 [[求导标量篇#全连接层]] 写成 autograd 的形式：

```python
class ManualNet(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, W1, W2, y):
        z1 = W1 @ x
        a1 = z1.sigmoid()
        z2 = W2 @ a1
        loss = 0.5 * (y - z2).pow(2).sum()
        ctx.save_for_backward(x, a1, W2, z2, y)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        x, a1, W2, z2, y = ctx.saved_tensors
        delta = z2 - y
        grad_W1 = ((W2.T @ delta) * (a1 * (1-a1))) @ x.T
        grad_W2 = delta @ a1.T
        # 返回的梯度必须和输入一一对应, x, y 的梯度都是 None
        return None, grad_W1 * grad_output, grad_W2 * grad_output, None
```

然后与官方的进行对比，这里是在 `loss_custom` 上进行 apply，所以 `grad_output` 其实等于 1

```python
x = torch.randn(3, 1)
y = torch.randn(2, 1)
W1 = torch.randn(4, 3, requires_grad=True)
W2 = torch.randn(2, 4, requires_grad=True)

W1_copy = W1.detach().clone().requires_grad_(True)
W2_copy = W2.detach().clone().requires_grad_(True)

loss_custom = ManualNet.apply(x, W1, W2, y)
loss_custom.backward()

# PyTorch
z1_ref = W1_copy @ x
a1_ref = z1_ref.sigmoid()
z2_ref = W2_copy @ a1_ref
loss_ref = 0.5 * (y - z2_ref).pow(2).sum()
loss_ref.backward()

torch.testing.assert_close(W1.grad, W1_copy.grad)
```

## MoE Aux Loss

```python
class MoEAuxLossGrad(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: Tensor, loss: Tensor, coeff: float):
        ctx.save_for_backward(loss)
        ctx.coeff = coeff
        return x

    @staticmethod
    def backward(ctx, dx):
        loss = ctx.saved_tensors[0]
        coeff = ctx.coeff

        if coeff is None or coeff == 0.0:
            return dx, None, None

        aux_loss = torch.full((1,),
                              fill_value=coeff,
                              dtype=loss.dtype,
                              device=torch.cuda.current_device())
        return dx, aux_loss, None
```

与上面的例子最后 apply 不同，这段代码在中间进行 apply，因为首先是从 final loss 一路 backward 回来，就不需要手动 backward。这段代码等价于：

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{orig}} +\lambda \mathcal{L}_{\text{aux}}, \, \frac{{\partial \mathcal{L}_{\text{total}}}}{\partial \mathcal{L}_{\text{aux}}} = \lambda\rule{0pt}{1.4em}
$$

接着从 `aux_loss` 出发，继续前传梯度

```python
class MoE(nn.Module):

    def forward(self, x):
        flat_x = rearrange(x, 'b m d -> (b m) d')
        router_logits: Tensor = self.router(flat_x.float())
        if self.activation_func == 'sigmoid':
            router_scores = F.sigmoid(router_logits)
        else:
            router_scores = F.softmax(router_logits, dim=-1)
        tokens_ratio = tokens_per_expert / tokens_per_expert.sum()
        if self.activation_func != 'softmax':
            router_scores = router_scores / router_scores.sum(dim=-1, keepdim=True)
        probs_ratio = router_scores.mean(dim=0)
        aux_loss = self.num_experts * (tokens_ratio * probs_ratio).sum()
        flat_x = MoEAuxLossGrad.apply(flat_x, aux_loss, self.aux_loss_coeff)
```
