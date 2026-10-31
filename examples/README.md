# examples

每个文件都是独立可加载的脚本：`(load "examples/xxx.lisp")` 只定义函数，**不会自动
开始训练**（唯一例外已经去掉 —— 早期两个文件末尾有裸调用，会把「加载」变成「跑一遍」）。

共性约定：

- 都需要 `(ql:quickload :clnn)`；MNIST / CIFAR 系列的还需要 `chipz` 解压。
- 训练入口都接受 keyword 参数，默认值偏小以便在 CPU 上跑通；要正式结果自己放大。
- 训练循环一律是 §3.1 的七步形态，参数更新一律 `(update! model opt)`。
- 时间都是在 8 核 CPU、SBCL 2.6.10、`--dynamic-space-size 4096` 下的量级参考。

## 清单

| 文件 | 入口 | 任务 | 依赖 | 参考耗时 |
|---|---|---|---|---|
| `mnist.lisp` | `(ensure-mnist-data)` → `(train-dense-mnist :epochs 5)` | MNIST 全连接基线（两个：plain / BN+dropout）；同时提供全局数据加载器，供其它文件复用 | chipz + 数据文件 | 5–10 min |
| `resnet9-mnist-lite.lisp` | `(train-resnet9-tiny)` | ResNet-9 mini on MNIST，峰值内存 < 300 MB，专为小机器调的通道数与 batch | chipz + 数据文件 | ~13 min |
| `resnet9-cifar.lisp` | `(run-resnet9-cifar)` / `(train-resnet9-cifar)` | ResNet-9 + `residual` 容器 + SGD/Nesterov/cosine-annealing。`run-` 是小规模快速跑通，`train-` 是默认 10000/2000 | CIFAR-10 binary | 5–15 min |
| `demo-last-bit.lisp` | `(demo-last-bit)` | 自定义层教学：`TakeLastTimeStep`（Embedding → LSTM → 取最后时间步 → Dense），判断 token 是否出现在序列里 | 无 | 1–3 min |
| `transformer-sum-classifier.lisp` | `(train-sum-classifier)` / `(run-sum-example)` | 序列求和是否超阈值 —— 必须跨位置聚合，是自注意力最直接的验证任务 | 无 | 2–5 min |
| `prove.lisp` | `(test-xor)` / `(test-spiral)` / `(test-gradient-dense)` … | 10 个逐层小实验，肉眼确认每块能跑。**不是回归套件** —— 严格断言在 `tests/` | 无 | 1–10 min |
| `dac-mnist.lisp` | `(dac-mnist-simple)` / `(dac-mnist-final :seed 10)` | 深度自适应聚类（AE + KMeans / DAC 微调），MNIST 上 ~60% ACC | chipz + 数据文件 | 45 min |
| `binocular-mnist.lisp` | `(train-binocular-mnist)` | 仿生双眼（周边视觉 + 中央凹采样 + 视差）替代均匀滑窗 | chipz + 数据文件 | ~60 min |
| `cartpole-reinforce.lisp` | `(train-reinforce :episodes 500)` / `(train-a2c)` / `(train-dqn)` / `(train-ppo-v3)` | 强化学习合集：REINFORCE / A2C / DQN ×3 / PPO，共 7 个示例。环境是自己实现的，不依赖 gym | 无 | 5–60 min |
| `conv2d-batched.lisp` | `(bench-conv2d-methods …)` / `(bench-conv2d-suite)` | **实验脚本**，不是示例：对比 `im2col + 2D matmul` 与批量 matmul 的卷积实现。不修改库本身 | 无 | 1 min |

## 想快速验证安装

```lisp
(ql:quickload :clnn)
(load "examples/prove.lisp")
(test-xor)          ; 几秒
(test-gradient-dense)   ; 有限差分对拍
```

或者跑完整套件（18 个文件，约 80 秒）：

```sh
cd tests && ./run-tests.sh
```

## 写新示例时

- 训练循环照抄 `mnist.lisp` 的七步形态，别自己发明顺序（§3.1 的顺序是契约）。
- 参数更新用 `(update! model opt)`。需要「多个子网络算一个优化步」时用
  `(with-optimizer-step (opt) …)`，不要连着调两次 `update!` 再指望它是一步。
- 加载已有模型后默认是推理态；继续训练要 `(train! model)`。
