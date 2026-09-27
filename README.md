简介
--------------------------------------------------------------------------------

clnn 是一个基于 clvt（Common Lisp 张量库）的纯 Lisp 神经网络库。

它提供工业级的层组件、损失函数、优化器、学习率调度器与训练工具，
覆盖从 MLP、CNN、RNN 到 Transformer 的完整架构栈，可用于分类、回归、
序列建模与强化学习等主流深度学习任务。

设计目标：

  * 完整       —— 核心 API 覆盖主流深度学习任务
  * 可组合     —— 所有组件基于 CLOS 泛型协议，用户扩展无需修改库代码
  * 可验证     —— 核心层有数值梯度对拍测试，损失函数有解析 vs 有限差分验证
  * 可序列化   —— 模型一键保存与加载，参数逐位一致
  * 无外部依赖 —— 除 clvt 外无硬依赖，sb-simd 为可选加速


特性
--------------------------------------------------------------------------------

层（Layers）

  全连接      dense
  卷积        conv2d（im2col 实现，支持 stride / padding）
  池化        max-pool2d / avg-pool2d / global-avg-pool2d
  归一化      batch-norm（2D 与 ND 双路径）/ layer-norm
  正则化      dropout（含训练与推理模式切换）
  激活        activation-layer（ReLU / LeakyReLU / Sigmoid / Tanh / GELU /
              Swish / Mish / Softplus / HardTanh / HardSigmoid /
              Softmax / LogSoftmax）
  序列        rnn-cell / rnn-sequence / lstm / gru
  嵌入        embedding（含 max-norm 与梯度频率缩放）
  注意力      scaled-dot-product-attention / multi-head-attention
  结构        transformer-block / residual / flatten
  容器        sequential


损失函数

  mse-loss / bce-loss / ce-loss（含 label smoothing）/ huber-loss /
  kl-divergence-loss / cosine-similarity-loss / smooth-l1-loss

  所有损失均支持 mean / sum / none 三种归约方式。


优化器

  sgd（含 momentum / nesterov / weight-decay）
  adam（含 amsgrad）
  adamw
  rmsprop（含 centered / momentum）
  adagrad（含 lr-decay）

  所有优化器均支持全局梯度范数裁剪。


学习率调度器

  step-lr
  exponential-lr
  cosine-annealing-lr
  reduce-on-plateau
  warmup-cosine-lr
  one-cycle-lr


初始化器

  he-normal（默认）
  he-uniform
  xavier-normal / xavier-uniform
  orthogonal-init
  kaiming-normal
  zeros-init / ones-init / constant-init
  truncated-normal-init


正则化器

  l1-regularizer / l2-regularizer / elastic-regularizer


工具

  序列化      save-model / load-model（plist 格式，可读，可版本控制）
  深拷贝      copy-network（基于 CLOS 泛型分发，自动递归容器）
  模型统计    param-count / flops-estimate
  梯度工具    compute-grad-norm / clipped-gradient-update!
  模式切换    with-training / set-model-training! / reset-training!


核心概念
--------------------------------------------------------------------------------

层的协议

  每个层通过一组泛型函数协作：前向、反向、参数枚举、梯度枚举、
  梯度槽定位、缓存槽定位。用户自定义层只需实现这几个方法即可
  被库完整接纳，无需修改任何库代码。


延迟初始化

  部分层在构造时不知道输入维度，而是在第一次前向时推断并创建权重。
  因此在统计参数量或访问参数之前，需要先用虚拟输入触发一次前向。


训练与推理模式

  库通过全局开关控制默认行为，并允许每个层单独覆盖。
  Dropout 与 BatchNorm 在两种模式下行为不同：
  训练模式启用随机失活与批统计量，推理模式启用恒等映射与滑动统计量。


标准训练循环

  一次完整训练步的标准顺序为：
  清零梯度 → 前向 → 计算损失 → 计算损失梯度 → 反向传播 →
  优化器更新参数 → 释放前向缓存。

  该顺序不可打乱：清零必须在反向之前，缓存释放必须在反向之后。


示例
--------------------------------------------------------------------------------

库附带多个端到端示例，覆盖不同架构与任务类型：

  MNIST 手写数字分类            Dense 网络与小型 CNN
  ResNet-9                     残差容器 + BatchNorm + 全局池化 + SGD + 余弦调度
  Transformer 序列分类          嵌入 + 位置编码 + Transformer Block + 分类头
  RNN / LSTM / GRU 复制任务     循环网络的多种变体与序列包装层
  强化学习（CartPole）          REINFORCE / A2C / DQN / PPO
  强化学习（MountainCar）       DQN（稀疏奖励场景）
  强化学习（LunarLander）       DQN（连续状态、多动作场景）

每个示例都是一个独立可运行的脚本，同时充当相应架构的最小参考实现。


测试
--------------------------------------------------------------------------------

库附带完整的回归测试套件，覆盖：

  * 历史 bug 的回归测试（含 stop-gradient、BatchNorm 边界、LSTM 初始状态
    梯度、KL 散度梯度、参数量估算、注意力偏置统计等）
  * 数值梯度对拍测试（SDPA、MHA、LSTM、GRU、Embedding、Conv2d、各类池化）
  * 优化器逐步数值验证（SGD / Momentum / Nesterov / Adagrad / RMSprop / AdamW）
  * 调度器精确值序列验证
  * 序列化往返测试与集成测试（loss 下降、准确率、保存加载一致性）

此外还包含：

  * 架构级压力测试（多分支、残差、注意力路由、深层网络等）
  * 端到端训练用例（多种损失函数与架构组合下的收敛验证）


性能与资源
--------------------------------------------------------------------------------

已做的优化

  * 二维矩阵乘法使用 SIMD 加速内核
  * 高维（批量）矩阵乘法使用 SIMD 与多线程加速
  * 卷积的 im2col / col2im 采用预计算偏移与去泛型分发

已知瓶颈

  卷积是纯 Lisp 实现的物理瓶颈。im2col 的操作量正比于
  批大小 × 空间尺寸 × 输入通道 × 卷积核面积，无法通过语言层面
  的优化绕开。同样的算法在有 BLAS 支持的环境下会快一个数量级，
  但在纯 Lisp 中这是语言本身的极限。

进一步的加速途径

  * 减少数据量或缩减模型规模
  * 通过外部函数接口调用 BLAS 库（尚未实现）
  * 使用 GPU 后端（尚未实现）

内存优化准则

  CNN 的内存占用大致正比于：
  批大小 × 空间尺寸 × 输入通道 × 卷积核面积 × 层数

  优化优先级：
    1. 尽早降低空间分辨率（影响是平方级的）
    2. 缩小批大小（影响是线性的）
    3. 减少通道数（影响是线性的）


设计原则
--------------------------------------------------------------------------------

  1. 泛型分发优先于类型分支
     所有容器与协议基于 CLOS 泛型函数，用户扩展不需要修改库代码。

  2. 显式断言优先于静默回退
     关键路径上的形状与类型检查使用断言，不做"看起来能跑"的静默处理。

  3. 默认安全，可选优化
     SIMD、多线程、低安全检查等均由开关控制，默认关闭。

  4. 序列化友好
     所有层都支持转为可读的 plist，模型可 diff、可版本控制。

  5. 测试即文档
     测试套件本身就是最完整的 API 使用示例。


已知限制
--------------------------------------------------------------------------------

  * 无 GPU 支持，纯 CPU 运行
  * 无自动微分，所有层的反向传播手写
  * 卷积在纯 Lisp 下有性能瓶颈
  * 不支持动态图，形状在前向时确定
  * 批量矩阵乘法仅浮点类型走 SIMD，整数类型回退到通用路径
  * 循环网络无专用加速，长序列性能受限


项目结构
--------------------------------------------------------------------------------

  包定义与导出
  协议与基类
  初始化器
  Dense / Activation / Flatten / Residual
  Conv2d / 各类池化 / im2col
  Dropout / BatchNorm / LayerNorm
  Embedding
  注意力与 Transformer
  RNN / LSTM / GRU / 序列包装
  损失函数
  优化器
  学习率调度器
  Sequential / 序列化 / 深拷贝
  示例目录
  测试目录
  探索性代码目录


常见问题
--------------------------------------------------------------------------------

  问：为什么参数量统计返回零？
  答：层采用延迟初始化，需要先执行一次前向传播。

  问：训练损失不下降怎么办？
  答：依次检查——
        梯度是否在反向之前清零？
        模型是否已切换到训练模式？
        参数量是否非零（延迟初始化是否触发）？
        学习率是否合理？

  问：内存爆炸怎么办？
  答：按"早期下采样 → 缩小批大小 → 缩小模型"的顺序调整。

  问：训练很慢怎么办？
  答：卷积是纯 Lisp 的物理瓶颈。减少数据、缩小模型、减少训练轮次
      是最直接的方案。

  问：为什么不做自动微分？
  答：库采用静态图风格，每个层的反向传播手动推导并经过数值验证。
      这牺牲了灵活性，但换来了可预测的性能与可读的代码。


状态
--------------------------------------------------------------------------------

  可用。

  核心层、损失函数、优化器、调度器、序列化均已在端到端任务中验证。
  速度上存在纯 Lisp 的物理瓶颈，性能敏感的场景建议配合 BLAS 后端使用。

================================================================================
