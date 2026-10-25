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

`examples/` 下的每个文件都是一个独立可运行的脚本，同时充当相应架构的
最小参考实现。加载脚本即开始训练（部分脚本耗时较长，见下表）：

  文件                              内容                                        耗时
  ────────────────────────────────  ──────────────────────────────────────────  ──────
  examples/mnist.lisp               MNIST 数据加载器 + 两个全连接基线          数分钟
  examples/resnet9-mnist-lite.lisp  ResNet-9 mini on MNIST（内存 < 300 MB）    数分钟
  examples/resnet9-cifar.lisp       ResNet-9 mini on CIFAR-10                  长
  examples/transformer-sum-classifier.lisp  Transformer 序列分类（判断序列和）  长
  examples/cartpole-reinforce.lisp  强化学习合集：REINFORCE / A2C / DQN / PPO   仅定义
  examples/prove.lisp               逐层正确性演示（例 1 ~ 例 10）              数秒
  examples/demo-last-bit-final.lisp LSTM 记忆最后一位                         数秒
  examples/conv2d-batched.lisp      批量 matmul 卷积的实现与基准对比            数秒
  examples/dac-mnist.lisp           DAC 深度自适应聚类                          长
  examples/binocular-mnist.lisp     仿生双眼（周边视觉 + 中央凹 + 视差）MNIST    长

注：`examples/cartpole-reinforce.lisp` 只做定义、不自动训练，需要手动
调用 `(train-reinforce …)` / `(train-dqn)` / `(train-ppo)` 等入口
（文件头部列出了全部 9 个入口）。

部分示例需要 `chipz`（解压 `.gz`）与本地 MNIST / CIFAR 数据。


测试
--------------------------------------------------------------------------------

`tests/` 下只放测试，全部带有断言，可以用一个脚本跑完：

    cd tests && ./run-tests.sh            # 全部（约 1.5 分钟）
    cd tests && ./run-tests.sh --fast     # 只跑快速套件（约 20 秒）
    cd tests && ./run-tests.sh test-architectures pdf-bugs-verify   # 指定文件

所有测试文件在存在失败项时都会以非 0 退出，因此可以直接接到 CI。

  test-architecture-fixes.lisp  架构审查缺陷的逐条回归（43 项）
  clnn-test-suite.lisp          主回归套件（26 项）
  test-regression-fixes.lisp    带缺陷编号的回归（26 项）
  test-copy-network.lisp        深拷贝保真（21 项）
  gradcheck-all-layers.lisp     全层/全损失 解析梯度 vs 中心差分
  test-architectures.lisp       架构级集成测试（CNN/LSTM/Transformer/…，14 组）
  test-end-to-end-training.lisp 端到端训练收敛（16 个用例）
  test-all-fixes.lisp           历史缺陷修复汇总（27 项）
  audit-verify.lisp             历史审查条目验证（11 项）
  pdf-bugs-verify.lisp          早期缺陷报告条目验证（7 项）
  test-cache-slots.lisp         前向缓存清理契约
  test-cosine-loss-final.lisp   cosine 损失梯度对拍
  smoke-mnist-convergence.lisp  真实 MNIST 上的收敛冒烟（需 chipz，约 1 分钟）

覆盖范围包括：历史 bug 回归（stop-gradient、BatchNorm 边界、LSTM 初始
状态梯度、KL 散度梯度、参数量估算、注意力偏置统计等）、数值梯度对拍
（SDPA / MHA / LSTM / GRU / Embedding / Conv2d / 各类池化）、优化器逐步
数值验证（SGD / Momentum / Nesterov / Adagrad / RMSprop / AdamW）、
调度器精确值序列、序列化往返，以及架构级压力测试与端到端收敛验证。


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

  包定义与导出              nn-package.lisp
  协议与基类                nn-protocol.lisp
  初始化器                  nn-initializer.lisp
  Dense / Activation / Flatten / Residual
                            nn-layers-dense.lisp
  Conv2d / 各类池化 / im2col
                            nn-layers-conv.lisp
  Dropout / BatchNorm / LayerNorm
                            nn-layers-norm.lisp
  Embedding                 nn-layers-embedding.lisp
  注意力与 Transformer      nn-layers-attention.lisp
  RNN / LSTM / GRU / 序列包装
                            nn-layers-rnn.lisp
  损失函数                  nn-loss.lisp
  优化器                    nn-optimizer.lisp
  学习率调度器              nn-scheduler.lisp
  Sequential / 序列化 / 深拷贝
                            nn-model.lisp
  示例（可运行脚本）        examples/
  测试（全部带断言）        tests/        入口 tests/run-tests.sh
  变更记录                  CHANGELOG.md


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

构建注意
--------------------------------------------------------------------------------

首次编译（冷缓存）需要给 SBCL 一个较大的堆，默认 1 GB 会在编译
clvt 的过程中耗尽：

    sbcl --dynamic-space-size 4096 \
         --eval '(ql:quickload :clnn)'

clvt 预编译完成后，后续加载 clnn 在默认堆下即可完成。


变更记录
--------------------------------------------------------------------------------

  各类缺陷修复、性能改进与行为变更的完整记录见 CHANGELOG.md。

  当前版本的关键修复（2026-10）：

    * 安全    load-model 未禁用 *read-eval*，模型文件可执行任意代码
    * 性能    ce-loss 用 (vt-eye C) 构造 one-hot，C=50000 时申请 20 GB 崩溃
    * 性能    embedding 的 max-norm 每次前向重归一化整表（O(V·D)）
    * 正确性  batch-norm 推理态反向梯度用错公式
    * 正确性  copy-network 丢弃 trainable / 优化器 params-grads 静默错位
    * 架构    协议泛型声明在最后加载的文件，加载顺序倒置

  每一条都在 CHANGELOG.md 中给出了现象、成因、修复与修复前后的量化对比，
  并配有 tests/ 下可复现的回归测试。

================================================================================
