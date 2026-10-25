# 变更记录

本文件记录 clnn 的缺陷修复、性能改进与行为变更。面向使用者的用法说明见
README.md；本文件面向维护者，重点是「现象 → 成因 → 修复 → 量化效果」。

格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循语义化版本（当前 `clnn.asd` 中为 `0.0.1`）。

---

## [未发布]

### 2026-10-25 · 卷积与池化热路径优化

动机：上游合并后做了一次 profile，发现 `conv2d` 的前向里 `im2col` 一步就占 57%
（37/64 ms），而池化的单次前向要 34 ms —— 两者都与「纯 Lisp 的物理瓶颈」这个
既有结论不符，属于实现问题而非语言问题。逐项定位后做了三处改写。

#### 1. `im2col` / `col2im`：内部/边界分离 + 基址预计算

旧实现对每个输出位置都逐元素判断 `(>= ih 0) (< ih in-h)`、`(>= iw 0) (< iw in-w)`，
并在内层重复计算 `(+ (* i sh) ki (- ph))` 这类乘法。8×32×28×28 → 32 通道 k3
配置下，1.8M 次内层迭代平均 20ns/元素 —— 而同样元素量的裸「读+写」只需 4.6 ms。

改写：

* **内部/边界分离**：先用一次判据 `(and i-inside (>= iw0 0) (<= (+ iw0 kw) in-w))`
  判出「整个 KH×KW 块完全落在输入内」的位置，走无逐元素检查的快路径；
  只有跨边界的位置才逐元素判断。典型 CNN（k=3, pad=1）中约 86% 走快路径。
* **基址预计算**：行基址（输入侧、输出侧）在通道循环外一次算好，
  内层只做「基址 + 常量偏移」。
* `col2im` 做同构改写（重叠区域仍是 `incf` 累加）。

另外补了一个 dtype 守卫：这几个核直接用 `vt-data` 读底层数组、并按
`(simple-array double-float (*))` 声明访问。SBCL 信任这类声明，若传入
float32 张量会读到错误的位模式（静默错值而不是报错）。现在入口统一检查
`vt-dtype`，给出「请先 `vt-astype` 转 float64」的明确提示。

| 项目（8×32×28×28 → 32ch, k3, pad=1） | 优化前 | 优化后 | 提升 |
|---|---:|---:|---:|
| `im2col` 单项 | 45.7 ms | 19.0 ms | 2.4× |
| `conv2d` forward | 64.5 ms | 43.2 ms | 1.49× |
| `conv2d` backward | 95.4 ms | 66.2 ms | 1.44× |
| `conv2d` forward（pad=0） | 57.0 ms | 32.0 ms | 1.78× |
| `conv2d` backward（pad=0） | 76.0 ms | 55.0 ms | 1.38× |

#### 2. `max-pool2d` / `avg-pool2d`：去掉逐元素闭包调用

旧实现用 `labels` 定义 `lin-idx` 计算行主序索引，每个候选值都要走一次
局部函数调用 + 4 次乘法。改为「每通道基址 + 行偏移」后只需加法。

`avg-pool2d` 还额外做了一件蠢事：反向传播里为了算 `1/cnt`，把「数一遍有效
元素」和「散射梯度」写成两个完全相同的 kh×kw 循环，白跑一遍。现在前向把
每个输出位置的有效元素数缓存成 (oh,ow) 的 fixnum 向量（只与 (i,j) 有关，
与 (b,c) 无关），反向直接查表。

| 项目（8×32×56×56, k2） | 优化前 | 优化后 | 提升 |
|---|---:|---:|---:|
| `max-pool2d` forward | 34.3 ms | 8.3 ms | 4.1× |
| `max-pool2d` forward 分配 | 22.2 MB | 10.0 MB | 2.2× |

#### 3. 端到端效果

| 项目 | 优化前 | 优化后 | 提升 |
|---|---:|---:|---:|
| 小型 CNN 训练步（16×1×28×28） | 33.1 ms | 21.9 ms | 1.51× |
| MNIST 冒烟测试（4000 张 / 240 步） | 58 s | 46 s | 1.26× |

#### 新增测试

`tests/test-conv-im2col.lisp`（26 项）。这类热路径改写最容易在边界上出错，
而梯度对拍只能发现「梯度不一致」，定位不到「展开错了」，因此这里直接用
**最直白的 6 重循环参考实现**逐元素对拍：

* `im2col` 11 组配置（无 padding / 同尺寸 / stride 2 / 1×1 核 / 5×5 核 /
  非方形核与 stride / kernel == 输入 / 小图）；
* `col2im` 5 组配置（含完全重叠的 sum-pooling 型）；
* `im2col` 与 `col2im` 的内积伴随性 `<Ax,c> = <x,Aᵀc>`（3 组）；
* `conv2d` 前向 vs 直接卷积参考实现（5 组）；
* 非连续输入（BatchNorm 输出）喂给池化/卷积与连续拷贝结果一致。

#### 未改动：`clvt` 的 GEMM 是当前最大的剩余瓶颈

profile 显示 `conv2d`/`dense` 的时间已基本全部落在 `vt-matmul` 上，
而它只跑到 **2.2–2.8 GMAC/s**：

| 形状 | 吞吐 |
|---|---:|
| (128×512)@(512×512) | 2.30 GMAC/s |
| (512×512)@(512×512) | 2.19 GMAC/s |
| (1024×1024)@(1024×1024) | 1.84 GMAC/s |
| (6272×288)@(288×32) | 2.79 GMAC/s |

参照点：本机（AVX2 + FMA + AVX512，32 核）上一条**纯标量** 4 链 FMA 循环可达
4.89 GMAC/s；关掉 `*simd-matmul-enabled*` 掉到 1.0 GMAC/s（说明 SIMD 快路径
确实生效，只是利用率低）。内核的 `jj/kk` 分块把 `l`（k 维）放在最内层，
每个 k 步都要从内存读-改-写 C 分块，属于带宽受限结构。

这是 `clvt` 的改动，本次未动；如要做，方向是「i 分块 + C 常驻寄存器」的
经典 GEMM 分块，预期 2–4 倍。

### 2026-10-25 · 目录重组与示例修复

#### 变更（目录与测试基建）

* **`tests/` 只保留测试，示例移入 `examples/`。**
  原先 `tests/` 混放了 10 个可运行示例（MNIST / ResNet-9 / 强化学习 /
  聚类 / 架构演示等），其中多数需要数分钟才能跑完，却和回归测试混在一起，
  导致「跑一遍 tests」既慢又不清晰。现在：

  | 目录 | 内容 | 数量 |
  |---|---|---|
  | `tests/` | 全部带断言的测试 | 13 |
  | `examples/` | 可独立运行的示例脚本 | 10 |

  移动的文件：`binocular-mnist`、`cartpole-reinforce`、`conv2d-batched`、
  `dac-mnist`、`demo-last-bit-final`、`mnist`、`prove`、`resnet9-cifar`、
  `resnet9-mnist-lite`、`transformer-sum-classifier`。

* **两个「名字像示例、内容其实是测试」的文件重命名**（保持在 `tests/`）：

  | 原名 | 新名 | 理由 |
  |---|---|---|
  | `example.lisp` | `test-architectures.lisp` | 内容是 14 组架构回归（断言 + 训练收敛），不是示例 |
  | `run-all-examples.lisp` | `test-end-to-end-training.lisp` | 内容是 16 个端到端训练用例，也不是「运行 examples/ 的入口」 |

* **所有测试文件在存在失败项时以非 0 退出**（`#+sbcl (sb-ext:exit :code 1)`）。
  原先只有 `test-architecture-fixes.lisp` 这么做，其余文件失败后进程仍返回 0，
  无法接入 CI。`gradcheck-all-layers.lisp` 的失败计数是列表（空时为 `NIL`），
  因此用 `(length *fails*)` 而非 `plusp` 判断。

* **新增 `tests/run-tests.sh`**：统一的测试入口，支持 `--fast` / `--slow` /
  指定文件名，逐个运行并汇总，失败时保留日志。

      cd tests && ./run-tests.sh          # 全部，约 1.5 分钟
      cd tests && ./run-tests.sh --fast   # 快速套件，约 20 秒

* 各测试/示例文件补齐头部说明（用途、用法、耗时），使两个目录自解释；
  示例内的 `(load "xxx.lisp")` 用法注释统一改为 `examples/xxx.lisp`。

#### 修复（examples）

* **`cartpole-reinforce.lisp` 加载到第 6 段就中断，后 3 个示例从未被定义。**

  该文件由 9 个曾经的独立脚本拼接而成，其中 LunarLander 的 v1 与 v2
  各自 `defstruct lunarlander`，但槽数不同（10 vs 9）：

      (defstruct lunarlander (x …) … (steps 0) (prev-leg-contact nil))   ; v1, 10 槽
      (defstruct lunarlander (x …) … (steps 0))                          ; v2,  9 槽

  第二次定义触发硬错误，`load` 在那一行终止：

      attempt to redefine the STRUCTURE-OBJECT class NN::LUNARLANDER
      incompatibly with the current definition

  后果是第 6/7/8 段（LunarLander v2、PPO、PPO v2、PPO v3）的函数
  **全部不存在**，调用 `(train-ppo)` 会直接 `undefined function`。

  修复：给 v2 段的符号加独立前缀 —— `lunarlander` → `lunarlander-v2`、
  `ll-*` → `llv2-*`，段内自洽重命名。验证结果：

  * `(load "examples/cartpole-reinforce.lisp")` 成功；
  * 9 个入口 `train-reinforce` / `train-a2c` / `train-dqn` /
    `train-mountaincar-dqn` / `train-lunarlander-dqn` /
    `train-lunarlander-dqn-v2` / `train-ppo` / `train-ppo-v2` /
    `train-ppo-v3` 全部 `fboundp`；
  * 两套环境互不干扰：v1 结构体 10 槽、`prev-leg-contact` 可读，
    v2 结构体 9 槽；各跑 1 个 episode 均正常返回。

* **`cartpole-reinforce.lisp` 加载时自动跑 500 episode 的 REINFORCE。**
  其余 8 段都只定义、由使用者显式调用，唯独第 1 段残留了顶层的
  `(train-reinforce)`，使「加载文件」变成「开始训练一分多钟」。
  已注释掉并在文件头部写明显式调用方式。

#### 已知问题（未改动，仅标注）

* `resnet9-cifar.lisp` 由两次迭代拼接而成，每个函数定义了两次
  （差异是 `train-resnet9-cifar` 默认 `epochs` 5 → 10），编译时报
  `Duplicate definition for … found in one file`。后一份覆盖前一份，
  加载与运行正常。已在文件头注明，如需清理可删掉前一份。
* `demo-last-bit-final.lisp` 保留了同一示例的三次迭代，`take-last`
  类被定义三次（槽结构相同），编译时报 layout 重定义警告；运行正常。
  已在文件头注明。

---

### 2026-10-25 · 架构审查修复

以下问题均由可复现的测试定位，修复后均有对应回归测试
（`tests/test-architecture-fixes.lisp`，43 项）。审查基线是当时的 HEAD：
仓库自带的 91 项断言全绿，也就是说**没有一个缺陷被既有测试覆盖**。

#### 安全

* **`load-model` 未禁用 `*read-eval*`**

  模型文件是纯数据 plist，但 `read` 默认允许 `#.` 读宏求值。构造一个含
  `#.(progn (with-open-file …) nil)` 的 `.sexp`，调用 `load-model` 后目标
  文件被创建 —— 模型加载等价于任意代码执行。

  修复：`load-model` 内绑定 `*read-eval*` 为 `NIL`。

#### 性能

* **`ce-loss` 用 `(vt-eye C)` + `vt-take` 构造 one-hot**

  `vt-eye C` 是 C×C 密集单位矩阵，且 `compute-loss` 与
  `compute-loss-gradient` 各做一次。

  | 类别数 C | 修复前 | 修复后 |
  |---:|---|---|
  | 1 000 | 4 ms / 7.6 MB | 1 ms |
  | 4 000 | 13 ms / 122 MB | 6 ms |
  | 50 000 | **申请 20 000 000 016 字节，堆耗尽崩溃** | 38 ms |

  原始报错：

      Heap exhausted during allocation: 6189875200 bytes available,
      20000000016 requested.

  修复：目标分布 = `(1-s)·one-hot + s/C`，按 `(batch, C)` 一次填充
  （初值 `s/C`，再把 `batch` 个标签位置改成 `(1-s)+s/C`），
  label smoothing 与 one-hot 共用这一次分配。复杂度 O(batch×C)。

* **`embedding` 的 `max-norm` 每次前向重归一化整张词表**

  旧实现对 (V, D) 表逐元素 `vt-ref` 求范数并缩放，复杂度 O(V·D)，与本次
  实际查表数量无关；同时也不符合 PyTorch 语义（`embedding_renorm_` 只处理
  本次用到的行）。

  | 词表 V / 维度 D | 修复前 | 修复后 |
  |---|---:|---:|
  | 1 000 / 16 | 0.35 ms | < 0.05 ms |
  | 10 000 / 32 | 6.6 ms | < 0.05 ms |
  | 50 000 / 64 | 70.9 ms | < 0.05 ms |

  （均只查 4 个索引；不开 max-norm 的同配置前向 < 0.05 ms 作为参照。）

  修复：新增 `renorm-embedding-rows!` —— 索引去重后用一次 `vt-take` +
  `vt-sum :axis 1` 向量化求出行范数，只对超限行逐元素缩放；无超限行时
  连 (V,) 向量都不分配。`renorm-embedding-weights!` 保留为显式整理全表的
  工具，同样改为向量化。

  **行为变更**：`max-norm` 语义从「整表约束」改为「只约束本次用到的行」
  （与 PyTorch 一致），README 与测试已同步。

* **`rmsprop` 每步为每个参数分配一次 `vt-zeros`**

  `(gethash key registry (vt-zeros …))` 的默认值是函数实参，即使键已存在
  也必定先求值，于是每步每参数都白分配一份参数规模的零张量。改为
  `or` + `setf` 惰性创建。

* **`global-avg-pool2d` 反向多分配一份满尺寸常量**

  旧实现用 `vt-const` 造一个 (B,C,H,W) 全 1 张量参与相乘，反向内存翻倍。
  改为对 (B,C,1,1) 缩放后经 `vt-broadcast-to` 物化，只保留梯度本身。

#### 修复

* **`copy-network` 丢弃 `trainable`**

  `sequential` / `activation-layer` / `flatten` / `dropout` / `residual`
  的拷贝分支没有把 `trainable` 传下去（`make-flatten` / `make-dropout`
  干脆不接受该关键字，`make-sequential` 硬编码 `t`），冻结层复制后会
  变回可训练，`sequential` 甚至从 `NIL` 变回 `T`。
  修复：给四个构造函数补 `:trainable` 关键字（向后兼容），5 个拷贝分支传递。

* **`batch-norm` 推理态的反向梯度是错的**

  两处缺陷叠加：推理分支不写 `xhat` / `std-inv` 缓存（反向读到上一轮训练
  残留的值），且反向无条件套用训练态公式（含 `-Σdxhat` 与
  `-xhat·Σ(dxhat·xhat)` 两个均值修正项）。推理态下均值/方差来自滑动统计量、
  与输入无关，正确结果是 `dx = dxhat · std_inv`。

  有限差分核对：修复前最大绝对误差 **1.27**，修复后 **< 1e-6**。
  修复：新增 `bn-fwd-mode` 记录本次前向用的是 `:batch` 还是 `:running`，
  推理分支同时写缓存，反向按模式分派公式（训练态公式本身正确，未改）。

* **`build-model` 不初始化 `rnn-sequence` 内部的 cell**

  `collect-all-layers` 把 `rnn-sequence` 当作叶子返回，但 `typecase` 没有
  对应分支，导致 `build-model` 之后 `param-count` 仍是 0。

* **`embedding` 关闭 `max-norm` 后 `norm-scale` 残留**

  缩放系数只在启用时写入，关闭后旧值仍在，反向会拿陈旧系数缩放梯度。

* **优化器对 `params` / `grads` 长度不一致静默截断**

  `optimizer-step` 用 `(loop for p in params for g in grads …)` 按位置配对，
  长度不同时 `loop` 在较短列表结束时静默退出：后半部分参数永远不更新，
  且没有任何提示。典型成因是自定义层的 `params` 与 `grads` 返回长度不同。

  修复：新增 `assert-param-grad-alignment`，5 个优化器（sgd / adam / adamw /
  rmsprop / adagrad）在更新前校验，不一致直接报错且参数未被修改。

* **`tensor-one-hot` 不支持多维索引**

  对 `(batch, seq)` 形式的 token id 会报「索引数量 1 与维度数 2 不匹配」，
  且不校验越界。修复：先 `vt-flatten` 索引并对越界索引报错。

#### 架构

* **协议泛型声明位置错误**

  `zero-grad-children` / `zero-grad!` / `transient-cache-slots` /
  `clear-forward-cache!` / `clear-step-caches!` 的 `defgeneric` 写在**最后
  加载**的 `nn-model.lisp`，而 `nn-layers-attention.lisp` 已先行定义了方法：

      WARNING: redefining NN:ZERO-GRAD-CHILDREN in DEFGENERIC

  这既产生编译告警，也违反 `nn-package.lisp` 中写明的「协议层必须最先
  加载」约定 —— 一旦 `nn-model.lisp` 加载失败，就会留下没有默认方法的
  残缺泛型，任何未实现该泛型的自定义层在 `zero-grad!` 时都会崩。

  修复：5 个声明前移到 `nn-protocol.lisp`，`nn-model.lisp` 只留方法。
  现在全量编译零告警。

#### 测试

新增三个测试文件：

| 文件 | 内容 |
|---|---|
| `tests/test-architecture-fixes.lisp` | 43 项，逐条对应上述缺陷 |
| `tests/gradcheck-all-layers.lisp` | 全层/全损失 解析梯度 vs 中心差分 |
| `tests/smoke-mnist-convergence.lisp` | 真实 MNIST 4000 张 / 240 步 / 9k 参数 CNN |

梯度对拍覆盖：dense 的 11 种激活、conv2d 的 5 种 stride/padding 组合、
三种池化、batch-norm 的 2D/ND/无仿射、layer-norm、SDPA/MHA、
transformer-block、残差、LSTM/GRU/rnn-sequence、embedding，以及 7 类损失
× 3 种归约 —— **全部一致，未发现任何反向传播错误**。

MNIST 冒烟结果：loss 2.43 → 0.093，测试准确率 92.08%。

另修正了一个原有测试的隐患：`test-regression-fixes.lisp` 的 max-norm 用例
断言「所有行都 ≤ 上限」，依赖随机初始化恰好不超限（约 2.5% 概率偶发失败）。
已改为断言确定性质（用到的行满足上限 + 未用到的行逐位不变）。

---

## 已知限制

* 首次冷编译需要 ≥ 4 GB 堆（默认 1 GB 会在编译 **clvt** 时耗尽，
  失败点在 clvt 而非 clnn）。clvt 预编译后 clnn 在默认堆下可正常编译。
  见 README「构建注意」。
* `clvt` 的 NT GEMM 内核（`%matmul-df-nt-*`）是死代码，而 `conv2d` 的
  `col·Wᵀ` 正是 NT 情形，目前多一次转置拷贝。属于 clvt 的改动。
* `ce-loss` 只接受 `(batch, n-classes)` 的 2D logits，3D 会给出明确报错
  而非静默算错（与库整体「最后一维是特征」的风格一致，未改语义）。
* `conv2d` 的 im2col 内存放大与纯 Lisp 吞吐上限（约 1.4 GMAC/s）是语言
  层瓶颈，需要 BLAS/GPU 后端。
