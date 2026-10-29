# clnn 框架设计契约 / 接口契约文档

| 项目 | 内容 |
|---|---|
| 文档对象 | clnn（Common Lisp 神经网络库），版本 0.0.1 |
| 源码指纹 | gitee.com/xizangqt/clnn @ `aecb6f8`（2026-10-25） |
| 底层张量库 | clvt @ `181dfc2`（2026-10-28） |
| 验证环境 | SBCL 2.6.8 x86-64-linux + Quicklisp，本文档 §15 的 8 项语义实测均在该环境通过 |
| 文档性质 | 行为契约（behavioral contract）：描述"代码当前承诺的行为"，含少量设计权衡说明与扩展指引 |

**阅读约定**：`槽位名` 指 CLOS slot；`函数名`（如 `zero-grad!`）指 `nn` 包导出符号；代码引用标注来源文件；带 `!` 结尾的是有副作用操作。除特别说明，所有张量为 `vt` 类型、`double-float` 元素。

---

## 1. 架构总览

### 1.1 分层结构

```
┌─────────────────────────────────────────────────────┐
│ 应用层   examples/ (mnist, resnet9, transformer…)    │
├─────────────────────────────────────────────────────┤
│ 训练工具层  nn-model.lisp                             │
│   sequential 容器 / model-forward / model-update!    │
│   zero-grad! / clear-*-caches! / save/load-model     │
│   build-model / clipped-gradient-update!             │
├─────────────────────────────────────────────────────┤
│ 调度层   nn-scheduler.lisp（学习率调度）              │
├─────────────────────────────────────────────────────┤
│ 组件层                                               │
│   nn-layers-dense     dense / activation / flatten / residual │
│   nn-layers-conv      conv2d / max-avg-pool2d / global-avg    │
│   nn-layers-norm      dropout / batch-norm / layer-norm       │
│   nn-layers-attention sdpa / mha / transformer-block  │
│   nn-layers-rnn       rnn-cell / lstm / gru / rnn-sequence    │
│   nn-layers-embedding embedding                       │
│   nn-loss             mse / bce / ce / huber / smooth-l1      │
│   nn-optimizer        sgd / adam / adamw / rmsprop / adagrad  │
│   nn-initializer      he / xavier / zeros / …                │
├─────────────────────────────────────────────────────┤
│ 协议层   nn-protocol.lisp                             │
│   layer 基类 / forward / backward / params / grads    │
│   grad-slots / cache-slots / zero-grad-children       │
│   training-p / set-training! / with-training          │
├─────────────────────────────────────────────────────┤
│ 张量层   clvt（vt 张量 + 全套数学算子）               │
└─────────────────────────────────────────────────────┘
```

### 1.2 加载顺序契约（clnn.asd）

`clnn.asd` 为 `:serial t`，加载顺序固定为：
`nn-package → nn-protocol → nn-layers-embedding → nn-optimizer → nn-initializer → nn-layers-norm → nn-layers-attention → nn-layers-rnn → nn-layers-conv → nn-loss → nn-layers-dense → nn-model → nn-scheduler`。

**契约**：协议泛型（`forward`/`backward`/`params`/`grads`/`grad-slots` 等）在 `nn-protocol.lisp` 中声明，必须先于一切实现加载。新增源文件时，凡实现协议或依赖其它组件，都要遵守 serial 顺序把文件插到正确位置。

### 1.3 包与命名

- 包名 `nn`（昵称 `neural-net`、`clnn`）；从 `clvt` 显式导入约 100 个 `vt-*` 张量算子。
- 命名约定：参数槽 `weights/bias`，梯度槽 `dw/db`（或 `dgamma/dbeta`、`dw-q/dw-k/…`；注意 mha 的访问器名是 `mha-dwq/…`，与槽名 `dw-q/…` 不一致），缓存槽 `*-cache`；构造器 `make-<组件名>`；破坏性操作以 `!` 结尾。

---

## 2. 训练循环契约（核心）

### 2.1 标准循环：步骤与顺序

唯一受支持的训练循环形态如下（与 `examples/mnist.lisp` 的 `train-dense-mnist` 等完全一致）：

```lisp
(dotimes (step n)
  (zero-grad! model)                                 ; ① 显式清零（梯度槽置 NIL）
  (let* ((pred (forward model x))                    ; ② 前向
         (loss (compute-loss loss-fn pred y))        ; ③ 损失值
         (grad (compute-loss-gradient loss-fn pred y))) ; ④ dL/dpred
    (backward model grad)                            ; ⑤ 反向（覆盖写各层 dw/db）
    (optimizer-step opt (params model) (grads model)); ⑥ 参数更新
    (clear-step-caches! model)))                     ; ⑦ 可选：清理本步前向缓存
```

**顺序契约**（不可交换）：

1. `backward` 只能读"**最近一次 forward**"写入的缓存（z/a/input-cache 等）。两次 forward 之间不插入 backward，第一次的缓存即作废。
2. `optimizer-step` 只消费"**最近一次 backward**"写入的 `dw/db`。
3. `forward → backward` 必须在同一训练模式下完成；中途切换 `training-p`（影响 dropout/batch-norm）会使缓存与前向不一致。

**典型错误**：忘记 ① 就训练——见 §5.4 的影响分析；在 ②⑤ 之间调用 `clear-step-caches!`——backward 拿到 NIL 缓存将出错或算错。

### 2.2 是否每步自动清零？——不是；清零语义由"覆盖写"隐含完成

这是本框架与 PyTorch 最重要的差异，必须精确理解：

1. **没有全局自动清零机制**。框架不会在每个 step 开头替你清梯度；训练循环必须自己调用 `(zero-grad! model)`。
2. **但覆盖式层的 backward 本身就是"清零 + 写入"**：`dense/conv2d/batch-norm/layer-norm/mha` 的 backward 用 `setf` 把 `dw/db` 槽整体替换为本次计算出的**全新张量**（§3）。上一步的梯度张量在该瞬间被丢弃，等价于清零。
3. `zero-grad!` 的实际动作是把梯度槽**置 `NIL`**（不是置零张量），并递归到所有子层（`nn-model.lisp` `defmethod zero-grad! ((component t))`）。

因此"忘记 `zero-grad!`"的后果分层讨论：

| 层族 | 忘记 zero-grad! 的后果 |
|---|---|
| 覆盖写层族（dense/conv/norm/mha） | **数值结果不变**（dw 每步被整体替换），仅旧梯度张量多存活一步 |
| 累加层族（rnn-cell/lstm/gru/embedding） | **结果错误**：backward 是 `+=` 语义，跨步会持续累加 |
| 冻结层（`trainable = nil`） | 旧梯度若滞留，理论上会被 optimizer-step 重新施加；`zero-grad!` 对冻结层同样清零（源码注释明确此点：*"梯度清零与「是否参与更新」解耦。冻结层同样要清零，否则上一次的陈旧梯度会在 update 时被重新施加"*） |

**契约结论**：始终显式调用 `zero-grad!`，把它当作每个训练步的第一条语句；不要依赖覆盖写的隐含清零。

### 2.3 覆盖写发生在哪里

覆盖写全部发生在 **backward 方法内部** 与 **optimizer-step 的参数 setter 内部**：

| 位置 | 动作 | 语义 |
|---|---|---|
| `dense` backward | `(setf (dense-dw l) dw)` `(setf (dense-db l) db)` | 覆盖 |
| `conv2d` backward | `(setf (conv-dw l) dw)` `(setf (conv-db l) db)` | 覆盖 |
| `batch-norm` backward | `(setf (bn-dgamma l) …)` `(setf (bn-dbeta l) …)` | 覆盖 |
| `layer-norm` backward | `(setf (ln-dgamma l) …)` `(setf (ln-dbeta l) …)` | 覆盖 |
| `mha` backward | `(setf (mha-dwq l) d-wq …)`（8 个槽） | 覆盖 |
| `rnn-cell` backward | `(setf (rnn-dwih l) (if (rnn-dwih l) (vt-+ …) dwih))` | **累加**（BPTT） |
| `lstm/gru` backward | `(setf (lstm-dweight-ih l) (vt-+ (or … zeros) acc))` | **累加**（BPTT） |
| `embedding` backward | `(setf (emb-dw l) dw)`，dw 由 scatter-add 构建 | **scatter-add**（见 §5.2） |
| `optimizer-step` | `(funcall setter new-p)`，如 `(setf (dense-weights l) v)` | 参数被**新张量替换** |

### 2.4 变体循环

- **全局梯度裁剪**：`(clipped-gradient-update! model optimizer max-norm)` = 计算全模型梯度全局 L2 范数 → 超阈值则 `scale-all-grads!` 等比缩放 → `model-update!`。注意它**只缩放梯度，不清零**；且必须只调用一次（其实现按容器类型递归，重复调用会重复缩放）。
- **模型级封装**：`(model-forward model x)` / `(model-backward model g)` / `(model-update! model opt)` 是薄封装，`model-update!` 等价于 `(optimizer-step opt (params model) (grads model))`。
- **学习率调度**：`nn-scheduler.lisp` 在每个 step/epoch 后调整 `optimizer-lr`，与上述循环正交。
- **推理**：只 `forward`，不 `backward`、不 `optimizer-step`；建议每个 batch 后 `clear-step-caches!` 释放缓存（`examples/mnist.lisp` 的评估循环即如此）。

---

## 3. 层协议（Layer Protocol）

### 3.1 基类与必选方法

所有层派生自 `layer`（槽位：`name`、`trainable`、`training`）。一个可训练层需要提供：

| 方法 | 义务 |
|---|---|
| `forward (l input)` | 输入张量（或张量列表），返回输出张量；**写入本步缓存** |
| `backward (l grad-output)` | 输入 = 最近 forward 输出同形状的梯度张量；**写入本层梯度槽**；返回对输入的梯度 dX（多输入/有状态层返回 `values`） |
| `params (l)` | `((owner-layer "name" 参数张量 setter) …)`，与 `grads` **严格同构**（见 §3.4） |
| `grads (l)` | `(("name" . 梯度张量-or-NIL) …)`，顺序与 `params` 一一对应 |
| `grad-slots (l)` | 参与清零的梯度槽名列表（`zero-grad!` 用） |
| `cache-slots (l)` | 保存前向缓存的槽名列表（`clear-step-caches!` 用） |
| `set-training! (l mode)` | 容器层需 `call-next-method` 后递归子层 |

### 3.2 forward 契约

- 输入输出形状见 §9 的形状总表；核心约束是 **deterministic、缓存齐全**：backward 需要的中间量（z、a、输入、掩码、形状）必须在 forward 时全部写入缓存槽。
- 无参数层（activation/flatten/dropout/池化/residual/stop-gradient）`trainable` 默认 `nil`，`params` 返回空。
- 延迟初始化：`dense/conv2d/embedding/batch-norm` 等允许参数槽为 `NIL`，在**首次 forward** 时用 `nn-initializer` 初始化（dense 默认 He-normal，bias 默认零）。`(build-model model &optional dummy-input)` 提供显式初始化入口：按类型触发初始化 → 可选 dry-run forward → **`clear-step-caches!` 清掉 dry-run 垃圾**。

### 3.3 backward 契约

- **输入**：`grad-output`，形状必须等于最近一次 forward 的**输出**形状。
- **返回**：dX，形状必须等于最近一次 forward 的**输入**形状（rank、各维完全一致，含还原多维，见 §9.1）。
- **多输入/多输出层**的返回约定：
  - `rnn-cell`：`(values dx dh-prev)`；
  - `lstm`：`(values grad-input dh dc)`（`h-0/c-0` 未显式给出时 dh/dc 为 `NIL`）；
  - `gru`：`(values grad-input dh)`；
  - `multi-head-attention`：forward 输入是 `(list Q K V)`，backward 返回 `(values dQ dK dV)`；
  - 其余单输入单输出层返回单值；`stop-gradient-node` 返回与输入同形的**全零张量**（保证 sequential 容器梯度链不断；`grad` 为 `NIL` 时返回 `NIL`）。
- backward 内部不得依赖 forward 之外的任何全局状态（`*training-mode*` 除外，如 dropout/batch-norm 的推理路径）。

### 3.4 params/grads 严格同构契约

`optimizer-step` 用 `(loop for p in param-list for g in grad-list)` **按位置配对**。因此：

> **`params` 与 `grads` 必须长度相等、顺序一致、名称一致。**

- 参数存在但梯度尚未计算时，`grads` 必须返回 `(名称 . NIL)` 占位，**不能省略该项**（`dense` 的 `grads` 源码注释明确此点：*"否则两个列表长度不等，优化器按位置配对时会用别的层的梯度更新本层参数"*）。
- 长度不一致时 `assert-param-grad-alignment` 会显式 `error`（这是框架把"静默错位"变成硬错误的防线）。
- `NIL` 梯度 + `(layer-trainable-p owner)` 为假 + 参数为 `NIL`，三者任一成立时该参数在 `optimizer-step` 中被**跳过**（不更新、不报错）。

### 3.5 缓存槽语义（z / a / a-prev 及各层变体）

| 层 | 缓存槽 | 语义 |
|---|---|---|
| `dense` | `input-cache` | `(cons 原始形状 x-flat)`：反向还原多维 dX 用 |
| | `z-cache` | 线性输出 z = x·W(+b)（**激活前**） |
| | `a-cache` | 激活输出 a = f(z)（**激活后**） |
| | （a-prev） | **没有独立槽位**。"a-prev" 即 `input-cache` 的 cdr（本层输入的展平视图），是算 `dw = a-prevᵀ·δ` 的左操作数 |
| `activation-layer` | `z-cache` / `a-cache` | 按 kind 只存需要的那一个（§8） |
| `flatten` | `cache` | forward 输入的原始形状 |
| `conv2d` | `input-cache` / `col-cache` / `output-shape-cache` | 原输入 (B,C,H,W)、im2col 矩阵、输出形状 |
| `max-pool2d` | `cache`（单槽） | 反向按最大值位置路由梯度 |
| `avg-pool2d` | `input-cache` / `count-cache` | 计数矩阵做平均 |
| `dropout` | `mask-cache` | 训练态缩放掩码；推理态置 `NIL` |
| `batch-norm` | `input-cache` / `xhat-cache` / `std-inv-cache` | 另有 `batch-size`、`fwd-mode`（`:batch`/`:running`）**不属于** cache-slots，不随缓存清理消失；`running-mean/running-var` 是持久统计量，**永不清理** |
| `layer-norm` | `input-cache` / `xhat-cache` / `std-inv-cache` | |
| `embedding` | `indices-cache` / `norm-scale` | |
| `rnn-cell` | `input-cache` / `h-prev-cache` / `h-cache` | 另有 `state`（跨 forward 的隐状态，`reset-rnn-cell-state!` 清） |
| `lstm` / `gru` | `cache` | plist：`:gates :hs :x :c …` 全部时间步门控值 |
| `rnn-sequence` | `x-cache` / `h-prev-cache` / `h-cache` | BPTT 所需的逐步张量列表 |
| `sdpa` / `mha` | `cache` | attn-raw、dropout mask、q/k/v 及 2D 展开等 |

**缓存契约**：
1. backward 只信任"最近一次 forward"的缓存；缓存槽一律**覆盖写**（`setf` 新张量）。
2. `clear-forward-cache!` 只清 `transient-cache-slots`（默认 `()`，普通层为空操作，任意时刻调用安全）；`clear-step-caches!` 清 `cache-slots` 全部槽位（置 `NIL`），**只能在 forward+backward 完成一个训练步后调用**；容器层（sequential/residual/transformer-block/rnn-sequence）递归清子层。
3. 推理循环每 batch 后调 `clear-step-caches!` 防内存累积。

### 3.6 容器契约（sequential）

- `sequential` 假设每个子层满足**单输入单输出、单梯度进单梯度出**；`rnn-cell`（双值返回）、`multi-head-attention`（三输入）、裸 `lstm/gru`（多值返回）**不得**直接放入，需用 `rnn-sequence` 包装或自定义容器。
- `params`/`grads` 按子层顺序 `nconc` 聚合；`zero-grad!`/`clear-step-caches!`/`set-training!` 经 `zero-grad-children` 等协议递归。
- 同一层对象**加入两个容器**即构成参数共享，默认不支持（§6.2）。

---

## 4. 梯度语义（重点）

### 4.1 dense 的 dw/db 是"本次反向传播计算出的梯度"，不是累计梯度

以 `dense` 为准（`nn-layers-dense.lisp` backward）：

```lisp
;; 反向主体（简化）
(d-act  (vt-* grad-flat (激活导数(z-cache 或 a-cache))))
(dw     (vt-matmul (vt-transpose a-prev) d-act))   ; (in-dim, out-dim)
(db     (vt-sum d-act :axis 0))                    ; (out-dim,)
;; —— 覆盖写 ——
(setf (dense-dw l) dw)
(setf (dense-db l) db)
;; —— 返回 dX ——
(dx-flat (vt-matmul d-act (vt-transpose w)))
(…(vt-reshape dx-flat 原输入形状))
```

**含义**：
1. `dw`/`db` 在每次 backward 时**从零计算、整体替换**。backward 后读到的 `dw` 恰好等于"本次 (grad-output, 缓存) 单独决定的那一个值"。连续两次 backward，第二次的 `dw` 与第一次**无关**（实测 §15-T1：第二次值 ≠ 第一次+增量，而是精确等于第二次单独计算值，逐位相等）。
2. **覆盖写隐含清零**：不需要先置零再加，`setf` 本身同时完成"丢弃旧值"和"写入新值"。这就是"本框架默认覆盖 + 隐式清零"的机制基础。
3. `dw` 的形状恒等于 `weights` 的形状（dense 为 `(in-dim, out-dim)`，注意与 PyTorch 的 `(out, in)` 权重布局相反），`db` 恒等于 `bias` 的形状。

### 4.2 各层族梯度写法总表

| 层族 | backward 写法 | 跨 backward 调用 | 语义 |
|---|---|---|---|
| dense / conv2d / batch-norm / layer-norm / mha | `setf` 新张量 | 覆盖 | "本次算出的梯度" |
| rnn-cell | `(if 旧值 (vt-+ 旧值 增量) 增量)` | **累加** | BPTT：同一 cell 的 backward 在一个序列内被调 T 次，权重梯度必须累加 |
| lstm / gru | `(vt-+ (or 旧值 zeros) acc)` | **累加** | 同上 |
| embedding | `dw = (or 旧值 zeros)`，再 scatter-add | **scatter-add** | 一次 backward 内同一行可被命中多次必须累加；起点取当前 `dw`（`zero-grad!` 置 `NIL` 后即从零开始） |
| 无参数层 | 无梯度槽 | — | 直接回传 dX |

**一致性保证**：标准训练循环中每步先 `zero-grad!`（全部置 `NIL`），因此对用户而言**所有层的每步有效语义都是"从零开始得到本步梯度"**——覆盖族由 setf 保证，累加族由 zero-grad! 保证。差异只在"忘记清零"和"步内多次 backward"时显现（§6.1/§6.3）。

### 4.3 NIL 梯度语义

- 槽位初始为 `NIL`；`zero-grad!` 后为 `NIL`；backward 后为 `vt`。
- `NIL` 表示"本步没有为该参数计算梯度"，`optimizer-step` 对其**静默跳过**（实测 §15-T3）。
- 推论：只 forward 不 backward 的层（如推理分支、冻结层）参数不会被更新；自定义层若 backward 未写梯度槽，优化器也不会动它——这既是宽容也是陷阱（拼写错槽名会静默失效），所以 `grads` 必须与 `params` 同构（§3.4）。

### 4.4 优化器不清零梯度

`optimizer-step` 消费 `grads` 返回的梯度后**不回写、不清零**：backward 之后梯度槽保持原值（实测 §15-T7，步进后 `(dense-dw d)` 仍是旧对象）。清零完全由训练循环下一步的 `zero-grad!` 负责。PyTorch 用户特别注意：这里没有 `optimizer.zero_grad()` 的等价职责。

### 4.5 冻结层

`trainable = nil` 的层：`optimizer-step` 跳过其参数；`zero-grad!` **仍然清零**其梯度槽（防陈旧梯度复活）。`param-count :trainable-only t` 可统计可训练参数量。

---

## 5. 限制与不支持场景（默认不支持 + 如何改）

以下场景**当前契约明确不支持**，给了改法但不建议直接在生产使用：

### 5.1 梯度累加（gradient accumulation）

- 现状：覆盖写层族第二次 backward 会**丢弃**上一次梯度；只有 rnn 族/embedding 会累加。不存在"多次 backward 累加再统一 step"的默认行为。
- 如需支持：
  1. **改梯度写入层**：把覆盖写层的 backward 从 `(setf (dense-dw l) dw)` 改为 `(setf (dense-dw l) (if (dense-dw l) (vt-+ (dense-dw l) dw) dw))`（rnn-cell 已示范此写法）；
  2. **或改调度**：引入 `*grad-accum-policy*`，在 `zero-grad!` 中按策略跳过覆盖写层，N 个 micro-batch 后再 `optimizer-step`。注意 embedding/rnn 族本来就会累加，方案 1 对它们幂等，方案 2 必须区别对待。
  3. loss 的 reduction 需相应按 micro-batch 数加权。

### 5.2 参数共享（parameter sharing / tied weights）

- 现状：`optimizer-state-registry` 以 `(list owner-layer 参数名)` 为键；同一层对象被两个容器引用时，`params` 会经两条路径把它收集**两次**，配对契约（§3.4）被破坏，更新与状态都会错位。框架不提供命名共享/权重绑定机制。
- 如需支持：实现"主实例 + 引用"协议——让引用包装层的 `params` 返回**主实例**的 `(owner name …)` 三元组，并保证 `collect-all-layers`/`zero-grad-children` 去重（如按 `eq` 建表）。Transformer 的 tied embedding/output head 需要此类改造。

### 5.3 同一层一次前向多次调用 / 一次前向多次 backward

- **同一层对象在一次 forward 中被调用两次**（权重复用、循环展开共享 cell 等）：后一次调用**覆盖**前一次的缓存（z/a/input-cache 都是单槽），第一次的计算结果无法 backward。
  - rnn-sequence 之所以可行，是因为它**不依赖单槽缓存**：序列信息存入 `x-cache/h-cache` **列表**，并把 cell 的单槽缓存按时间步逐一重写后再调 cell backward。
  - 如需支持：仿照 rnn-sequence——把单槽缓存改为按调用次序追加的列表，backward 按相反次序消费；同时该层梯度必须改为累加式（§4.2 rnn 族写法）。
- **一次 forward 多次 backward**：覆盖写层第二次 backward 返回的 dX 仍正确（缓存未动，输出确定性），但 `dw/db` 是同一值重复 setf，即**梯度不会翻倍**；累加族（rnn/embedding）则会**翻倍**——两种族行为不一致，跨层组合时结果不可预测。如需支持应统一改造（所有层走累加式 + 显式 zero_grad 语义，即全面 PyTorch 化）。

### 5.4 其它不支持项

| 场景 | 现状 | 出路 |
|---|---|---|
| 非线性/动态计算图（if/while 依赖数据） | 无自动微分磁带；backward 是手写静态链 | 在模型层显式组织控制流，保证每个 forward 链有对应的手写 backward 链 |
| 多输入多输出层进 sequential | 容器契约禁止（§3.6） | 写包装层或自定义容器（`zero-grad-children`/`params`/`grads`/递归清理四件套） |
| 二阶导 / 高阶优化 | 导数均为手写一阶 | 需要数值验证时用 `tests/gradcheck-all-layers.lisp` 的有限差分对拍 |
| 混合精度 / 稀疏梯度 | 全程 `double-float` 密集张量 | 无 |
| 线程安全 | 缓存/梯度/优化器状态全部是可变槽；`*training-mode*` 是全局动态变量 | 单线程训练契约；并发请每线程独立模型与优化器实例 |
| batch-norm batch=1 训练 | 训练态 running-var 不更新（源码 `when (> batch 1)`），批统计无意义 | 训练时 batch ≥ 2，或改用 layer-norm |

---

## 6. 与主流框架差异表

以 PyTorch 为参照：

| 维度 | PyTorch | clnn |
|---|---|---|
| 微分机制 | 动态计算图（autograd 磁带） | 手写静态 backward 链 + 显式缓存 |
| 梯度累加默认行为 | **默认累加**：`.grad += new` | **默认覆盖**：`setf` 新张量（dense/conv/norm/mha） |
| 清零方式 | 显式 `optimizer.zero_grad()` / `model.zero_grad()` | 显式 `zero-grad!`（把槽置 `NIL`）；覆盖式层的 backward **隐含清零** |
| 忘记清零的后果 | 梯度跨步累加 → 训练发散（对所有层一致） | 覆盖族无影响；**rnn/embedding 族会错**（不一致，见 §4.2） |
| 梯度存放在哪 | 参数对象的 `.grad` 属性 | 层对象独立槽位（`dw/db/dgamma/…`），与参数槽分离 |
| 优化器与梯度关系 | optimizer 不清零、只更新 | 相同：optimizer 不清零；`optimizer-step` 后梯度仍在 |
| NIL 梯度 | `.grad` 为 None 时该参数不更新 | `(name . NIL)` 占位 + 跳过；两者等价，但 clnn 要求占位以维持列表同构 |
| backward 调用对象 | `loss.backward()`（从损失节点出发） | `(backward model grad)`（显式传入 dL/dpred；损失与模型解耦） |
| 损失梯度 | autograd 沿图回传 | `compute-loss-gradient` 返回 dL/dpred，交给 `backward` 作为链起点 |
| 参数更新方式 | 原地 `param.data -= …` | **setter 替换整个张量对象**（`(setf (dense-weights l) new-vt)`），更新后参数是**新对象** |
| 权重布局 | Linear: `(out, in)` | dense: `(in, out)`（`dw = a_prevᵀ·δ` 的自然形状） |
| 训练/推理切换 | `model.train()/eval()`（模块树状态） | `set-model-training!` / `with-training`；层未显式设置时回退全局 `*training-mode*` |
| 梯度裁剪 | `clip_grad_norm_`（全局范数，用户手动调用） | 优化器 `:grad-clip`（step 内自动全局范数裁剪）或 `clipped-gradient-update!` |
| AdamW 权重衰减默认 | 0.01 | `make-adamw` 默认 0.01（解耦衰减）；`make-adam` 默认 **0.0**（L2 衰减需显式传参，与 PyTorch Adam 默认一致） |
| 学习率调度 | 调度器包装优化器 | 直接改 `optimizer-lr` 槽 |
| 序列化 | torch.save（pickle） | `save-model`（plist 文本序列化，逐位一致；`load-model` 已禁 `*read-eval*`） |
| batch-norm 标签约定 | (B,C,H,W)，C=第 1 维 | rank-2：(B,F) F=末维；ND 路径将特征维移到末轴归一化 |

---

## 7. 激活函数缓存约定

`activation-layer` 与 `dense`（内置激活）采用同一套约定。**导数函数期望的输入**是理解反向代码的关键：

### 7.1 缓存选择表

| kind | forward 缓存 | backward 使用的量 | 导数期望输入 |
|---|---|---|---|
| `:relu` | `z-cache`（输入） | `grad · vt-relu-derivative(z)` | **z**（激活前） |
| `:leaky-relu` | `z-cache` | `grad · vt-leaky-relu-derivative(z)` | **z** |
| `:gelu` | `z-cache` | `grad · vt-gelu-derivative(z)` | **z** |
| `:swish` | `z-cache` | `grad · vt-swish-derivative(z)` | **z** |
| `:mish` | `z-cache` | `grad · vt-mish-derivative(z)` | **z** |
| `:softplus` | `z-cache` | `grad · sigmoid(z)`（导数就地计算） | **z** |
| `:hard-tanh` | `z-cache` | `grad · (−1≤z≤1 ? 1 : 0)`（就地计算） | **z** |
| `:hard-sigmoid` | `z-cache` | `grad · vt-hard-sigmoid-derivative(z)` | **z** |
| `:sigmoid` | `a-cache`（输出） | `grad · a(1−a)` | **a**（激活后） |
| `:tanh` | `a-cache` | `grad · (1−a²)` | **a**（激活后） |
| `:softmax` | `z-cache` + `a-cache` **双缓存** | `gz = y ⊙ (gy − Σ gy·y)`，y = a-cache | **a** |
| `:log-softmax` | `z-cache` + `a-cache` **双缓存** | `gz = gy − s·Σ gy`，s = **由 z-cache 重算** `softmax(z)` | **z**（重算 s） |
| `:linear` / `:none` | 均为 `NIL` | `grad` 原样回传 | — |

### 7.2 语义说明

- **z-cache 语义**：本层输入（对 activation-layer）或线性变换输出（对 dense）——即"进入激活函数之前"的值。凡导数能写成 z 的显式函数（分段线性、含 z 的代数式）的激活，一律用 z-cache。
- **a-cache 语义**：激活输出。凡导数用输出表示更简单（`σ′ = a(1−a)`、`tanh′ = 1−a²`）或根本离不开输出（softmax 的雅可比）的激活，用 a-cache。
- **公开导数函数的期望输入**：`vt-relu-derivative / vt-leaky-relu-derivative / vt-gelu-derivative / vt-swish-derivative / vt-mish-derivative / vt-hard-sigmoid-derivative` 期望 **z**；`vt-sigmoid-derivative / vt-tanh-derivative` 期望 **a**。自定义层复用这些函数时不要传错对象。
- softmax/log-softmax 双缓存的原因：softmax 的雅可比要用输出 y，log-softmax 的梯度公式用 s = softmax(z)——前者有现成 a-cache 可直接用，后者为避免数值问题选择**从 z 重算** softmax 而非 exp(a)。
- dense 与独立 activation-layer 的分工：`(make-dense n :activation :relu)` 把激活并入 dense（缓存 z/a 各一）；显式 `(make-activation-layer :relu)` 则是独立层。两者反向约定一致，混用时注意 dense 的 a-cache 是激活后值。

---

## 8. 形状约定

### 8.1 dense：rank 1/2/N 的展平与还原（含 grad-flat reshape 规则）

forward（`nn-layers-dense.lisp`）：

```lisp
(in-dim      (or (dense-in-dim l) (car (last in-shape))))  ; 末维=特征维（PyTorch 约定）
(batch-size  (if (= rank 1) 1 (reduce #'* (butlast in-shape)))) ; 前面所有维相乘=Batch
(x-flat      (if (and (= rank 2) (= (second in-shape) in-dim))
                 input                                    ; 标准 2D：零拷贝
                 (vt-reshape input (list batch-size in-dim)))) ; rank≠2：展平
```

| 输入 rank | 处理 | 输出形状 | backward 输入 | dX 返回形状 |
|---|---|---|---|---|
| 1 `(D,)` | 视为 batch=1，reshape `(1, in-dim)` | `(out-dim,)`（去掉 batch 维） | `(out-dim,)`，内部 unsqueeze 成 `(1, out-dim)` | `(D,)` |
| 2 `(B, in-dim)` | 零拷贝直用 | `(B, out-dim)` | `(B, out-dim)` | `(B, in-dim)` |
| ≥3 `(B₁,…,Bₖ, in-dim)` | reshape `(B₁·…·Bₖ, in-dim)` | `(B₁,…,Bₖ, out-dim)`（激活后按原始 in-shape 还原多维） | 同输出形状，内部再展平 `(·, out-dim)` | **还原为原始输入形状**（`input-cache` 的 car 记录的 in-shape） |

**grad-flat reshape 规则**（反向侧的镜像）：
1. backward 把 `grad-output` 按与 forward 相同的规则展平成 `(batch, out-dim)`；
2. 计算 `dx-flat = d-act · Wᵀ`，形状 `(batch, in-dim)`；
3. **用 `input-cache` 缓存的原始形状整体 reshape 还原**——不是逐维裁剪，因此任何 rank 的输入都能还原。`dw`/`db` 与展平方式无关，只由 `(in-dim, out-dim)` 决定。
4. `in-dim` 处理：未声明时以**输入末维**回填（PyTorch 风格）；已声明时以声明值为准。**dense 没有专门的形状断言**——声明值与输入元素总数不匹配时，错误会在展平/`vt-reshape`（元素数不符）处抛出，而非清晰的形状检查；使用 `build-model` 或首次 forward 传标准形状输入可提前暴露此类问题。

### 8.2 其它层形状总表

| 层 | forward 输入 → 输出 | backward 输入 → 返回 (dX) |
|---|---|---|
| conv2d | `(B,C,H,W)` → `(B,out-c,OH,OW)`（im2col：`(B·OH·OW, C·Kh·Kw)`） | `(B,out-c,OH,OW)` → `(B,C,H,W)`（col2im 按位置累加还原） |
| max/avg-pool2d | `(B,C,H,W)` → `(B,C,OH,OW)` | 同形互逆 |
| global-avg-pool2d | `(B,C,H,W)` → `(B,C)` | `(B,C)` → `(B,C,H,W)` |
| flatten（start-dim=1） | `(B,d₁,…,dₖ)` → `(B, d₁·…·dₖ)` | `(B,·)` → 缓存的原始形状 |
| batch-norm | rank-2 `(B,F)`；ND 沿"特征维在末轴"路径（轴置换后 `(spatial, nf)` 归一化） | 同形互逆 |
| layer-norm | 任意 rank，按 `normalized-shape`（末 K 维）归一化 | 同形互逆 |
| embedding | 整数索引任意 rank `(i₁,…,iₖ)` → `(i₁,…,iₖ, D)` | `(i₁,…,iₖ,D)` → **`vt-zeros(idx-shape)`**（索引是整数，无梯度，dX 恒为零张量） |
| rnn-cell | `(B,in)` → `(B,H)` | `(B,H)` → `(values (B,in) (B,H))` |
| rnn-sequence | `(B,T,in)` → `(B,T,H)` | `(B,T,H)` → `(B,T,in)` |
| lstm | `(B,T,in)` → `(values output (B,T,H) h (B,H) c (B,H))` | `(B,T,H)` → `(values (B,T,in) dh? dc?)` |
| mha | `(list Q K V)`，各 `(B,S,dim)` → `(B,Sq,embed-dim)` | `(B,Sq,ed)` → `(values dQ (B,Sq,dim) dK (B,Sk,dim_k) dV (B,Sk,dim_v))` |
| transformer-block | `(B,S,D)` → `(B,S,D)`（pre-norm） | 同形互逆（MHA 分支与残差直连梯度相加） |
| ce-loss | logits 必须为 2D `(N,C)`（类别维=第 1 维）；标签为**一维 int64 整数张量** | — |

**通用规则**：除 dense 的 batch 展平与 flatten 的 start-dim 外，**所有层都不改变输入 rank**，backward 严格按 forward 缓存的形状还原；多维到 2D 的任何转换都发生在层内部且必须可逆。

---

## 9. 优化器契约

- **输入**：`param-list = ((owner-layer "name" param setter) …)` 与 `grad-list = (("name" . grad-or-NIL) …)`，按位置配对，长度必须相等（`assert-param-grad-alignment` 硬校验）。
- **步内流程**：`global-clip-coeff`（全参数梯度全局 L2 范数裁剪，`clip ≤ 0` 不裁）→ 逐参数：`NIL` 梯度/冻结层跳过 → L2 权重衰减 `g + wd·param`（adamw 为解耦衰减）→ 更新优化器状态（registry 以 `(owner 参数名)` 为键，惰性建零张量）→ `funcall setter new-param` **替换参数张量**。
- **状态**：`state-registry` 哈希表 + `step-count`（每 step 增 1，作 Adam 偏差校正）。跨 step 持久；换优化器实例即状态归零。
- **家族**：sgd（可选 momentum/nesterov）、adam（可选 amsgrad）、adamw、rmsprop（alpha/centered/momentum）、adagrad（lr-decay）。
- **不清零梯度**（§4.4）；不校验梯度是否"新鲜"——backward 之后忘记 step 或 step 两次是用户错误。

## 10. 损失函数契约

- `compute-loss (loss predicted target) → 标量（或 :none 时逐元素张量）`；`compute-loss-gradient → dL/dpredicted，形状恒等于 predicted`。
- `reduction ∈ {:mean :sum :none}`；`:mean` 时梯度除以元素总数（huber/ce 按 batch 等，详见各自实现）。
- **CE 专项契约**：predicted 必须是 2D `(N,C)` logits（内部 log-softmax + NLL，数值稳定）；target 是一维 int64 类别索引；支持 label-smoothing。序列任务须先 reshape 成 `(N·T, C)`。
- 损失对象无状态、无缓存，可与任意模型自由组合；标准循环中"损失梯度"就是 backward 的入口张量（§2.1 ④⑤）。

## 11. 序列化契约

- `save-model (model filepath)`：遍历 `collect-all-layers`，槽位转 plist 文本；`load-model (filepath)` 重建并逐位还原参数。加载环境需存在同名层类；`load-model` 已绑定 `*read-eval* = nil`（安全修复：防止模型文件执行任意代码）。`copy-network` 提供内存内深拷贝（含 trainable 与优化器梯度槽对齐）。

---

## 12. 自定义层开发契约（checklist）

实现一个新层的最小义务（缺一不可）：

```lisp
(defclass my-linear (layer)
  ((w :accessor my-w) (dw :accessor my-dw :initform nil)
   (x-cache :accessor my-x-cache :initform nil)))

(defmethod forward ((l my-linear) x)
  (setf (my-x-cache l) x)                 ; 1. 缓存 backward 需要的一切
  (vt-matmul x (my-w l)))

(defmethod backward ((l my-linear) grad-output)
  (let* ((x  (my-x-cache l))
         (dw (vt-matmul (vt-transpose x) grad-output)))
    (setf (my-dw l) dw)                   ; 2. 梯度槽覆盖写（本步梯度，非累加）
    (vt-matmul grad-output (vt-transpose (my-w l)))))  ; 3. 返回 dX，形状=输入

(defmethod params ((l my-linear))         ; 4. params/grads 严格同构
  (list (list l "w" (my-w l) #'(lambda (v) (setf (my-w l) v)))))
(defmethod grads ((l my-linear))
  (list (cons "w" (my-dw l))))            ; 未反传时 NIL 占位
(defmethod grad-slots ((l my-linear)) '(dw))
(defmethod cache-slots ((l my-linear)) '(x-cache))
```

验收清单：① `params`/`grads` 长度顺序一致；② backward 返回值形状 == 最近 forward 输入形状；③ 有限差分对拍（参照 `tests/gradcheck-all-layers.lisp`）；④ `zero-grad!` 后梯度槽为 `NIL`；⑤ `clear-step-caches!` 后 backward 不再被调用；⑥ 容器层补 `zero-grad-children` + 递归清理 + `set-training!`。

## 13. 缓存生命周期总结

```
build-model ──dry-run──→ clear-step-caches!   （清掉试探缓存）
    │
训练步: zero-grad! → forward(写缓存) → backward(读缓存/写梯度) → optimizer-step → clear-step-caches!
    │
推理步: forward → clear-step-caches!
```

- `clear-forward-cache!`：只清 `transient-cache-slots`（默认空），**任意时刻安全**，供特殊层释放大块可再生物（实测 §15-T6）。
- `clear-step-caches!`：清 `cache-slots` 全部 + 递归子层，**只能在整个训练步结束后调用**；`build-model` 用它清 dry-run 垃圾（注释明确选它的原因：dry-run 后不接 backward，要"清全部"而不是"清可丢弃"）。
- 不属于缓存、清理不触碰的持久状态：`running-mean/running-var`（BN 统计量）、`rnn-state`（用 `reset-rnn-cell-state!` 单独清）、优化器 registry。

## 14. 版本与演进注意

本文档描述行为以 §开头指纹的源码为准。`CHANGELOG.md` 记录的行为级修复（如 batch-norm 推理反向公式、copy-network 梯度槽对齐、协议泛型加载顺序）都直接影响本契约，升级版本时应重新核对 §4.2/§5 的层级行为表。

---

## 15. 实证记录（SBCL 2.6.8，源码指纹见文档头）

以下 8 项由 `scripts/clnn-contract-smoke.lisp` 实际运行验证（`sbcl --load`，断言全过）：

| # | 验证点 | 结果 |
|---|---|---|
| T1 | dense 连续两次 backward：`dw` 精确等于第二次单独计算值（覆盖写，非累加） | ✅ 逐位相等（maxdiff = 0.0） |
| T2 | `zero-grad!` 将 `dw/db` 置 `NIL`；`params/grads` 保持同构 | ✅ |
| T3 | 全部梯度为 `NIL` 时 `optimizer-step` 跳过所有参数、不报错、参数对象不变 | ✅ |
| T4 | rnn-cell 连续两次 backward：`dwih/dwhh/dbih` 累加（BPTT 契约）；forward 持久化 `rnn-state`，单步使用须先 `reset-rnn-cell-state!` | ✅ |
| T5 | embedding backward 为 scatter-add：未清零时跨调用累加；`zero-grad!` 后从零重建 | ✅ |
| T6 | `clear-forward-cache!` 在 forward/backward 之间调用安全（不影响 backward）；`clear-step-caches!` 后 `input-cache/z-cache/a-cache` 全为 `NIL` | ✅ |
| T7 | `optimizer-step` 后参数张量被 setter 替换为新对象；梯度槽保持旧值（清零归训练循环） | ✅ |
| T8 | 标准六步循环（§2.1）跑通 2 层 MLP + CE + Adam 50 步，loss 稳定降至 ln3≈1.0986 以下 | ✅ |

## 附录 A：源文件地图

| 文件 | 职责 |
|---|---|
| `nn-package.lisp` | 包定义与 clvt 符号导入 |
| `nn-protocol.lisp` | `layer` 基类、`forward/backward/params/grads/grad-slots/cache-slots/zero-grad-children/clear-*-caches!` 泛型、训练模式协议、loss/optimizer 基类、`stop-gradient-node` |
| `nn-layers-dense.lisp` | dense / activation-layer / flatten / residual、全部激活导数函数 |
| `nn-layers-conv.lisp` | im2col/col2im、conv2d、三种池化 |
| `nn-layers-norm.lisp` | dropout / batch-norm（2D+ND 双路径）/ layer-norm |
| `nn-layers-rnn.lisp` | rnn-cell / lstm / gru / rnn-sequence（BPTT） |
| `nn-layers-attention.lisp` | sdpa / multi-head-attention / transformer-block |
| `nn-layers-embedding.lisp` | embedding（max-norm / scale-grad-by-freq） |
| `nn-optimizer.lisp` | 五个优化器、全局范数裁剪、配对断言 |
| `nn-loss.lisp` | mse / bce / ce / huber / smooth-l1 |
| `nn-model.lisp` | sequential、训练工具、清零与缓存协议实现、序列化、copy-network、neural-network-compat |
| `nn-scheduler.lisp` | 学习率调度 |
| `tests/gradcheck-all-layers.lisp` | 全层有限差分对拍 |

## 附录 B：常见错误速查

| 症状 | 根因 | 对照契约 |
|---|---|---|
| rnn/embedding 训练 loss 不降 | 忘记每步 `zero-grad!`（累加族真的累加了） | §2.2 |
| 部分参数永远不更新 | 自定义层 `params` 与 `grads` 项数不一致但未触发断言（旧版本），或 backward 忘写梯度槽 | §3.4/§4.3 |
| backward 拿到旧值/崩溃 | 一次 forward 两次 backward，或两次 forward 之间夹了 backward | §2.1/§5.3 |
| 优化器状态错位 | 同一层对象放进两个容器（参数共享） | §5.2 |
| CE 报形状/类型错 | logits 非 `(N,C)` 或标签非一维 int64 | §10 |
| BN batch=1 训练统计异常 | running-var 仅在 batch>1 时更新 | §5.4 |
| dense 输入末维不匹配 | 多维输入未按"末维=特征维"组织 | §8.1 |
