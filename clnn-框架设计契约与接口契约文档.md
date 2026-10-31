# clnn 框架设计契约 / 接口契约文档

| 项目 | 内容 |
|---|---|
| 文档对象 | clnn（Common Lisp 神经网络库），版本 0.0.1 |
| 文档版本 | **v2.3** |
| 底层张量库 | clvt @ `f19d8e1` |
| 验证环境 | SBCL 2.6.10 x86-64-linux + Quicklisp |
| 文档性质 | **行为契约**（代码当前承诺什么）+ **接口契约**（clnn 怎么用 clvt）+ **缺口清单** |

**这份文档解决什么问题**：clnn 是纯手写求导的框架（§1.5）—— 没有 autograd 替你
记录计算图、传播精度、累加梯度。框架能做对的只有那些「在契约里写清楚、并被测试
钉住」的事；没写清的，使用者会以最难查的方式踩到：**不报错、数值上讲得通、效果悄悄
变差**。所以本文档只描述当前代码承诺的行为，凡是有代价的约定都直接给出后果。

**阅读约定**：`槽位名` 指 CLOS slot；`函数名`（如 `zero-grad!`）指 `nn` 包导出符号；
带 `!` 结尾的是有副作用操作；**D1–D5** 是类型铁律、**M1–M7** 是纯手写求导架构的
协定（§1.5）、**G-*** 是契约缺口。

---

## 0. 文档结构

### 0.1 章节地图

| 域 | 章节 | 回答什么问题 |
|---|---|---|
| 骨架 | §1 架构总览 | 有哪些层、怎么加载、包怎么组织、与 clvt 的边界在哪 |
| 类型 | §2 类型契约 | 哪些 dtype 允许出现在哪里，违反会怎样 |
| 行为 | §3 训练循环 / §4 层协议 / §5 梯度语义 | 一个训练步的正确写法与顺序约束 |
| 接口 | §6 vt-* 调用面契约 | clnn 怎么用 clvt，哪些用法被禁止、为什么 |
| 边界 | §7 限制 / §8 与 PyTorch 差异 | 什么不支持、与主流框架差在哪 |
| 细节 | §9–§13 激活 / 形状 / 优化器 / 损失 / 序列化 | 各子系统的具体契约 |
| 扩展 | §14 自定义层 / §15 缓存生命周期 | 加一层要满足什么 |
| 演进 | §16 clvt 变更跟踪矩阵 | clvt 升级后要检查 clnn 的哪些地方 |
| 证据 | §17 验收探针 | 哪些结论有可复跑的断言支撑 |

### 0.2 本文档的三条写作原则

1. **只写被验证过的行为**。凡是「实测」条目都能在 `tests/` 里找到对应断言；
   未实测的推断明确标为「推断」，或者不写。
2. **区分「现状」与「应有」**。契约描述现状；现状违背设计意图的地方进入 §2.8 的
   缺口清单，而不是被悄悄写成「应然」—— 把缺陷写成文档比没有文档更坏。
3. **可执行**。§2 的类型契约由 `tests/dtype-contract-test.lisp`（77 断言）钉住，
   §3/§5/§13 的运行期契约由 `tests/test-runtime-contract.lisp`（26 断言）钉住，
   其余由 §17 列出的套件覆盖。

### 0.3 修订记录

| 版本 | 主要变化 |
|---|---|
| v1.0 | 首次成文：训练循环、层协议、梯度语义、形状、缓存、序列化 |
| v2.0 | 新增 §2 类型契约、§6 vt-* 接口契约、§16 clvt 变更跟踪矩阵 |
| v2.1 | 新增 §1.5 纯手写求导架构（M1–M7）；梯度语义统一为覆盖写 |
| v2.2 | 单一计算域：`+COMPUTE-DTYPE+ = :float64`，f32 降级为「入口被规约的输入」 |
| **v2.3** | 补齐三处静默失败：损失 target 不过闸门（§5.7）、优化步步号按调用次数计（§3.5）、`load-model` 不切推理态（§13.2）；文档删除历史叙述 |

---


## 1. 架构总览

### 1.1 分层结构

```
┌──────────────────────────────────────────────────────────────┐
│ 应用层    examples/（mnist, resnet9, transformer, cartpole…） │
├──────────────────────────────────────────────────────────────┤
│ 训练工具层 nn-model.lisp                                      │
│   sequential 容器 / model-forward / model-backward           │
│   model-update! / zero-grad! / clear-*-caches!               │
│   build-model / clipped-gradient-update! / save-model        │
│   tensor-* 工具族（where / masked-fill / one-hot / gather…）  │
├──────────────────────────────────────────────────────────────┤
│ 调度层   nn-scheduler.lisp（lr 调度，与训练循环正交）          │
├──────────────────────────────────────────────────────────────┤
│ 组件层                                                        │
│   nn-layers-dense      dense / activation / flatten / residual│
│   nn-layers-conv       conv2d / max-avg-pool2d / global-avg   │
│   nn-layers-norm       dropout / batch-norm / layer-norm      │
│   nn-layers-attention  sdpa / multi-head-attention / TB       │
│   nn-layers-rnn        rnn-cell / lstm / gru / rnn-sequence   │
│   nn-layers-embedding  embedding                              │
│   nn-loss              mse / bce / ce / huber / kl / cosine   │
│   nn-optimizer         sgd / adam / adamw / rmsprop / adagrad │
│   nn-initializer       he / xavier / orthogonal / zeros / …   │
├──────────────────────────────────────────────────────────────┤
│ 协议层   nn-protocol.lisp                                     │
│   layer 基类 / forward / backward / params / grads           │
│   grad-slots / cache-slots / zero-grad-children              │
│   training-p / set-training! / with-training                 │
│   dtype 契约工具（§2）：integer-dtype-p / assert-index-dtype…  │
├──────────────────────────────────────────────────────────────┤
│ 张量层   clvt（vt 张量 + 全套数学算子 + :bool / 复数 dtype）   │
└──────────────────────────────────────────────────────────────┘
```

**设计原则**：clnn 是 clvt 的**消费者**，不是它的合作者。
两者之间只有单向依赖：clnn 调 clvt 的公开 API，clvt 不知道 clnn 存在。因此

- clnn **不重定义** clvt 的语义（不改 `vt-*` 的 dtype 规则、不改提升表）；
- clnn 在**自己的边界上**把 clvt 的宽松语义收窄成框架语义（例如 clvt 允许布尔下标，
  clnn 拒绝）；
- clvt 的语义变化必须先进入 §16 的影响矩阵，再落到 §2 的铁律和 §6 的禁令里。

### 1.2 加载顺序契约（clnn.asd）

`clnn.asd` 为 `:serial t`，顺序固定：

```
nn-package → nn-protocol → nn-layers-embedding → nn-optimizer → nn-initializer
→ nn-layers-norm → nn-layers-attention → nn-layers-rnn → nn-layers-conv
→ nn-loss → nn-layers-dense → nn-model → nn-scheduler
```

**契约**：

1. 协议泛型（`forward`/`backward`/`params`/`grads`/`grad-slots`/`cache-slots`/
   `zero-grad-children`/`clear-*-caches!`）必须在 `nn-protocol.lisp` 声明 —— 否则
   `nn-layers-attention.lisp` 里的 `defmethod` 会变成「先有方法后有泛型」，SBCL 会告警
   并留下残缺泛型（勿回退）。
2. **§2 的 dtype 工具函数也必须放在 `nn-protocol.lisp`**：`embedding`（第 3 个加载）
   与 `nn-loss`、`nn-model` 都要用 `assert-index-dtype`。放到 `nn-model.lisp` 会让
   embedding 在编译期看到未定义函数。
3. 新增源文件时，按「被依赖者在前」插入；`nn-scheduler.lisp` 依赖 `optimizer` 槽位
   访问器，必须最后加载。

### 1.3 包与命名

- 包名 `nn`（昵称 `neural-net`、`clnn`）；`vt-*` 命名空间下共有 **106 个可调用符号**：
  其中 **94 个**从 `clvt` 显式导入，**12 个**是 clnn 自己定义的 `nn::vt-*`（见 G-10）。
- 显式导入而非 `:use`：避免 clvt 新增导出时污染 `nn` 包（clvt 有 700+ 导出符号）。
- 命名约定：

| 类别 | 约定 | 例 |
|---|---|---|
| 参数槽 | `weights` / `bias` / `gamma` / `beta` | `dense-weights` |
| 梯度槽 | `dw` / `db` / `dgamma` / `dbeta` / `dweight-ih` | `dense-dw` |
| 缓存槽 | `*-cache` / `cache` | `dense-z-cache` |
| 构造器 | `make-<组件>` | `make-dense` |
| 破坏性操作 | 尾部 `!` | `zero-grad!` |
| 私有辅助 | 前缀 `%` | `%ce-target`、`%mask-true-p` |

**已知不一致（保留，勿「顺手修」）**：`mha` 的槽名是 `dw-q/db-q…`，访问器却是
`mha-dwq/mha-dbq`（没有连字符）。改名会破坏 `save-model` 兼容性与外部代码，
故只在文档中登记。

### 1.4 依赖边界（clnn 只用 clvt 的公开 API）

| 项 | 契约 |
|---|---|
| 允许 | `clvt:vt-*` 等**导出**符号；`clvt` 包内的公开访问器（`vt-shape`/`vt-dtype`/`vt-data`/`vt-offset`/`vt-strides`） |
| 禁止 | 任何 `%` 开头的 clvt 内部函数（如 `%vt-ref-raw`）。它们随重构消失，且语义不在 clvt 的契约范围内 |
| 禁止 | 直接操作 `vt-data` 缓冲区做写入（`vt-data` 只读遍历用；写请走 `vt-ref`/`vt-put`/`vt-copy-into`） |
| 校验方式 | §17.4 的符号审计：`nn` 包内 106 个 `vt-*`（fboundp）中，94 个的符号家是 `clvt` 且为 `:external`，12 个为 clnn 自有 |
| 版本 | `clnn.asd` 只声明 `:depends-on (:clvt …)`，**不锁版本号**。clvt 的语义变更靠 §16 矩阵人工跟踪 |

### 1.5 纯手写求导架构（本框架的立身之本）

#### 1.5.1 事实：clvt 不含任何微分机制

clvt 是**纯张量库**：它提供存储、广播、算子、归约与 IO，但**没有**计算图、没有
`grad_fn`、没有磁带（tape）、没有伴随模式（adjoint/reverse-mode）、也没有对偶数
（dual number）。它的 `vt-gradient`（`src/reduce-stats.lisp`）是**数值梯度**——
对标 `numpy.gradient` 的有限差分，用于「已知网格上的采样数据的斜率」，既不知道
也不关心某个张量是不是「由谁算出来的」。

因此：

> **clnn 没有任何自动微分可依赖。每一条反向传播公式都是人推的、手写的、逐层落到
> `(defmethod backward …)` 里的。**

**这不是「一开始就这么设计」，而是试过、做不成、删掉了。** git 里留着证据：

```
dd0ea79 重构：删除 autograd、修复 bug、完善逻辑
  删除 nn-autograd.lisp（clvt 无法完全支持 diff-tensor 自动微分）
```

那 195 行是一个逆向模式的 autograd：`diff-tensor` 包一层 `data`/`grad`/`grad-fn`，
只覆盖 `ag-+`、`ag-matmul` 等少数算子，且与层的 `cache`/`grad` 槽体系无法共存 ——
层是按「forward 自己缓存、backward 手写公式」写的，autograd 要的是「算子粒度 +
计算图」，两套体系要么全改层、要么全改 autograd。它被删掉了。

所以现在的定位是：**个人研究级别，可读、可预测、可逐位对拍**，但起点是一次失败的
尝试。它是后面所有条款（§5 梯度语义、§7.3 配对限制、§2.0 单一计算域）的根因。

#### 1.5.2 一次训练步里，数学是怎么流动的

```
forward:   x ──► 各层把「反向要用到的中间量」写进自己的 *-cache 槽 ──► y
                                            │
compute-loss-gradient:  dL/dy 由损失函数手写（不是 loss.backward()）
                                            │
backward:  dL/dy ──► 各层读自己的 *-cache，按手推公式算 ──┬─► 写参数梯度槽 dw/db/…
                                                          └─► 返回 dX 给上一层
optimizer-step:  (params, grads) 两张同构表逐位更新参数对象
```

三处容易被误认成「自动求导」的东西，都不是：

| 名字 | 它其实是什么 | 不是什么 |
|---|---|---|
| `compute-loss-gradient` | 每个损失类手写的一阶导（`nn-loss.lisp` 里 7 个 `defmethod`） | 不是 `loss.backward()`，它不会向下游传播 |
| `stop-gradient-node` | 一个 `forward` 直通、`backward` 返回零张量的**层** | 不是「切断计算图」——没有图可切，它只是把梯度置零 |
| `params` / `grads` | 两张手写枚举表，靠 `§4.4` 的同构契约配对 | 不是「参数对象自带 `.grad`」 |

#### 1.5.3 七条协定（M1–M7）

这七条是「手写」这个前提**必然推出**的约束，不是风格偏好。违反任何一条，症状都是
**静默算错**（loss 少降一点、梯度小一截），而不是抛异常——所以逐条列出，供代码审查
与自定义层自查。

| 编号 | 协定 | 违反的后果 | 守护手段 |
|---|---|---|---|
| **M1** | **每层必须自缓存**：`backward` 用到的前向中间量，必须由本层 `forward` 写进 `cache-slots` 列出的槽；`backward` 不得重新跑前向 | 重算会让「反向用的激活」与「前向实际输出」不一致（参数已改、dropout 已换掩码），梯度静默偏离 | `manual-backward-contract-test` M1a/M1b（行为级）+ **M1c（文本扫描：21 个 `backward` 方法体内 `forward` 调用数 = 0）** |
| **M2** | **缓存是单次前向的快照**：`backward` 只读不写前向缓存 | 写回会让同一份缓存支撑的第二次 `backward` 用错中间量 | 同上（人工审查 + §17 实证） |
| **M3** | **forward 与 backward 一一配对**：一次 `forward` 只支撑一次 `backward`（多时间步展开由 `rnn-sequence` 在层内部完成） | 第二次 `backward` 配的是同一份缓存，结果与第一次相同（覆盖写），但调用者常误以为是「又算了一步」 | §7.3 明确列为不支持场景 |
| **M4** | **梯度槽语义统一为覆盖写**：一次 `forward`+`backward` 之后，`grads` 给的是「本次反向算出的梯度」，与上一次无关 | 混合语义：忘记 `zero-grad!` 时 dense/conv 不受影响、rnn/embedding 翻倍，同一模型两种行为 | §5.1/§5.2 表格 + `contract-probes` T4 |
| **M5** | **容器只调度，不算数学**：`sequential`/`residual`/`transformer-block` 的 `backward` 只做「倒序调用子层 + 传递 dX」 | 容器里塞进「自己的导数」会让梯度与层的组合关系脱钩，无法单独验证 | §4.6；三者的 `backward` 均为循环调用 |
| **M6** | **只提供一阶解析导**：二阶导、Hessian、数值差分**不在库内**（有限差分只出现在 `tests/`，用于对拍） | 把数值差分当生产梯度：正确但慢几个量级，且步长敏感 | `tests/gradcheck-all-layers.lisp`（对拍用）；src 中无 `vt-gradient` 调用 |
| **M7** | **参数更新必须显式两步**：`grads ← backward`、`params ← optimizer-step` / `update!`；没有「自动读梯度」 | 以为调了某个函数就更新了参数（`update!` 的默认方法若是空操作，正是这种陷阱） | §5.4；`update!` 的 LAYER 方法 |

**M1 的量化证据**：对 21 个 `(defmethod backward …)` 逐个体检，函数体内 `forward`
调用数为 **0**、数值差分调用数为 **0**。换言之「重算前向」与「差分近似」这两种
最省事的作弊方式，在本库里一次都没发生。

#### 1.5.4 与 autograd 框架的责任划分

| 职责 | PyTorch（autograd） | clnn（手写） | 谁在 clnn 里负责 |
|---|---|---|---|
| 记录算子与依赖 | 框架自动（磁带） | **不存在** | — |
| 前向中间量保存 | 框架按需保留张量 | 每层自己 `setf` 进 `*-cache` | 各层的 `forward`（M1） |
| 反向公式 | 每个算子的 `backward` 内核 | 每层一个 `defmethod backward` | 各层作者（人） |
| 反向调度顺序 | 拓扑排序 | `sequential` 倒序循环 | 容器（M5） |
| 梯度存放 | 参数对象 `.grad` | 层对象的独立槽（`dw/db/…`） | `grad-slots` 协议 |
| 梯度清零 | `zero_grad()`（可选，默认累加） | `zero-grad!`（可选，默认覆盖，M4） | 训练循环 |
| 参数更新 | `optimizer.step()` 读 `.grad` | `(optimizer-step opt (params m) (grads m))` | 调用者（M7） |

结论：clnn 里**没有任何一环是「自动」的**——每一环都有一个具体的、可以指认的
函数或方法在负责。这也意味着**没有任何一环能替用户兜底**：漏缓存就是漏缓存，
漏清零就是漏清零。所以本库的策略是「把每一条都写成断言」。

#### 1.5.5 扩展者自查清单（写一个新层时逐条打勾）

1. `forward` 里把 `backward` 需要的**每一个**中间量都存进槽，并在 `cache-slots` 里登记。
2. `backward` 的首行就把缓存取出；若缓存为 `NIL`，明确报错而不是继续（如 `dense` 的做法）。
3. 参数梯度**覆盖写**（`setf` 新张量），不要 `(vt-+ 旧值 增量)`——除非你也实现并文档化了「累加模式」。
4. dX 的 dtype 跟随**本次 forward 的输入**（`vt-zeros-like input`），不要 `(vt-zeros shape)`。
5. 零张量一律 `vt-zeros-like` 或显式 `:dtype`（§2.2 D2/D3）。
6. `grads` 必须与 `params` 同构（同名、同序、NIL 占位），否则 `optimizer-step` 会错位（§4.4）。
7. 在 `grad-slots` 里列出自己的梯度槽名，否则 `zero-grad!` 清不到（§5.5）。
8. 用 `tests/gradcheck-all-layers.lisp` 的有限差分对拍一遍——**这是唯一能证明手推公式正确的办法**。

---

## 2. 类型契约（dtype）

本章回答三件事：**哪种 dtype 允许出现在哪个位置、违反时是报错还是静默出错、
以及为什么**。核心决定是单一计算域（§2.0）：`float32` 不是一种可选精度，而是一个
会在入口被规约掉的输入类型。

### 2.0 单一计算域：为什么只有一个浮点精度

**决定**：clnn 的全部计算——参数、梯度、前向缓存、激活、损失、优化器状态——**恒为
`:float64`**，由 `+COMPUTE-DTYPE+` 唯一定义。`float32` 不被拒绝，但会在进入层的
边界处被规约一次；`float32` 之后的任何中间量都不会再出现。

这条决定的依据有两条，一条是架构、一条是实测。

**① 架构依据：clnn 没有 autograd，精度传播只能靠人肉追踪。**

§1.5 已确立 clnn 是纯手写求导。多精度在**有**自动微分的框架里是可控的（框架知道每个
中间量的来源，可以在图上统一标注）；在纯手写架构里，它变成「每一处内部缓冲都要人肉
判断该跟谁同 dtype」。而 clnn 的这类缓冲有几十处：

| 缓冲类型 | 处数 | 涉及文件 |
|---|---|---|
| 状态缓冲（RNN 的 h / c） | 8 | `nn-layers-rnn.lisp` |
| 梯度累加器（`dwih-acc` 等） | 12 | `nn-layers-rnn.lisp`、`nn-layers-embedding.lisp` |
| running 统计量（bn 的 mean/var） | 2 | `nn-layers-norm.lisp` |
| 零 dX / 零梯度 | 14 | 优化器、dropout、residual、stop-gradient |
| 掩码、目标分布、损失中间量 | 5 以上 | `nn-layers-norm.lisp`、`nn-loss.lisp` |

这类缓冲单看都「正确地继承了参照张量」，合起来却能让 f32 模型在第五层之后悄悄
变成 f64 —— 而且没有任何报错。多精度路线的隐性成本不是写代码，是**每次改动都要
重新证明这几十处仍然一致**。

**② 实测依据：混着跑比统一精度慢一个数量级。**

环境：SBCL 2.6.10 x86-64-linux，4000×4000 张量，每个操作重复 20 次取总。

| 运算 | 耗时 | 相对 f64 基准 |
|---|---|---|
| `vt-+` f64 ⊔ f64（基准） | 852 ms | 1.0× |
| `vt-+` f32 ⊔ f32 | 584 ms | 0.69× |
| `vt-*` 标量 × f32（弱标量，无转换） | 600 ms | 0.70× |
| **`vt-+` f32 ⊔ f64** | **7165 ms** | **8.4×** |

注意第二行：**纯 f32 比纯 f64 快**（内存带宽减半，这是真的）。慢的不是 f32，是
「f32 与 f64 混在同一张图里」—— 每个算子都要先按 §3.4 的提升表定结果 dtype、分配
f64 临时数组、再逐元素转换。而最容易混的恰恰是默认配置：参数 f64（初始化器决定）、
输入可能是 f32（数据源决定）。

另外，clvt 唯一的 AVX-512 GEMM 微内核（`src/gemm-avx512.lisp`）**只发射 float64
指令**；f32 的矩阵乘走的是通用 sb-simd 路径。也就是说，即便纯 f32 在逐元素运算上
占便宜，在 clnn 最吃算力的 matmul 上也拿不到好处。

**实现：一个闸门，六个 `:around` 方法。**

```
用户数据 ──► [闸门 ensure-compute-dtype] ──► 层内部（恒 f64） ──► 输出 f64
                     ▲
        FORWARD / BACKWARD / COMPUTE-LOSS 的 :AROUND
```

| 位置 | 机制 | 说明 |
|---|---|---|
| 层的 `forward` / `backward` | `(defmethod forward :around ((l layer) input))` | 一处覆盖全部 20 个层；新增层自动被覆盖 |
| 损失 | `compute-loss` / `compute-loss-gradient` 的 `:around` | `predicted` 与 `target` 都过闸门；闸门本身按域分流（§5.7） |
| 初始化器 | `init-weight` / `init-bias` 的 `:around` | 一处覆盖全部 10 个初始化器 |
| 外部导入 | `normalize-params!` | 序列化存档、手工权重表的正规归一通道 |

**为什么 `target` 也过闸门却不会被误伤**：`target` 有两种身份 —— CE 的整数类别标签
（索引域）与 mse/bce/huber/cosine/kl 的数值回归目标（计算域）。`ensure-compute-dtype`
本身按域分流：整数与 `:bool` 原样放行，只折 `:float32`。所以「标签保持整数、
数值目标被规约」是同一个闸门的两种结果，不需要按损失类型分派。见 §5.7。

**代价，以及为什么不选另一条路。**

| 方案 | 代价 | 结论 |
|---|---|---|
| 多精度共存 | 几十处缓冲要人肉追踪；混跑慢 8.4×；缺口反复复发 | 不采用 |
| **单一计算域 + 入口规约** | 入口一次 astype（1600 万元素约 50 ms）；f32 数据源每步都要付 | **采纳** |
| 单一计算域 + 拒绝 f32（报错） | 数据管道必须自己转，读 CSV / 图像会频繁踩 | 过严；规约已经够用 |

**入口转换的成本是可以消掉的**：把 `vt-astype` 放到数据管道里做一次、之后复用
（`dataloader` 出 f64），闸门就永远走「已经是 f64 → 原样返回」的快路径，零分配。
首次遇到 f32 时框架会 `warn` 一次（每次会话一条，不刷屏），提示的正是这件事。

**这条决定不影响 clvt**：clvt 支持 11 种 dtype 是它的正确职责（它是 NumPy 的移植，
`np.zeros` 就是 float64，多 dtype 是规范要求）。clnn 的取舍只发生在自己这一层 ——
**能力在下面，纪律在上面**。

### 2.1 类型域划分

clvt 有 11 个逻辑 dtype。clnn 把它们划成**三个可用域 + 两个禁区**，每个域的
**允许位置**是硬约束：

| 域 | dtype | 允许出现的位置 | 禁止出现的位置 |
|---|---|---|---|
| **计算域** | `:float64`（**唯一**，由 `+COMPUTE-DTYPE+` 定义） | 参数、梯度、前向缓存、激活输出、损失值、优化器状态、**全部内部缓冲** | 索引、标签 |
| **索引域** | `:int8 :int16 :int32 :int64 :uint8 :uint16` | 索引（embedding / gather / scatter）、类别标签、`argmax` 结果 | 参数、梯度、缓存（会截断浮点结果） |
| **掩码域** | `:bool` | **仅** API 边界的条件/掩码：`tensor-where` 的 cond、`tensor-masked-fill` 的 mask、用户的比较结果 | autodiff 链内任何位置（参数/梯度/缓存/乘性掩码） |
| **C 复数** | `:complex128 :complex64` | **不支持**。仅允许作为「会被立即拒绝」的输入出现 | 全部 |
| **保留** | `:uint32 :uint64` | clvt 内部的保留符号，clnn 不产生也不接受 | 全部 |

**`float32` 不在任何域里** —— 它是「计算域的窄版本」，进入时被折成 `:float64`
（§2.0）。识别它的是 `ensure-compute-dtype`，判断「是否已是计算域」的是
`compute-dtype-p`；`float-dtype-p` 两者都认，只用于识别。

**为什么布尔不能进 autodiff 链**：`vt-*` 在「两个操作数都是 `:bool`」时会改派逻辑与
（`True*True=True`，`CONVENTIONS` §17.3）。dropout 的 `input * mask` 若两边都是布尔，
保留位的 `1/(1-p)` 会塌成 `1`，缩放静默消失；两个布尔相乘也不再是「数值 0」，而是
「假」。这类错误不会抛异常，只会让 loss 少降一点 —— 属于最难定位的一类（D-12/D-13）。

### 2.2 五条铁律

| 编号 | 铁律 | 现状 | 守护手段 |
|---|---|---|---|
| **D1** | `forward` 的输出 dtype **恒为** `+COMPUTE-DTYPE+`。入口的 f32 被规约后，层内部不再出现其它浮点 dtype | 由 `forward :around` 闸门保证；20 个层一处覆盖 | `dtype-contract-test` I1–I10 |
| **D2** | `backward` 返回的 dX dtype 恒为 `+COMPUTE-DTYPE+` | 同上（`backward :around`） | I3 |
| **D3** | 梯度槽的 dtype 恒等于参数的 dtype，且两者都在计算域 | 参数由 `init-weight :around` 规约；优化器状态用 `vt-zeros-like` 继承 | I4、D 组 |
| **D4** | 索引/标签必须是整数 dtype（:int8…:uint16） | 布尔索引被 clvt 当成 0/1 下标，**静默取错行**；浮点索引报 clvt 口径的错误 | `assert-index-dtype` 在 5 个入口拦截 |
| **D5** | 布尔张量只作条件，不作数值；需要数值先 `vt-astype :float64` | 布尔⊗布尔 = 逻辑与；布尔减法直接报错 | 文档 + §6 禁令 + 测试 B4/B5 |

提升表仍然在 clvt 里工作（`vt-+` 等算子的语义没变），只是 clnn 的闸门让参与提升的
操作数永远同 dtype —— 于是提升表在 clnn 的热路径上退化为恒等。

**D2/D3 的实现只有一条规则**：

> **所有内部缓冲直接 `(vt-zeros shape)`。** 缺省就是 `:float64`，就是计算域。

`%rnn-zeros` 保留但退化为 `(vt-zeros shape :dtype +compute-dtype+)`（保留形状参数
的签名只是为了不牵动 14 个调用点）；`value-float-dtype` 恒返回 `+COMPUTE-DTYPE+`。

### 2.3 各层的域归属与转换点

既然所有层的输出都是 f64，需要清楚的就不是「哪层会变宽」，而是
**每一层接受哪个域的输入** —— 这决定了你喂错东西时是「被规约」还是「被拒绝」。

| 层 | 输入域 | f32 输入的处理 |
|---|---|---|
| `dense` / `activation-layer` / `flatten` / `residual` / `dropout` / `global-avg-pool2d` | 计算域 | 闸门规约 → 正常计算 |
| `batch-norm` / `layer-norm` | 计算域 | 闸门规约（含 running 统计量，它们本就是 f64） |
| `conv2d` / `max-pool2d` / `avg-pool2d` | 计算域 | 闸门规约 → 正常计算 |
| `rnn-cell` / `lstm` / `gru` / `rnn-sequence` | 计算域 | 闸门规约 |
| `sdpa` / `mha` / `transformer-block` | 计算域（Q/K/V 三元 list 逐项规约） | 闸门规约 |
| `embedding` | **索引域**（整数） | 不规约；非整数会由 `assert-index-dtype` 拒绝 |
| 损失：`mse` / `bce` / `huber` / `cosine` / `kl` / `smooth-l1` | 计算域（两侧） | `predicted` 与数值型 `target` 都被规约 |
| 损失：`ce-loss` | `predicted` 计算域 + `target` 索引域 | `target` 是整数 → 原样透传；整数标签校验仍然生效 |

**唯一的「拒绝」来自索引域**：`embedding` 与 `ce-loss` 的标签必须是整数，浮点标签
会报错而不是被静默截断（§10.2）。这是刻意的 —— 「0.5 号类别」没有合理语义。

**内部内核仍然只认 f64**：`im2col` / `col2im` / 池化的底层循环用
`(SIMPLE-ARRAY DOUBLE-FLOAT (*))` 声明读数组，它们不是层方法、不经闸门，因此
`%ensure-float64-data` 的哨兵对它们仍然有效。用户正常使用碰不到这条路径 ——
**看到这个错误等于看到了框架的 bug**（某条路径绕过了闸门），不是「你需要 astype」。

### 2.4 布尔契约（B 域）

#### 2.4.1 布尔可以出现在哪里

| 位置 | 合法写法 | 说明 |
|---|---|---|
| 条件选择 | `(tensor-where (vt-> x 0) a b)` | `tensor-where` / `tensor-masked-fill` 内部用 `%mask-true-p` 判定「非零为真」，收 `:bool`/整数/浮点三种 |
| 取被选中元素（1-D） | `(vt-take x (first (vt-nonzero mask)))` | `vt-nonzero` 返回**逐轴索引的列表**（像 `torch.nonzero(as_tuple=true)`），1-D 掩码要取 `first`。不能直接 `(vt-take x mask)`（D4/G-5） |
| 取被选中元素（任意维，展平语义） | `(vt-extract mask x)` | 等价 `numpy.extract(condition, arr)`，按**展平顺序**返回被选中元素 |
| 计数 | `(vt-item (vt-sum mask))` | `vt-sum` 对 `:bool` 已给 `:int64`，不用先 astype |
| 断言/日志 | `(vt-to-list mask :bool t)` → `(T NIL T)` | 默认 `(vt-to-list mask)` 给 `(1 0 1)`（物理视图） |
| 单元素读取 | `(vt-item mask)` / `(vt-ref mask i)` → `T`/`NIL` | 布尔标量走逻辑视图（`CONVENTIONS` §17.8） |

#### 2.4.2 布尔**禁止**进入的位置

| 位置 | 后果 | 正确做法 |
|---|---|---|
| 参数 / 梯度槽 | `vt--` 报错；`vt-+` 变逻辑或 | 不要为掩码建参数 |
| 乘性掩码（`vt-* x mask` 两边都布尔） | 逻辑与，缩放丢失（§17.2 D-A2/D-C） | `(vt-astype mask :float64)` 或 `(vt-scale mask s)` |
| `dropout` / `sdpa` 的输入 | 掩码被传染成布尔（已修：掩码显式 `:dtype :float64`） | 不要把布尔张量喂进模型 |
| `embedding` / `gather` 的索引 | 静默取第 1 行（G-5，已修） | `(first (vt-nonzero mask))` 转整数索引 |
| `vt-concatenate` | clvt 报 `TYPE-ERROR`（`… is not of type REAL`，G-8） | 先 `vt-astype :float64` 再拼 |
| CE / one-hot 的标签 | 直接报 `T can't be converted to FIXNUM` 而不提「标签」（已改为明确文案） | 标签用 `:int64` |

#### 2.4.3 三种「布尔化」必须区分

clvt 对「变成布尔」有三条**语义不同**的路径，混用会得到不同结果：

| 路径 | 语义 | 例（`0.5` / `2` / `-0.0` / `NaN`） |
|---|---|---|
| `vt-astype … :bool` | **真值化**（`%bool-flag`） | `1 / 1 / 0 / 1` |
| 比较/逻辑/谓词内核 | 真值化 | `(vt-> 0.5 0)` → `1` |
| 整数截断（**错误用法**） | `truncate` | `0 / 2（越界） / 0 / 0` |

第三行不是「布尔化」，而是**把数值硬写进 0/1 存储**：`0.5` 被截断成 `0`（丢真值），
`2` 原样留在只有 1 字节的存储里（破掉「布尔只含 0/1」的不变量，之后任何一个归约都会
把它放大成怪数字）。clvt `CONVENTIONS` §3.4 与 §17.10 记录了这类缺陷的多个实例；在
clnn 侧的等价物是 **`vt-map` 的 dtype=nil 推断**：`(vt-map (lambda (x) 0.5) int64-array)`
得到 `(0 0)` 而不是 `(1 1)`（§17.2 D-A9）。因此凡是用 `vt-map` 生产浮点结果的地方都
必须显式给 `:dtype`（dropout / sdpa 的两处掩码就是这么修的）。

**可用性备注**：`vt->` / `vt-=` / `vt-where` / `vt-nonzero` / `vt-extract` 等条件类算子
**未导入** `nn` 包（`nn-package.lisp` 只导入 clnn 内部用到的 94 个符号，不是 clvt 的全部导出）。写掩码时需要
`clvt:` 前缀，或在自己的包里 `(import-from :clvt :vt-> :vt-= :vt-nonzero)`。这是刻意的：
`nn` 的公开面只保留建模所需的最小集，避免把 clvt 的 700+ 导出符号全带进来。

### 2.5 复数契约（C 域）

**结论：clnn 不支持复数。** 现状是「部分算子拒绝、部分算子静默接受」，见下表：

| 入口 | 现状 | 性质 |
|---|---|---|
| `vt-relu` / `vt-sigmoid` / `vt-leaky-relu` / `vt-tanh` … | **显式报错**（clvt §16.11，与 numpy 一致） | ✅ 确定性拒绝 |
| `vt-softmax` / `vt-log` / `vt-clip` / `vt-exp` | 静默接受并返回复数（numpy 亦如此） | ⚠️ 语义可疑但非 clnn 责任 |
| `dense` / `conv2d` 前向 | **静默**产出复数输出（`matmul` 提升） | ❌ 缺口 G-4 |
| `mse` / `bce` / `ce` 损失 | 静默产出**复数损失值** | ❌ 缺口 G-4 |
| `compute-grad-norm` / `global-grad-norm` | 返回**复数**，随后 `(> n clip)` 抛 `T is not of type REAL` | ❌ 缺口 G-4，错误点远离成因 |
| `vt-astype complex :float64` | **静默丢弃虚部**（`1+2i → 1.0`） | ⚠️ numpy 会发 `ComplexWarning`，clvt 不发 |

**契约（用户侧）**：不要在 clnn 里使用复数张量。需要复数信号处理时，把实部/虚部拆成
两个实数张量（两个通道或两个 batch 维）再进入框架 —— 这也是绝大多数 DL 框架的通行做法。

**契约（扩展者侧）**：给层加复数支持**不是**「换个 dtype」那么简单，至少要同时解决
①复数激活的定义（`relu` 在复数上没有公认定义）、②梯度范数要用 `|z|` 而不是 `z²`、
③ `vt-item` 返回复数导致所有 `(> (vt-item …) …)` 类型错。因此不开放复数。

### 2.6 计算域的逐层验收

**没有任何层需要「保住 f32」，因为入口就折掉了。** 下表是逐层验收结果；
标了断言编号的行由 `dtype-contract-test` 直接断言，标「结构同源」的行没有单列断言 ——
它们走的是同一个挂在 `layer` 基类上的 `:around`，但这一点是推断，不是实测。

| 层 | fwd | dX | 参数梯度 | 断言 |
|---|---|---|---|---|
| `dense`（11 种激活） | f64 | f64 | f64 | I4（`:tanh` 现场） |
| `activation-layer`（`:tanh`/`:sigmoid`/`:softmax`） | f64 | f64 | — | I5 |
| `batch-norm`（训练态 / 推理态） | f64 | f64 | f64 | I6a–I6d |
| `layer-norm` | f64 | f64 | f64 | 结构同源 |
| `conv2d` / `max-pool2d` / `avg-pool2d` | f64 | f64 | f64 | I10（f32 输入 → f64 输出） |
| `rnn-cell` | f64 | f64 | f64 | I7a |
| `lstm` / `gru` / `rnn-sequence` | f64 | f64 | f64（全槽） | I7 |
| `embedding` | f64 | f64（恒零） | f64 | I8a（索引仍是 int64，见 I8b） |
| `sdpa` / `mha` / `transformer-block` | f64 | f64 | f64 | 结构同源（Q/K/V 三元 list 逐项规约） |
| `dropout` / `flatten` / `residual` / `global-avg-pool2d` | f64 | f64 | — | C 组（掩码不是 f32） |
| `ce` / `mse` / `huber` / `smooth-l1` / `cosine` | f64 | f64 | — | I9 |

**「f64」在这里是唯一答案，不是「本层恰好保住了」**。一旦某层返回别的 dtype，
D1 被违反，`dtype-contract-test` 立刻红。

#### 2.6.1 接入 f32 数据源的正确姿势

f32 数据（CSV、图像、别的框架导出的 npy）不需要特殊处理 —— 直接喂给模型即可，
闸门会转。但**每步都转就是每步都付一次全量转换**（1600 万元素约 50 ms）。
正规做法是在数据管道里一次性规约：

```lisp
;; 反例：每个 batch 都是 f32，每步都要转换一次
(loop for batch in (load-f32-batches)          ; (2 3) f32
      do (train-step model batch labels))

;; 正例：在数据管道出口转一次，之后复用 f64 张量
(defun prepare-batch (raw)
  (let ((x (vt-astype (batch-images raw) :float64)))
    x))                                        ; 缓存或预生成整个数据集

;; 或者，如果模型是从别处加载的（存档 / 手工权重表）：
(load-model "checkpoint.sexp")                 ; 内部已调 normalize-params!
;; 或对已有模型：
(normalize-params! model)                      ; 返回归一的参数个数
```

**首遇 f32 时框架会 `warn` 一次**（每次会话一条），文案里就写着这条建议。它不会
每步都刷 —— 那是噪音，不是信息。

#### 2.6.2 为什么没有「全 f32 模型的构造法」

**给参数一个 dtype 选项等于承诺它可选**，而这条路的代价（§2.0 的架构成本 + 8.4×
性能）由使用者承担、且不会立刻显现。所以初始化器不接受 `:dtype`，层也没有 dtype 槽。

如果你确实需要单精度训练，唯一正确的做法是改一个地方：

```lisp
(setf nn::+compute-dtype+ :float32)
```

—— 但**不要在生产里这么做**。它是全局的：改完之后 `conv2d` / `max-pool2d` /
`avg-pool2d` 会立刻在 `%ensure-float64-data` 处报错（那个哨兵**硬编码**
`:float64`，刻意不跟常量走 —— 它守的是内核的物理能力，不是框架的策略）。
真要支持 f32 计算域，需要做的是：把池化与卷积内核改成按 dtype 分派、把 §2.6
的验收表逐行重跑、并重新评估手写求导下的缓冲一致性。那是一次独立工程，
不是改个常量。

### 2.7 与 clvt 约定的对应关系

| clnn 契约 | 依据的 clvt 章节 | 关系 |
|---|---|---|
| F 域缺省 float64 | §3.1/§3.2 dtype 系统与提升表 | 直接继承 |
| 整数插槽只收整数 | §3.1 逻辑 dtype | 直接继承（并收窄：布尔虽可当整数，clnn 拒绝） |
| 布尔真值化 | §17.1/§17.3 | 直接继承 |
| 布尔不参与算术 | §17.3 | 直接继承（clnn 侧追加「不进 autodiff」） |
| 布尔索引拒绝 | §17.10.7（numpy 也不支持 `a[mask]`） | **收窄**：clvt 允许 `vt-take` 收 `:bool`，clnn 拒绝 |
| 复数拒绝 | §16.11 | 部分继承（激活拒绝、损失未拒） |
| nn 函数必须走浮点 | §3.4 | 直接继承（clnn 的 `vt-map` 调用点必须显式 `:dtype`） |
| CE 两入口不可混用 | §3.5 | **不适用**：clnn 自带 CE，不调用 `vt-cross-entropy*`（§12.4） |

### 2.8 契约缺口清单

状态含义：**已修** = 有代码修复与对应测试；**已记录** = 行为已写进契约但不打算改；
**待修** = 已知违背设计意图，留待后续。

| 编号 | 现象 | 根因 | 证据 | 状态 |
|---|---|---|---|---|
| **G-1** | `dropout`/`sdpa` 的掩码 dtype 跟随输入：整数输入把 `1/(1-p)` 截断，布尔输入让掩码变布尔 | `vt-map` 未给 `:dtype`，走输入提升 | §17.2 D-C1…C5 | 已修 |
| **G-2** | float32 模型训练一步后参数变 float64 | 优化器状态 `(vt-zeros (vt-shape param))` 缺省 f64 | §17.2 D-D1 | 已修（改用 `vt-zeros-like`） |
| **G-3** | 零梯度/零 dX 的 dtype 恒为 f64 | `(vt-zeros shape)` 缺省 f64 | 静态审计 + `manual-backward-contract-test` | 已关闭：计算域就是 f64，这不再是缺陷 |
| **G-4** | 复数输入在 `dense`/`conv`/损失/梯度范数上**静默**流通，最终在 `(> n clip)` 处类型错 | 无复数守卫 | §17.2 D-E1…E5、H3 | 待修（契约：不支持复数，见 §2.5） |
| **G-5** | 布尔索引被 `vt-take` 当作 1 → `embedding`/`gather` 静默取第 1 行 | clvt 索引校验放行 `:bool` | §17.2 D-G4/G6 | 已修（`assert-index-dtype`） |
| **G-6** | CE 标签 dtype 错误时文案指向 `FIXNUM` 而非「标签」 | 校验位置在 `coerce` | §17.2 D-G1/G2 | 已修 |
| **G-7** | `ce-loss` 恒以 float64 计算 | `%ce-target` 固定 double-float 数组 | I9a/I9b | 已关闭：f64 就是计算域 |
| **G-8** | `vt-concatenate` 拒绝布尔张量 | clvt 实现按 `REAL` 累加 | §17.2 D-F2 | 已记录（clnn 不拼布尔；如需请 `vt-astype`） |
| **G-9** | 除 dropout/sdpa 外的 31 处 `vt-map` 未显式给 `:dtype`：浮点输入下正确，整数输入会截断 | 同上（`vt-map` 的 dtype 推断） | `grep vt-map nn-*.lisp` | 已记录（契约：这些函数只接受浮点输入，见 §6.5） |
| **G-10** | clnn 自定义的 12 个 `nn::vt-*` 与 clvt 命名空间**同形前缀**，易被误认为 clvt API | 命名历史（跟随 clvt 风格） | §17.4 符号审计 | 已记录（不重命名，避免破坏外部代码） |
| **G-11** | f32 模型前向 f32、梯度 f64，一步后参数被拉回 f64 | ① clvt `vt--` 把首操作数无条件 `ensure-vt`，绕过弱标量提升；② clnn 的 RNN/embedding 缓冲从 `(vt-zeros (list …))` 起 | `dtype-contract-test` A11/I 组 | 已关闭：上游 `§16.31.7` 补齐弱标量；且 clnn 不再有多精度，「f32 能否全链保住」这个问法被废除 |
| **G-12** | f32 与 f64 混在同一张图里慢 8.4×（7165 ms vs 852 ms） | 每个算子按提升表定结果 dtype、分配 f64 临时数组、逐元素转换 | §2.0 实测表 | 已关闭：单一计算域 + 入口规约，内部永不再转换 |

---

## 3. 训练循环契约（核心）

### 3.1 标准循环：步骤与顺序

唯一受支持的训练循环形态如下（与 `examples/mnist.lisp` 的 `train-dense-mnist` 完全一致）：

```lisp
(dotimes (step n)
  (zero-grad! model)                                   ; ① 显式清零（梯度槽置 NIL）
  (let* ((pred (forward model x))                      ; ② 前向
         (loss (compute-loss loss-fn pred y))          ; ③ 损失值
         (grad (compute-loss-gradient loss-fn pred y))) ; ④ dL/dpred
    (backward model grad)                              ; ⑤ 反向（覆盖写各层 dw/db）
    (optimizer-step opt (params model) (grads model))  ; ⑥ 参数更新
    (clear-step-caches! model)))                       ; ⑦ 清理本步前向缓存
```

**顺序契约（不可交换）**：

1. `backward` 只能读「**最近一次 forward**」写入的缓存。两次 forward 之间不插 backward，
   第一次的缓存即作废。
2. `optimizer-step` 只消费「**最近一次 backward**」写入的 `dw/db`。
3. `forward → backward` 必须在**同一训练模式**下完成；中途切换 `training-p`（影响
   dropout / batch-norm）会让缓存与前向不一致。
4. ③④ 是本框架与 autograd 框架的关键差异：**损失与模型解耦**，`compute-loss-gradient`
   产出的 `dL/dpred` 就是 `backward` 的入口张量，不存在「从 loss 节点反向」这回事。

### 3.2 是否每步自动清零？——不是；清零语义由「覆盖写」隐含完成

这是本框架与 PyTorch 最重要的差异，必须精确理解：

1. **没有全局自动清零机制**。训练循环必须自己调用 `(zero-grad! model)`。
2. **覆盖式层的 backward 本身就是「清零 + 写入」**：`dense/conv2d/batch-norm/
   layer-norm/mha` 的 backward 用 `setf` 把 `dw/db` 槽整体替换为本次算出的**全新张量**。
3. `zero-grad!` 的实际动作是把梯度槽**置 `NIL`**（不是置零张量），并递归到所有子层。

因此「忘记 `zero-grad!`」的后果分层讨论：

| 层族 | 忘记 zero-grad! 的后果 |
|---|---|
| **全部有参数层**（dense/conv/norm/mha/rnn 系列/embedding） | **数值结果不变**：backward 用步内局部累加器算完，再 `setf` 覆盖整槽 |
| 冻结层（`trainable = nil`） | 旧梯度若滞留，会被 `optimizer-step` 重新施加；`zero-grad!` 对冻结层同样清零 |

BPTT 的累加发生在**一次 backward 之内**（局部累加器 `dwih-acc`），不跨调用。
这也意味着 **「不清零」无法用来实现梯度累积**（§7.1）。

**契约结论**：始终显式调用 `zero-grad!`，把它当作每个训练步的第一条语句。

### 3.3 覆盖写发生在哪里

| 位置 | 动作 | 语义 |
|---|---|---|
| `dense` backward | `(setf (dense-dw l) dw)` `(setf (dense-db l) db)` | 覆盖 |
| `conv2d` backward | `(setf (conv-dw l) dw)` `(setf (conv-db l) db)` | 覆盖 |
| `batch-norm` backward | `(setf (bn-dgamma l) …)` `(setf (bn-dbeta l) …)` | 覆盖 |
| `layer-norm` backward | `(setf (ln-dgamma l) …)` `(setf (ln-dbeta l) …)` | 覆盖 |
| `mha` backward | `(setf (mha-dwq l) …)`（4 或 8 个槽，随 `:use-bias`） | 覆盖 |
| `rnn-cell` backward | 步内写局部累加器，末尾 `(setf (rnn-dwih l) acc)` | **覆盖**（累加只在步内） |
| `lstm` / `gru` backward | 同上，`dwih-acc` / `dwhh-acc` / `dbih-acc` 三个局部累加器，末尾逐个 `setf` | **覆盖**（累加只在步内） |
| `embedding` backward | 步内 `(vt-zeros …)` 起步，命中行 `+=`，末尾 `setf` | **覆盖**（scatter-add 在步内，§5.2） |
| `optimizer-step` | `(funcall setter new-p)` | 参数被**新张量替换** |

### 3.4 变体循环

- **全局梯度裁剪**：`(clipped-gradient-update! model optimizer max-norm)` = 全模型梯度
  全局 L2 范数 → 超阈值则 `scale-all-grads!` 等比缩放 → `model-update!`。它**只缩放
  梯度、不清零**，且必须只调用一次（重复调用会重复缩放）。
- **模型级封装**：`model-forward` / `model-backward` / `model-update!` 是薄封装；
  `model-update!` 等价于 `(optimizer-step opt (params model) (grads model))`。
- **学习率调度**：`nn-scheduler.lisp` 在每个 step/epoch 后改 `optimizer-lr`，与循环正交。
- **推理**：只 `forward`，不 `backward`、不 `optimizer-step`；建议每 batch 后
  `clear-step-caches!` 释放缓存。

### 3.5 优化步的边界：一次 `update!` = 一步

`optimizer-step-count` 记的是**执行了多少个参数更新步**，不是 `optimizer-step` 被调了
多少次。这个区别有实际后果：Adam 的偏差校正 `1/(1−β^t)` 与 Adagrad 的 `1/(1+λt)` 直接
吃 `t`；梯度裁剪也按「一次调用传入的梯度表」算范数。

```lisp
;; A：整模型一次 —— 1 步
(update! model opt)

;; B：逐层 —— 2 步（这是 2 次独立更新，不是 1 次更新的两个分片）
(update! layer1 opt)
(update! layer2 opt)

;; C：分片但算一步 —— 1 步
(with-optimizer-step (opt)
  (update! layer1 opt)
  (update! layer2 opt))
```

| 入口 | 推进步数 | 说明 |
|---|---|---|
| `update!` / `model-update!` | 1 | 二者等价，`model-update!` 是兼容别名 |
| `with-optimizer-step` 内的 `update!` | 0（外层已授予） | 同一优化器嵌套时不重复推进 |
| 直接调 `optimizer-step` | 1 | 不在步作用域内时自行推进（兼容旧用法） |

**什么时候用 C**：显存受限要分片更新，或一个优化器同时管多个子网络（多 head、GAN 的
双网络）。**A 与 B 都对，只是语义不同**；过去两种写法都让 `step-count += N`，于是 B
的偏差校正算错而没有任何提示。

>C 的注意点：梯度裁剪仍按每次 `optimizer-step` 传入的梯度表算范数，所以分片时每片
>各自裁剪。需要真正的全局范数，用 `clipped-gradient-update!`（它先对整个模型缩放一次）。

---

## 4. 层协议（Layer Protocol）

### 4.1 基类与必选方法

所有层派生自 `layer`（槽位：`name`、`trainable`、`training`）。一个可训练层需要提供：

| 方法 | 义务 |
|---|---|
| `forward (l input)` | 输入张量（或张量列表），返回输出张量；**写入本步缓存**；输出 dtype 遵守 D1 |
| `backward (l grad-output)` | 输入 = 最近 forward 输出同形状的梯度；**写入本层梯度槽**；返回 dX（多输入/有状态层用 `values`）；dX dtype 遵守 D2 |
| `params (l)` | `((owner-layer "name" 参数张量 setter) …)`，与 `grads` **严格同构**（§4.4） |
| `grads (l)` | `(("name" . 梯度张量-or-NIL) …)`，顺序与 `params` 一一对应 |
| `grad-slots (l)` | 参与清零的梯度槽名列表（`zero-grad!` 用） |
| `cache-slots (l)` | 保存前向缓存的槽名列表（`clear-step-caches!` 用） |
| `set-training! (l mode)` | 容器层需 `call-next-method` 后递归子层 |

**实测的槽位声明**（§17.1 T-15，逐一导出确认，新增层可对照）：

| 层 | `cache-slots` | `grad-slots` |
|---|---|---|
| `dense` | `(input-cache z-cache a-cache)` | `(db dw)` |
| `activation-layer` | `(z-cache a-cache)` | `()` |
| `flatten` | `(cache)` | `()` |
| `dropout` | `(mask-cache)` | `()` |
| `batch-norm` | `(input-cache xhat-cache std-inv-cache)` | `(dgamma dbeta)` |
| `layer-norm` | `(input-cache xhat-cache std-inv-cache)` | `(dgamma dbeta)` |
| `conv2d` | `(input-cache col-cache output-shape-cache)` | `(db dw)` |
| `max-pool2d` | `(cache)` | `()` |
| `avg-pool2d` | `(input-cache count-cache)` | `()` |
| `global-avg-pool2d` | `(cache-shape)` | `()` |
| `embedding` | `(indices-cache norm-scale)` | `(dw)` |
| `rnn-cell` | `(input-cache h-prev-cache h-cache)` | `(dwih dwhh dbih)` |
| `lstm` / `gru` | `(cache)` | `(dweight-ih dweight-hh dbias-ih dbias-hh)` |
| `sdpa` | `(cache)` | `()` |
| `mha` | `(cache)` | `(dw-q dw-k dw-v dw-o db-q db-k db-v db-o)` |
| `transformer-block` / `residual` / `stop-gradient-node` | `()` | `()`（容器递归子层） |

### 4.2 forward 契约

- 输入输出形状见 §10 的形状总表；核心约束是 **deterministic、缓存齐全**：backward 需要
  的中间量（z、a、输入、掩码、形状）必须在 forward 时全部写入缓存槽。
- **dtype 义务（D1）**：输出 dtype 必须等于输入 dtype。唯一例外是 `embedding`
  （整数索引 → 浮点向量）。
- 无参数层（activation/flatten/dropout/池化/residual/stop-gradient）`trainable` 默认
  `nil`，`params` 返回空。
- 延迟初始化：`dense/conv2d/embedding/batch-norm` 允许参数槽为 `NIL`，在**首次
  forward** 时用 `nn-initializer` 初始化（dense 默认 He-normal，bias 默认零）。
  `(build-model model &optional dummy-input)` 提供显式初始化入口。

### 4.3 backward 契约

- **输入**：`grad-output`，形状必须等于最近一次 forward 的**输出**形状，dtype 亦同。
- **返回**：dX，形状必须等于最近一次 forward 的**输入**形状（rank、各维完全一致），
  dtype 必须等于输入 dtype（D2；零张量用 `vt-zeros-like`）。
- **多输入/多输出层**的返回约定：
  - `rnn-cell`：`(values dx dh-prev)`；
  - `lstm`：`(values grad-input dh dc)`（`h-0/c-0` 未给出时 dh/dc 为 `NIL`）；
  - `gru`：`(values grad-input dh)`；
  - `multi-head-attention`：forward 输入是 `(list Q K V)`，backward 返回 `(values dQ dK dV)`；
  - `sdpa`：同 mha；
  - 其余单输入单输出层返回单值；`stop-gradient-node` 返回与输入同形的**全零张量**
    （`grad` 为 `NIL` 时返回 `NIL`）。
- backward 内部不得依赖 forward 之外的任何全局状态（`*training-mode*` 除外）。
- **容器只取第一个返回值**：`sequential` 的 backward 写的是 `(setf current (backward
  layer current))`，所以多值层必须先用 `rnn-sequence` 包起来，否则 dh/dc 被丢弃且不报错。

### 4.4 params/grads 严格同构契约

`optimizer-step` 用 `(loop for p in param-list for g in grad-list)` **按位置配对**。因此：

> **`params` 与 `grads` 必须长度相等、顺序一致、名称一致。**

- 参数存在但梯度尚未计算时，`grads` 必须返回 `(名称 . NIL)` 占位，**不能省略该项**。
- 长度不一致时 `assert-param-grad-alignment` 显式 `error`；名称错位由
  `assert-layer-name-alignment` 抓出。
- `mha` 的 `params`/`grads` 项数随 `:use-bias` 为 4 或 8，两侧必须**同时**条件化
  （源码已如此，自定义层不要只条件化一侧）。
- `NIL` 梯度 + `(layer-trainable-p owner)` 为假 + 参数为 `NIL`，三者任一成立时该参数在
  `optimizer-step` 中被**跳过**（不更新、不报错）。

### 4.5 缓存槽语义（z / a / a-prev 及各层变体）

| 层 | 缓存槽 | 语义 |
|---|---|---|
| `dense` | `input-cache` | `(cons 原始形状 x-flat)`：反向还原多维 dX 用 |
| | `z-cache` | 线性输出 z = x·W（**激活前**） |
| | `a-cache` | 激活输出 a = f(z)（**激活后**） |
| | （a-prev） | **没有独立槽位**。"a-prev" 即 `input-cache` 的 cdr（本层输入的展平视图） |
| `activation-layer` | `z-cache` / `a-cache` | 按 kind 只存需要的那一个（§9） |
| `flatten` | `cache` | forward 输入的原始形状 |
| `conv2d` | `input-cache` / `col-cache` / `output-shape-cache` | 原输入 (B,C,H,W)、im2col 矩阵、输出形状 |
| `max-pool2d` | `cache`（单槽） | 反向按最大值位置路由梯度 |
| `avg-pool2d` | `input-cache` / `count-cache` | 计数矩阵做平均 |
| `global-avg-pool2d` | `cache-shape` | 输入形状 |
| `dropout` | `mask-cache` | 训练态缩放掩码（**浮点 dtype**）；推理态置 `NIL` |
| `batch-norm` | `input-cache` / `xhat-cache` / `std-inv-cache` | 另有 `batch-size`、`fwd-mode` **不属于** cache-slots；`running-mean/var` 是持久统计量，**永不清理** |
| `layer-norm` | `input-cache` / `xhat-cache` / `std-inv-cache` | |
| `embedding` | `indices-cache` / `norm-scale` | `norm-scale` 为 max-norm 的逐行系数，关闭时为 `NIL` |
| `rnn-cell` | `input-cache` / `h-prev-cache` / `h-cache` | 另有 `state`（跨 forward 隐状态，`reset-rnn-cell-state!` 清） |
| `lstm` / `gru` | `cache` | plist：`:gates :hs :x :c …` 全部时间步 |
| `rnn-sequence` | `x-cache` / `h-prev-cache` / `h-cache` | BPTT 所需的逐步张量列表 |
| `sdpa` / `mha` | `cache` | attn-raw、dropout mask、q/k/v 及 2D 展开等 |

**缓存契约**：

1. backward 只信任「最近一次 forward」的缓存；缓存槽一律**覆盖写**。
2. `clear-forward-cache!` 只清 `transient-cache-slots`（全部层默认 `()`，即空操作），
   任意时刻调用安全。
3. `clear-step-caches!` 清 `cache-slots` 全部槽位（置 `NIL`），**只能在
   forward+backward 完成一个训练步后调用**；容器层递归清子层。
4. 推理循环每 batch 后调 `clear-step-caches!` 防内存累积。

### 4.6 容器契约（sequential）

- `sequential` 假设每个子层满足**单输入单输出、单梯度进单梯度出**；
  `rnn-cell`（双值）、`multi-head-attention`（三输入）、裸 `lstm/gru`（多值）**不得**
  直接放入，需用 `rnn-sequence` 包装或自定义容器。
- `params`/`grads` 按子层顺序 `nconc` 聚合；`zero-grad!`/`clear-step-caches!`/
  `set-training!` 经协议递归。
- 同一层对象**加入两个容器**即构成参数共享，默认不支持（§7.2）。

---

## 5. 梯度语义

### 5.1 梯度槽是「本次反向传播计算出的梯度」，不是累计梯度

```lisp
;; 反向主体（简化）
(d-act  (vt-* grad-flat (激活导数(z-cache 或 a-cache))))
(dw     (vt-matmul (vt-transpose a-prev) d-act))   ; (in-dim, out-dim)
(db     (vt-sum d-act :axis 0))                    ; (out-dim,)
(setf (dense-dw l) dw)                             ; 覆盖写
(setf (dense-db l) db)
(dx-flat (vt-matmul d-act (vt-transpose w)))       ; 返回 dX
```

**含义**：

1. `dw`/`db` 每次 backward **从零计算、整体替换**。连续两次 backward，第二次的值与
   第一次**无关**（§17.1 T1：与「第二轮单独计算」逐位相等，maxdiff = 0.0）。
2. **覆盖写隐含清零**：`setf` 同时完成「丢弃旧值」与「写入新值」。
3. `dw` 形状恒等于 `weights`（dense 为 `(in-dim, out-dim)`，与 PyTorch 的 `(out, in)`
   **相反**），dtype 恒等于参数 dtype（D3）。

**这条契约覆盖全部层族**，包括 BPTT 的 `lstm`/`gru`/`embedding`：
它们内部的「跨时间步累加」/「同一行多次命中累加」都发生在**一次 backward 内部**，
用局部累加器完成，最后一次性 `setf` 到梯度槽。`rnn-cell` 被 `rnn-sequence` 复用时
需要跨**调用**累加，但那由 `rnn-sequence` 在自己的 `backward` 里绑定
`*rnn-accumulate-grads*` 完成，并在循环前清空 cell 的梯度槽 —— 对外仍是覆盖写。

**为什么必须统一**（§1.5 的 M4）：手写架构里没有 autograd 来管理梯度生命周期，
如果同一张网络里 dense 覆盖写、lstm 累加，那么「漏掉 `zero-grad!`」会让一半层正确、
另一半翻倍。这种「部分正确」比全错更难查 —— 观测到的只是 loss 曲线比预期差一点。

### 5.2 各层族梯度写法总表

| 层族 | backward 写法 | 跨调用 | 步内累加方式 |
|---|---|---|---|
| dense / conv2d / batch-norm / layer-norm / mha / sdpa / rnn-cell（独立用） | `setf` 新张量 | 覆盖 | — |
| lstm / gru | `(setf 槽 acc)`，`acc` 是局部累加器 | 覆盖 | 局部 `dwih-acc` 沿 T 步 `vt-+` |
| rnn-sequence | 循环内驱动 cell，循环前清空 cell 的槽、绑 `*rnn-accumulate-grads*` | 覆盖 | cell 的槽跨时间步 `vt-+` |
| embedding | `(vt-fill dw 0)` 后逐行 `+=` | 覆盖 | 同一行被多次命中时相加（scatter-add） |
| 无参数层 | 无梯度槽 | — | 直接回传 dX |

**一致性保证**：全部层族对外的语义都是「本次 backward 的梯度」，与上一次无关。
因此 `zero-grad!` **不是正确性的前提**，而只是「显式丢弃上一次结果」的手段
（§5.4 说明优化器不清零）。守护测试：`manual-backward-contract-test` 的 M4 组
对 10 个层族逐个验证「两次 forward+backward 的梯度逐位相同」。

### 5.3 NIL 梯度语义

- 槽位初始为 `NIL`；`zero-grad!` 后为 `NIL`；backward 后为 `vt`。
- `NIL` 表示「本步没有为该参数计算梯度」，`optimizer-step` 对其**静默跳过**（§17.1 T3）。
- 推论：只 forward 不 backward 的层（推理分支、冻结层）参数不会被更新；自定义层若
  backward 未写梯度槽，优化器也不会动它 —— 这既是宽容也是陷阱，所以 `grads` 必须与
  `params` 同构（§4.4）。

### 5.4 优化器不清零梯度

`optimizer-step` 消费 `grads` 返回的梯度后**不回写、不清零**：backward 之后梯度槽保持
原值（§17.1 T7：步进后 `(dense-dw d)` 仍是旧对象）。清零完全由训练循环下一步的
`zero-grad!` 负责。PyTorch 用户特别注意：这里没有 `optimizer.zero_grad()` 的等价职责。

### 5.5 冻结层

`trainable = nil` 的层：`optimizer-step` 跳过其参数；`zero-grad!` **仍然清零**其梯度槽
（防陈旧梯度复活）。`param-count :trainable-only t` 可统计可训练参数量。

### 5.6 梯度范数与 dtype

`compute-grad-norm` / `global-grad-norm` 均用 `Σ g²` 后开方。对实数没问题；对复数会得到
**复数范数**（§17.2 D-E4），随后任何 `(> n clip)` 都抛类型错 —— 这是 G-4 的一部分。
若将来支持复数，须改成 `Σ |g|²`（即 `vt-abs` 后平方），`nn-protocol.lisp` 的
`regularizer-penalty`（l1）已经用的是 `vt-abs`，可作参照。

### 5.7 损失 target 属于哪个域

`compute-loss` / `compute-loss-gradient` 的**两个**参数都过闸门（§2.0），但它们的身份不同：

| 损失 | `target` 的身份 | 域 | 闸门行为 |
|---|---|---|---|
| `ce-loss` | 整数类别标签 `(B,)` | 索引域 | 原样透传；整数校验仍然生效（§10.2） |
| `mse` / `bce` / `huber` / `smooth-l1` / `cosine` / `kl` | 与 `predicted` 同形的数值目标 | 计算域 | `:float32` → `:float64` |
| 任意损失，传 `:bool` 目标 | 掩码 / 二值标签 | 掩码域 | 原样透传（不折成 0.0/1.0） |

这是一个真实的缺口（已修）：闸门只规约 `predicted`、跳过 `target`，于是 `mse` 在 f32 目标下
每个算子都跑一次 f32⊔f64 提升表 —— 实测损失 160 → 765 ms（**4.8×**）、梯度
117 → 774 ms（**6.6×**）。数值仍然正确，所以不会有任何报错。

**为什么不需要按损失类型分派**：`ensure-compute-dtype` 本身按域判断 —— 整数与 `:bool`
原样返回，只折 `:float32`。所以「标签保持整数、数值目标被规约」是同一个闸门的两种输出。

---

## 6. vt-* 调用面契约（接口契约）

> 本章回答：**clnn 如何使用 clvt**。它既是给扩展者的「可用/禁用」清单，也是
> clvt 升级时的检查表。全部结论来自 §17 的实测与 §17.4 的符号审计。

### 6.1 总则

1. **只用导出符号**（§1.4）；`%` 开头的一律禁止。
2. **dtype 由 clnn 显式钉住**：凡需要 float32 保持、或需要浮点结果的地方，必须显式
   `:dtype`；不得依赖「输入提升」的缺省行为（§2.4.3）。
3. **零张量用 `vt-zeros-like`**，不用 `(vt-zeros (vt-shape x))`（§2.2 D2/D3）。
4. **索引/标签先校验**：`assert-index-dtype` 是唯一入口守卫（§2.2 D4）。
5. **不假设返回类型**：比较/逻辑/谓词一律返回 `:bool`（不再是 `:int8`）。

### 6.2 构造类

| 函数 | clvt 签名要点 | clnn 侧契约 |
|---|---|---|
| `vt-zeros` / `vt-ones` / `vt-const` | `(shape &key (dtype :float64))` / `(shape value &key dtype)` | **缺省 float64**。造参数须显式给 dtype；内部零张量改 `vt-zeros-like` |
| `vt-zeros-like` / `vt-ones-like` | `(vt &key dtype)` | 继承 dtype 与 shape，**内部首选** |
| `vt-random-normal` / `vt-random-uniform` / `vt-random` | `(shape &key mean std dtype rng)` 等 | 缺省 float64；初始化器必须能产出 f32（显式 dtype） |
| `vt-from-sequence` | `(contents &key dtype fast)` | 索引/标签必须 `:int64`（不要靠推断）；`:bool` 模式下 `NIL` 是假而非空表 |
| `vt-from-array` | `(arr &key dtype fast)` | `:fast t` 表示不拷贝（调用方此后不得改原数组） |
| `vt-arange` / `vt-eye` | `(total-num &key start step dtype)` / `(rows &key cols k value dtype)` | 缺省 f64；索引场景显式 `:int64` |

### 6.3 形状 / 视图类

| 函数 | 契约 |
|---|---|
| `vt-shape` / `vt-strides` / `vt-offset` / `vt-size` | 纯查询，无副作用 |
| `vt-reshape` | **不改变元素顺序、不改变 dtype**；元素总数必须一致，否则报错 |
| `vt-flatten` | 等价 `reshape (-1)`；返回**视图或拷贝不确定**，故不可依赖别名 |
| `vt-squeeze` | 仅移除长度为 1 的轴 |
| `vt-transpose` | 只交换轴，不改数据；`(vt-transpose m)` 反转全部轴，`(vt-transpose m perm)` 按 perm |
| `vt-broadcast-to` | 广播视图；stride 0 合法。**不要对广播视图做 `setf vt-ref`**（写会命中同一元素多次） |
| `vt-contiguous` / `vt-contiguous-p` | 连续性检查；GEMM/AVX512 快路径的入口条件 |
| `vt-slice` | `(vt &rest specs)`，spec 为 `(list :all)` / `(list i)` / `(list lo hi)`。**支持 `(setf (vt-slice …) v)` 原地写**（rnn 用它填 grad-input），写入会改基数组 |
| `vt-split` | `(tensor indices-or-sections &key (axis 0))` |
| `vt-concatenate` | `(axis &rest tensors)` —— **axis 是第一个参数**，不是关键字（易错）。布尔张量被拒绝（G-8） |

### 6.4 索引类

| 函数 | 契约 |
|---|---|
| `vt-ref` | `(vt &rest indices)`。整数张量返回数；**`:bool` 返回 `T`/`NIL`**（逻辑视图）；复数返回复数 |
| `(setf vt-ref)` | 写入前做 `coerce` 到元素类型；向 `:bool` 写 `T/NIL` 由 clvt 归一为 0/1 |
| `vt-take` | `(tensor indices &key axis)`。**索引必须整数**：clnn 侧先 `assert-index-dtype`（G-5）；`clvt` 本身会放行 `:bool` 并当成 0/1 |
| `vt-put` | `(tensor indices values &key mode)`；`mode ∈ {:raise :wrap :clip}`；布尔值 `T/NIL` 在 clvt ≥ 8003a88 可识别 |
| `vt-nonzero` | `(condition)` → **逐轴索引张量的列表**（`torch.nonzero(as_tuple=true)` 语义）。1-D 掩码取 `first` 即为整数索引向量；N 维要按轴分别 `vt-ref` 组合。展平语义请用 `vt-extract` |

**clnn 的用法**：`embedding`（`vt-take :axis 0`）、`tensor-gather`（`vt-take`）、
`tensor-scatter`（`vt-put`）、CE 标签（`vt-ref` 逐元素）。

### 6.5 算术类

| 函数 | 契约 / 陷阱 |
|---|---|
| `vt-+` / `vt-*` / `vt-/` | 逐元素，遵守 clvt 提升表。**bool⊗bool = 逻辑与、bool+bool = 逻辑或**（§2.4）。标量留在参数位时走 numpy 式弱提升：`(vt-* 2.0d0 x-f32)` 仍是 f32 |
| `vt--` | 同上，遵守 clvt 提升表：`(vt-- 1.0d0 x-f32)` 与 `(vt-- x-f32 1.0d0)` **都**是 f32（标量在任一位置都走弱提升）。`bool-bool` 报错（§2.4）。早期 clvt 把首操作数无条件 `ensure-vt`，导致两种写法不对称（G-11 的根因），`§16.31.7` 已修 —— 因此不需要 `(vt-+ 1.0d0 (vt-* -1.0d0 x))` 这类绕法 |
| `vt-scale` | `(a b &key out dtype)`，等价 `a * b` 但走专用快路径。**bool × 浮点标量会提升为 float64**（不是逻辑运算），故 bool 掩码可以用 `vt-scale` 乘 |
| `vt-square` / `vt-abs` / `vt-sqrt` / `vt-exp` / `vt-log` / `vt-expt` | 数学函数。`vt-map` 家族在整数输入上会截断，这些函数内部已做浮点提升（clvt §3.4）；但**不要自己用 `vt-map` 重造这些函数** |
| `vt-clip` | `(vt &optional min max &key out dtype)`；bce 用它做 `[eps,1-eps]` 裁剪 |
| `vt-astype` | 显式转换。`:bool` 走**真值化**；`complex → float64` **静默丢虚部**（§2.5） |
| `vt-map` | `(fn &rest args)`，**dtype 缺省 = 输入提升**。生产浮点结果时必须给 `:dtype`（否则整数输入截断，§2.4.3）；写回 `:bool` 会真值化 |
| `vt-reduce` | `(tensor axis init-val reducer-fn &key out dtype keepdims)`；clnn 目前未使用（导入但未调用） |
| `vt-do-each` | 遍历宏；clnn 目前未使用（导入但未调用） |

### 6.6 归约类

| 函数 | 契约 |
|---|---|
| `vt-sum` / `vt-mean` | `(tensor &key axis keepdims dtype out)`。`sum` 对 `:bool` 给 `:int64`；`mean` 对 `:bool` 给 `:float64`；f32 保持 f32 |
| `vt-var` / `vt-std` | 带 `:ddof`（缺省 0，与 numpy 一致；PyTorch 缺省 1 —— 跨框架对拍时注意） |
| `vt-amax` / `vt-amin` | 空张量报错（与 numpy 一致）；NaN 传播见 clvt §5.4 |
| `vt-argmax` / `vt-argmin` | 返回**整数** dtype（`:int64`）0 维张量；`(vt-item …)` 或 `(vt-ref … 0)` 取标量。**不要 `(vt-ref (vt-argmax m) 0)` 后再当索引前忘记它是 fixnum** |
| `vt-norm` | 向量/矩阵范数；clnn 的 `compute-grad-norm` **未**使用它（自己算 `Σ g²`，原因是要跨参数聚合） |
| `vt-sum-axis` / `vt-mean-axis` | **clnn 自己的薄封装**（`nn::`，见 G-10），转发到 `vt-sum`/`vt-mean` 的 `:axis` |

### 6.7 矩阵 / 缩并类

| 函数 | 契约 |
|---|---|
| `vt-matmul` | 2D × 2D / 批量（ND）矩阵乘；f64 走 AVX512/ sb-simd 快路径，f32 亦快；复数见 §2.5 |
| `vt-dot` / `vt-outer` / `vt-einsum` | 输出 dtype 遵守提升；布尔输入按逻辑缩并（clvt §17.10.6） |
| `vt-trace` | `(matrix &key offset axis1 axis2 dtype out)`；布尔输入给 `:int64` |
| `vt-inv` / `vt-det` / `vt-solve` | 只能实数方阵；奇异矩阵行为见 clvt §16.11；clnn 仅在初始化器（orthogonal）与工具函数中用 |

### 6.8 神经网络专用（clvt `nn.lisp`）

| 函数 | 契约 |
|---|---|
| `vt-softmax` / `vt-log-softmax` | `(vt &key axis dtype out)`，`axis` 缺省 **-1**（末轴）。**dtype 必须浮点**（clvt §3.4：整数进整数出会截断），故不要在整数张量上直接调 |
| `vt-sigmoid` / `vt-relu` / `vt-leaky-relu` / `vt-gelu` / `vt-swish` / `vt-mish` / `vt-softplus` / `vt-hard-tanh` / `vt-hard-sigmoid` | 同上；**复数显式拒绝**；f32→f32，其余→f64 |
| `vt-mean-squared-error` / `vt-binary-cross-entropy` | **参数顺序是 `(y-true y-pred)`**（真值在前）。clnn **不使用**这两个函数（自定义损失见 §12） |
| `vt-cross-entropy` / `vt-cross-entropy-logits` | 两个语义完全不同的入口（概率式 vs logits 式，clvt §3.5）。clnn **不使用**，因为 CE 需要 label-smoothing 与自定义 reduction |
| `vt-stop-gradient` | **不是 clvt 函数**，是 clnn 的兼容入口（返回 `stop-gradient-node` 层，G-10） |

### 6.9 搬运 / 序列化类

| 函数 | 契约 |
|---|---|
| `vt-to-list` | `(vt &key bool)`。缺省**物理视图**（`:bool` → `(1 0 1)`）；`:bool t` 给 `(T NIL T)` |
| `vt-to-array` | `(vt &key dtype fast)`；`:fast t` 可能返回内部缓冲区，**只读** |
| `vt-data` | 原始缓冲区；clnn 只读 |
| `vt-copy` / `vt-copy-into` | 深拷贝 / 拷入目标；`copy-network` 靠它做参数隔离 |
| `vt-pad` / `vt-tile` / `vt-repeat` | 填充/平铺/重复；`tensor-pad/tile/repeat` 是薄封装 |

### 6.10 随机类

| 函数 | 契约 |
|---|---|
| `vt-random` / `vt-random-normal` / `vt-random-uniform` | 缺省 float64；`with-seed` 包住种子作用域 |
| `clnn` 的初始化器 | 全部经 `nn-initializer` 的 `init-weight`/`init-bias`，**不要**在层里直接掷随机数（dropout 例外，它需要逐元素独立采样） |

### 6.11 禁止用法清单（13 条）

| # | 禁止 | 原因 | 正确做法 |
|---|---|---|---|
| 1 | `(vt-zeros (vt-shape x))` | 恒 f64，破坏 D2/D3 | `(vt-zeros-like x)` |
| 2 | `(vt-map f int-array)` 期望浮点结果 | 整数截断 | 显式 `:dtype :float64` |
| 3 | `(vt-* bool-mask bool-mask)` 当乘性掩码 | 逻辑与，缩放丢失 | `vt-astype :float64` / `vt-scale` |
| 4 | `(vt-take x bool-mask)` | 布尔被当 0/1 下标，静默错行 | `(vt-take x (first (vt-nonzero mask)))`（1-D）或 `(vt-extract mask x)` |
| 5 | `(vt-- bool …)` | clvt 按 numpy 拒绝 | 用 `vt-logical-xor` 或先 astype |
| 6 | `(vt-concatenate a b)` 省 axis | 签名是 `(axis &rest tensors)` | `(vt-concatenate 0 a b)` |
| 7 | `(vt-item (vt-all …))` 后当数字用 | 返回 `T`/`NIL` | 需要数字用 `(vt-sum …)` 或 `%item-number` 风格 |
| 8 | `(vt-ref complex-tensor i)` 后做 `>`/`sqrt` | 返回复数 | 不支持复数（§2.5） |
| 9 | 依赖 `vt-astype complex :float64` 取实部 | 静默丢虚部 | `(vt-real …)` / 拆两路实数 |
| 10 | 对广播视图 `setf vt-ref` | 一次写命中多个元素 | 先 `vt-contiguous` |
| 11 | 用 `vt-map` 重写 `vt-sigmoid`/`vt-softmax` | 丢失 clvt 的浮点提升与数值稳定实现（max-shift） | 直接调 clvt 的 nn 函数 |
| 12 | 拼布尔张量 / 把布尔存进参数 | G-8 / 破坏 D5 | 转 float64 |
| 13 | 在 f32 路径上**依赖**「`(vt-- 1.0d0 x)` 恒为 f64」的旧行为（例如用它当「显式升精度」手段） | clvt §16.31.7 修好后它跟随 x 的 dtype，靠它升精度会静默失效 | 要升精度就写 `(vt-astype x :float64)`，别靠标量提升的副作用 |

### 6.12 使用频次与责任划分

核心源码（`nn-*.lisp`）对 clvt 的调用分布（§17.4 统计）：

| 类别 | 高频符号（次数） | 谁负责保证 dtype |
|---|---|---|
| 逐元素算术 | `vt-*`(111) `vt-+`(97) `vt-scale`(90) `vt--`(60) `vt-/`(14) | 调用点（操作数同 dtype） |
| 形状/视图 | `vt-reshape`(80) `vt-shape`(57) `vt-transpose`(50) `vt-slice`(35) `vt-flatten`(6) | clvt（保持 dtype） |
| 构造 | `vt-zeros`(5) `vt-zeros-like`(23) `vt-ones`(7) `vt-random-normal`(17) | **调用点必须显式** |
| 归约 | `vt-sum`(48) `vt-mean`(16) `vt-sum-axis`(10) `vt-amax`(4) | clvt 提升表 |
| 线性代数 | `vt-matmul`(56) `vt-einsum`(1) `vt-dot`(1) `vt-outer`(1) | clvt |
| 神经网络 | `vt-sigmoid`(13) `vt-tanh`(11) `vt-softmax`(6) `vt-relu`(5) `vt-log-softmax`(4) | clvt（§3.4 浮点规则） |
| 索引 | `vt-ref`(22) `vt-take`(7) `vt-put`(3) `vt-nonzero`(1) | **调用点先 `assert-index-dtype`** |
| 搬运 | `vt-copy`(47) `vt-data`(12) `vt-from-array`(10) `vt-item`(8) | clvt |
| 自定义 `nn::vt-*` | `vt-relu-derivative` 等 12 个 | clnn（见 G-10） |

---

## 7. 限制与不支持场景

### 7.1 梯度累加（gradient accumulation）

- 现状：**所有层**在第二次 backward 时都会**丢弃**上一次梯度（§5.1 的覆盖写）。
  `gradient accumulation` 因此无法靠「不清零」实现。
- 如需支持（按代价从低到高）：
  1. **在训练循环里自己攒**：把 micro-batch 的 `grads` 归约成张量后相加，最后一次性
     写回梯度槽 —— 不改库，最省事，但要自己处理 NIL 占位；
  2. **加一层「累积容器」**：包装层在 `backward` 里把子层梯度 `vt-+` 到自己的累加器，
     子层仍保持覆盖写。这样累加是**显式选择**，而不是所有层的默认；
  3. **改梯度写入层**（不推荐）：把 `setf` 改成 `(if 旧值 (vt-+ 旧值 增量) 增量)` ——
     等于把「漏清零会翻倍」重新装回库里，违反 §1.5 的 M4。
  不论哪种，loss 的 reduction 都要按 micro-batch 数加权。

### 7.2 参数共享（tied weights）

- 现状：`optimizer-state-registry` 以 `(list owner-layer 参数名)` 为键；同一层对象被两个
  容器引用时，`params` 会收集两次，配对契约（§4.4）被破坏，更新与状态都会错位。
- 如需支持：实现「主实例 + 引用」协议 —— 引用包装层的 `params` 返回**主实例**的三元组，
  并让 `collect-all-layers`/`zero-grad-children` 按 `eq` 去重。

### 7.3 同一层一次前向多次调用 / 一次前向多次 backward

- **同一层对象在一次 forward 中被调用两次**：后一次**覆盖**前一次的缓存（单槽），第一次
  无法 backward。`rnn-sequence` 之所以可行，是因为它把序列信息存成**列表**，并把 cell 的
  单槽缓存按时间步逐一重写后再调 cell backward。
- **一次 forward 多次 backward**：所有层都是同一值重复 `setf`（**不翻倍**），结果与
  只做一次完全相同（`manual-backward-contract-test` M4 的判据）。所以「一次前向多次
  反向」不会算错，但也**不会累加** —— 若需要两个 loss 的梯度之和，请先各自 backward
  到独立模型副本，或在 loss 层就把两项加成一个标量。

### 7.4 其它不支持项

| 场景 | 现状 | 出路 |
|---|---|---|
| 非线性/动态计算图（if/while 依赖数据） | 无自动微分磁带；backward 是手写静态链 | 在模型层显式组织控制流 |
| 多输入多输出层进 sequential | 容器契约禁止（§4.6） | 写包装层或自定义容器 |
| 二阶导 / 高阶优化 | 导数均为手写一阶 | 用 `tests/gradcheck-all-layers.lisp` 有限差分对拍 |
| 混合精度（amp） | 无 | §2.6 |
| f32 训练 | **不支持，也不需要**：clnn 的计算域恒为 `:float64`（§2.0）。f32 数据直接喂进来会被闸门规约 —— 那不是「用单精度算」，而是「先转成双精度再算」 | 要省内存/带宽请从数据管道入手（`vt-astype` 后复用），不要指望 clnn 用 f32 算 |
| `conv2d` / `max-pool2d` / `avg-pool2d` 的 f32 | **接受并规约**：层入口把 f32 折成 f64 再算（§2.3）。只有绕开层直接调内核（`im2col` 等私有函数）才会撞上 f64 哨兵 | 无需处理；f32 数据直接喂即可 |
| **复数张量** | 不支持，部分入口静默（G-4） | §2.5 |
| **布尔张量作为数据** | 不支持（只能作掩码） | §2.4 |
| 稀疏梯度 | 全程密集 | 无 |
| 线程安全 | 缓存/梯度/优化器状态全是可变槽；`*training-mode*` 是全局动态变量 | 单线程契约；并发请每线程独立模型与优化器 |
| batch-norm batch=1 训练 | 训练态 `running-var` 不更新（源码 `when (> batch 1)`） | batch ≥ 2，或改用 layer-norm |
| `ce-loss` 作用于序列 | 只接受 2D logits | 先 `reshape (B*T, C)` |

---

## 8. 与主流框架差异表

以 PyTorch 为参照：

| 维度 | PyTorch | clnn |
|---|---|---|
| 微分机制 | 动态计算图（autograd 磁带） | 手写静态 backward 链 + 显式缓存 |
| 梯度累加默认行为 | **默认累加** `.grad += new` | **默认覆盖** `setf` 新张量（**全部层族**，§5.1） |
| 清零方式 | 显式 `optimizer.zero_grad()` | `zero-grad!`（置 `NIL`）；backward 的覆盖写**隐含清零** |
| 忘记清零的后果 | 梯度跨步累加 → 发散（所有层一致） | **无影响**（所有层一致地覆盖写，§5.2） |
| 梯度存放 | 参数对象的 `.grad` | 层对象独立槽位（`dw/db/dgamma/…`） |
| 优化器与梯度 | optimizer 不清零、只更新 | 相同 |
| `None` 梯度 | 该参数不更新 | `(name . NIL)` 占位 + 跳过（占位是硬要求） |
| backward 调用对象 | `loss.backward()` | `(backward model grad)`（显式传 dL/dpred） |
| 参数更新方式 | 原地 `param.data -= …` | **setter 替换张量对象** |
| 权重布局 | Linear `(out, in)` | dense `(in, out)` |
| 训练/推理切换 | `model.train()/eval()` | `set-model-training!` / `with-training` |
| 梯度裁剪 | `clip_grad_norm_` | 优化器 `:grad-clip`（step 内全局范数）或 `clipped-gradient-update!` |
| AdamW 衰减默认 | 0.01 | `make-adamw` 0.01；`make-adam` 0.0（与 torch Adam 一致） |
| 学习率调度 | 调度器包装优化器 | 直接改 `optimizer-lr` 槽 |
| 序列化 | `torch.save`（pickle） | `save-model`（plist 文本，含 `:dtype`；`load-model` 禁 `*read-eval*`） |
| 默认 dtype | float32 | **float64** |
| dtype 纪律 | 隐式类型提升（严格但自动） | 显式契约（§2），违反时可能静默 |
| 布尔掩码 | `a[mask]` 花式索引 | 不支持 `vt-ref`；用 `vt-extract`（展平）/ `(vt-take a (first (vt-nonzero mask)))`（1-D） |
| `Uniform` 初始化 | `U(-1/√fan, 1/√fan)` | `he-uniform`/`xavier-uniform` 各有定义，见 `nn-initializer.lisp` |

---

## 9. 激活函数缓存约定

`activation-layer` 与 `dense`（内置激活）采用同一套约定。**导数函数期望的输入**是理解
反向代码的关键。

### 9.1 缓存选择表

| kind | forward 缓存 | backward 使用的量 | 导数期望输入 |
|---|---|---|---|
| `:relu` | `z-cache` | `grad · relu'(z)` | **z**（激活前） |
| `:leaky-relu` | `z-cache` | `grad · leaky-relu'(z)` | **z** |
| `:gelu` | `z-cache` | `grad · gelu'(z)` | **z** |
| `:swish` | `z-cache` | `grad · swish'(z)` | **z** |
| `:mish` | `z-cache` | `grad · mish'(z)` | **z** |
| `:softplus` | `z-cache` | `grad · sigmoid(z)`（就地） | **z** |
| `:hard-tanh` | `z-cache` | `grad · (−1≤z≤1 ? 1 : 0)`（就地） | **z** |
| `:hard-sigmoid` | `z-cache` | `grad · hard-sigmoid'(z)` | **z** |
| `:sigmoid` | `a-cache`（输出） | `grad · a(1−a)` | **a**（激活后） |
| `:tanh` | `a-cache` | `grad · (1−a²)` | **a** |
| `:softmax` | `z-cache` + `a-cache` **双缓存** | `gz = y ⊙ (gy − Σ gy·y)`，y = a-cache | **a** |
| `:log-softmax` | `z-cache` + `a-cache` **双缓存** | `gz = gy − s·Σ gy`，s 由 z **重算** `softmax(z)` | **z** |
| `:linear` / `:none` | 均为 `NIL` | `grad` 原样回传 | — |

### 9.2 语义说明

- **z-cache 语义**：本层输入（activation-layer）或线性变换输出（dense）——「进入激活之前」。
- **a-cache 语义**：激活输出。凡导数用输出表示更简单的（`σ′ = a(1−a)`、`tanh′ = 1−a²`）
  或离不开输出的（softmax 雅可比），用 a-cache。
- softmax/log-softmax 双缓存：前者要用输出 y，后者为避免数值问题**从 z 重算** softmax。
- dense 与独立 activation-layer 的分工：`(make-dense n :activation :relu)` 把激活并入
  dense（缓存 z/a 各一）；`(make-activation-layer :relu)` 是独立层。两者反向约定一致。

### 9.3 clnn 内部导数函数（**不是 clvt API**）

`nn-layers-dense.lisp` 顶部定义了 11 个 `nn::vt-*-derivative` 与 2 个薄封装：

| 符号（`nn::`） | 期望输入 | 说明 |
|---|---|---|
| `vt-relu-derivative` / `vt-leaky-relu-derivative` | z | 分段常数 |
| `vt-gelu-derivative` / `vt-swish-derivative` / `vt-mish-derivative` | z | 闭式近似 |
| `vt-softplus-derivative`（内联在 backward） | z | `sigmoid(z)` |
| `vt-hard-sigmoid-derivative` / `vt-hard-tanh`（内联） | z | 分段 |
| `vt-sigmoid-derivative` / `vt-tanh-derivative` | **a** | 用输出表示 |
| `vt-mean-axis` / `vt-sum-axis` | — | 转发 `vt-mean`/`vt-sum` 的 `:axis` |

**注意（G-10）**：这些符号在 `nn` 包里，与 clvt 的 `vt-` 前缀**同形但不属于 clvt**。
clvt 目前没有 `vt-*-derivative`；若将来引入，二者仍是不同包的不同符号，不会冲突，但
**阅读代码时不要假设前缀相同就语义相同**：它们是 `nn::` 私有符号，未导出，
clvt 里没有同名 API。

---

## 10. 形状约定

### 10.1 dense：rank 1/2/N 的展平与还原

```lisp
(in-dim      (or (dense-in-dim l) (car (last in-shape))))        ; 末维 = 特征维
(batch-size  (if (= rank 1) 1 (reduce #'* (butlast in-shape))))  ; 前面所有维相乘
(x-flat      (if (and (= rank 2) (= (second in-shape) in-dim))
                 input                                            ; 标准 2D：零拷贝
                 (vt-reshape input (list batch-size in-dim))))
```

| 输入 rank | 处理 | 输出形状 | backward 输入 | dX 返回形状 |
|---|---|---|---|---|
| 1 `(D,)` | batch=1，reshape `(1, in-dim)` | `(out-dim,)` | `(out-dim,)`，内部 unsqueeze | `(D,)` |
| 2 `(B, in-dim)` | 零拷贝直用 | `(B, out-dim)` | `(B, out-dim)` | `(B, in-dim)` |
| ≥3 `(B₁,…,Bₖ, in-dim)` | reshape `(B₁·…·Bₖ, in-dim)` | `(B₁,…,Bₖ, out-dim)` | 同输出形状 | **还原为原始输入形状** |

**grad-flat reshape 规则**：backward 用 `input-cache` 的 car 记录的原始形状**整体
reshape** 还原（不是逐维裁剪），因此任何 rank 都能还原。`in-dim` 未声明时以输入末维
回填（PyTorch 风格）；dense 没有专门的形状断言，声明值不匹配时错误会在 `vt-reshape`
（元素数不符）处抛出。

### 10.2 其它层形状总表

| 层 | forward 输入 → 输出 | backward 输入 → 返回 (dX) |
|---|---|---|
| conv2d | `(B,C,H,W)` → `(B,out-c,OH,OW)` | 同形互逆（col2im 按位置累加） |
| max/avg-pool2d | `(B,C,H,W)` → `(B,C,OH,OW)` | 同形互逆 |
| global-avg-pool2d | `(B,C,H,W)` → `(B,C)` | `(B,C)` → `(B,C,H,W)` |
| flatten（start-dim=1） | `(B,d₁,…,dₖ)` → `(B, d₁·…·dₖ)` | → 缓存的原始形状 |
| batch-norm | rank-2 `(B,F)`；ND 走「特征维在末轴」路径 | 同形互逆 |
| layer-norm | 任意 rank，按 `normalized-shape`（末 K 维）归一化 | 同形互逆 |
| embedding | 整数索引任意 rank `(i₁,…,iₖ)` → `(i₁,…,iₖ, D)` | `(i₁,…,iₖ,D)` → **`vt-zeros(idx-shape)`**（索引无梯度） |
| rnn-cell | `(B,in)` → `(B,H)` | `(B,H)` → `(values (B,in) (B,H))` |
| rnn-sequence | `(B,T,in)` → `(B,T,H)` | `(B,T,H)` → `(B,T,in)` |
| lstm | `(B,T,in)` → `(values out (B,T,H) h (B,H) c (B,H))` | `(B,T,H)` → `(values (B,T,in) dh? dc?)` |
| gru | `(B,T,in)` → `(values out (B,T,H) h (B,H))` | `(B,T,H)` → `(values (B,T,in) dh?)` |
| sdpa | `(list Q K V)` 各 `(B,S,d)` → `(B,Sq,dv)` | → `(values dQ dK dV)` |
| mha | `(list Q K V)` 各 `(B,S,dim)` → `(B,Sq,embed)` | → `(values dQ dK dV)` |
| transformer-block | `(B,S,D)` → `(B,S,D)`（pre-norm） | 同形互逆 |
| ce-loss | logits 必须 2D `(N,C)`；标签一维**整数** | — |

**通用规则**：除 dense 的 batch 展平与 flatten 的 start-dim 外，**所有层都不改变输入
rank**；多维到 2D 的转换都发生在层内部且必须可逆。

### 10.3 embedding 的 dX 为什么是零

`embedding` 的输入是**索引**（整数，不可导）。backward 必须返回一个与输入同形状的张量
才能让 `sequential` 的链不断，于是返回 `(vt-zeros idx-shape)`。它的 dtype 是 float64
（`vt-zeros` 缺省），与「上游梯度」的 dtype 无关 —— 因为上游根本没有梯度可言，这个
张量只被丢弃。若 embedding 不是第一层（例如 `embedding → dense`），dense 的 backward
会把 dX 传给 embedding 的 backward，后者**忽略**传入的 `grad-output`（只用
`indices-cache` 与 `norm-scale`），这是有意为之。

---

## 11. 优化器契约

- **输入**：`param-list = ((owner "name" param setter) …)` 与
  `grad-list = (("name" . grad-or-NIL) …)`，按位置配对，长度必须相等
  （`assert-param-grad-alignment` 硬校验），名称必须一致
  （`assert-layer-name-alignment`）。
- **步内流程**：`global-clip-coeff`（全参数梯度全局 L2 范数裁剪，`clip ≤ 0` 不裁）→
  逐参数：`NIL` 梯度/冻结层跳过 → L2 衰减 `g + wd·param`（adamw 为解耦）→ 更新状态
  （registry 以 `(owner 参数名)` 为键，**惰性建 `vt-zeros-like param`**）→
  `funcall setter new-param` **替换参数张量**。
- **状态 dtype（D3）**：全部状态张量由 `vt-zeros-like param` 创建，与参数同 dtype。
  参数恒为计算域（§2.2 D3），所以状态也恒为 f64。
- **状态**：`state-registry` 哈希表 + `step-count`。步号由 `update!` 授予、`optimizer-step`
  只读取（§3.5）—— 一次 `update!` 恰好一步，与「`optimizer-step` 被调了几次」解耦。
  跨 step 持久；换优化器实例即状态归零。
- **家族**：sgd（momentum/nesterov）、adam（amsgrad）、adamw、rmsprop（alpha/centered/
  momentum）、adagrad（lr-decay）。
- **不清零梯度**（§5.4）；不校验梯度是否「新鲜」。

---

## 12. 损失函数契约

### 12.1 接口

- `compute-loss (loss predicted target) → 0 维张量`（`:none` 时为逐元素张量）。
- `compute-loss-gradient → dL/dpredicted`，形状恒等于 `predicted`，dtype 恒等于
  `predicted`（`ce-loss` 例外，见 §12.3）。
- `reduction ∈ {:mean :sum :none}`。
- 损失对象**无状态、无缓存**，可与任意模型自由组合。

### 12.2 各损失的 reduction 口径（含 G-9）

| 损失 | `:mean` 除的是什么 | 与 PyTorch 的差异 |
|---|---|---|
| `mse-loss` | **元素总数** `n = ∏ shape` | torch 的 `reduction='mean'` 也是元素总数 ✅ |
| `bce-loss` | 元素总数 | 同上 ✅ |
| `ce-loss` | **batch 数**（每样本先按类别求和） | torch 也是 batch 均值 ✅ |
| `huber-loss` | 元素总数 | ✅ |
| `cosine-similarity-loss` | batch 数 | torch 同 ✅ |

> 说明：「`:mean` = 除以元素总数」只对逐元素损失成立；`ce-loss` 与
> `cosine-similarity-loss` 按 batch 求均值（每个样本先内部归约）。
> 自定义损失务必明确自己除的是哪个数 —— 这个差别在「多分类 + 类别数很大」时会直接
> 表现为学习率等效值差 C 倍。

### 12.3 CE 专项契约

- `predicted` 必须是 2D `(N,C)` logits（内部 log-softmax + NLL，max-shift 数值稳定）。
- `target` 必须是**整数** dtype（D4：`assert-index-dtype` 拦截；浮点/布尔/复数都会报
  「标签必须是整数 dtype」）。
- 标签越界（含负数）显式报错，不静默取值。
- 支持 label-smoothing：`(1−s)·one-hot + s/C`，O(batch×C) 构造（不做 `eye(C)`，
  避免 4000 类时 122 MB 的临时单位阵）。
- **dtype（G-7）**：`%ce-target` 用 `double-float` 数组构造目标分布，与 f32 logits
  相乘后提升为 f64，故 `ce-loss` 的输出恒为 f64。这是已知缺口，见 §2.6。

### 12.4 为什么 clnn 不用 clvt 的 CE

clvt 提供 `vt-cross-entropy`（概率式）与 `vt-cross-entropy-logits`（logits 式，
§3.5）。clnn 仍自带 CE，原因是：①需要 label-smoothing；②需要 `:none` reduction 返回
逐元素张量；③需要与 `compute-loss-gradient` 的解析梯度配对（不重跑一遍前向）。
**契约**：两者数值口径必须一致 —— §17.2 D-CE 记录了 `(N,C)`+int64 标签下与
`vt-cross-entropy-logits` 的逐位对照（2.4076059644443806d0 双方一致）。

---

## 13. 序列化契约

### 13.1 往返

- `save-model (model filepath)`：遍历 `collect-all-layers`，槽位转 plist 文本
  （**含 `:dtype` 与 `:shape`**）；`load-model` 重建并逐位还原。
- `load-model` 绑定 `*read-eval* = nil`（防模型文件执行任意代码）。
- **布尔 / 复数 roundtrip**（`tests/dtype-contract-test.lisp` G 组）：`:bool` 与
  `:complex128` 都能原样往返（`vt-to-list` 给数值，`vt-from-sequence … :dtype` 按 dtype 重建）。
- `copy-network` 提供内存内深拷贝（含 `trainable` 与梯度槽对齐）。
- 加载环境需存在同名层类；跨版本加载不做兼容承诺（plist 里没有版本号，
  字段缺失会走 `getf` 的默认值 —— 改槽名时这是静默的）。

### 13.2 加载后的状态：推理态、计算域

`load-model` 做两件额外的事，都不是"还原存档"的字面含义，但都是最常用流程需要的：

| 动作 | 原因 |
|---|---|
| `normalize-params!` | 存档可能是 f32 时代的产物或从别处转来的权重。就地规约到 `+COMPUTE-DTYPE+`，避免第一次 forward 逐层 astype |
| `set-model-training! nil` | `*TRAINING-MODE*` 默认 `T`，若不切，加载后 dropout 仍然生效、BN 用 batch 统计量 —— **同一输入两次 forward 结果不同**，且没有任何报错 |

要继续训练，显式切回：`(train! model)`。手工构造的模型也建议显式表态：

```lisp
(defparameter *model* (build-model (make-mlp) sample))
(eval! *model*)        ; 推理
(train! *model*)       ; 训练
```

两个函数都返回 `model`，可以串联。

---

## 14. 自定义层开发契约（checklist）

最小义务（缺一不可）：

```lisp
(defclass my-linear (layer)
  ((w :accessor my-w) (dw :accessor my-dw :initform nil)
   (x-cache :accessor my-x-cache :initform nil)))

(defmethod forward ((l my-linear) x)
  (assert-index-dtype-or-float x "my-linear")   ; ① 输入 dtype 守卫（可选但推荐）
  (setf (my-x-cache l) x)                       ; ② 缓存 backward 需要的一切
  (vt-matmul x (my-w l)))                       ; ③ 输出 dtype = 输入 dtype（D1）

(defmethod backward ((l my-linear) grad-output)
  (let* ((x  (my-x-cache l))
         (dw (vt-matmul (vt-transpose x) grad-output)))
    (setf (my-dw l) dw)                         ; ④ 覆盖写（本步梯度，非累加）
    (vt-matmul grad-output (vt-transpose (my-w l)))))  ; ⑤ dX dtype = 输入 dtype（D2）

(defmethod params ((l my-linear))               ; ⑥ params/grads 严格同构
  (list (list l "w" (my-w l) #'(lambda (v) (setf (my-w l) v)))))
(defmethod grads ((l my-linear))
  (list (cons "w" (my-dw l))))                  ; 未反传时 NIL 占位
(defmethod grad-slots ((l my-linear)) '(dw))
(defmethod cache-slots ((l my-linear)) '(x-cache))
```

**验收清单**：

| # | 项 | 判据 |
|---|---|---|
| 1 | `params`/`grads` 长度、顺序、名称一致 | 否则 `assert-param-grad-alignment` 报错 |
| 2 | backward 返回值形状 == 最近 forward 输入形状 | `tests/gradcheck-all-layers.lisp` |
| 3 | backward 返回值 dtype == 输入 dtype | 用 `vt-zeros-like`，不用 `vt-zeros` |
| 4 | 梯度槽 dtype == 参数 dtype | 同上 |
| 5 | 索引/标签走 `assert-index-dtype` | 否则布尔掩码会静默取错行 |
| 6 | 有限差分对拍通过 | `gradcheck` |
| 7 | `zero-grad!` 后梯度槽为 `NIL` | —— |
| 8 | `clear-step-caches!` 后缓存为 `NIL`，且此后不得再 backward | —— |
| 9 | 容器层补 `zero-grad-children` + 递归清理 + `set-training!` | —— |
| 10 | 不引入裸浮点字面量（一律 `d0` 后缀） | clnn.asd 未 pin `*read-default-float-format*`（clvt 已 pin）；新增代码须自律 |

---

## 15. 缓存生命周期

```
build-model ──dry-run──→ clear-step-caches!        （清掉试探缓存）
    │
训练步: zero-grad! → forward(写缓存) → backward(读缓存/写梯度) → optimizer-step → clear-step-caches!
    │
推理步: forward → clear-step-caches!
```

- `clear-forward-cache!`：只清 `transient-cache-slots`（默认空），**任意时刻安全**
  （§17.1 T6）。
- `clear-step-caches!`：清 `cache-slots` 全部 + 递归子层，**只能在整个训练步结束后调用**。
- 不属于缓存、清理不触碰的持久状态：`running-mean/running-var`（BN 统计量）、
  `rnn-state`（用 `reset-rnn-cell-state!` 单独清）、优化器 registry。

---

## 16. clvt 语义变更跟踪矩阵

> 用法：clvt 每次升级，先读它的 `CONVENTIONS.md` 新增章节，再对照本表检查 clnn 的
> 受影响点。

| clvt 变更 | 章节 | 受影响 clnn 代码 | 影响 | 处理 |
|---|---|---|---|---|
| 比较/逻辑/谓词返回 `:bool`（原 `:int8`） | §17.1 | 无（clnn 未导入比较算子） | 无 | 记录：`nn` 包不导入 `vt->`/`vt-=`，用户需用 `clvt:` 前缀（§6.4 备注） |
| `vt-*`/`vt-+` 对 bool 改派逻辑内核 | §17.3 | dropout/sdpa 掩码、`tensor-where`/`masked-fill` | **静默错值** | 掩码显式 `:dtype`；`%mask-true-p` 收 bool 条件 |
| `vt-map` 写回 `:bool` 时真值化 | §17.10.1 | 两处 dropout 掩码 | 掩码被传染 | 已修 |
| `vt-map` 整数输入截断浮点 | §3.4（旧问题） | 同上 | 缩放被截断 | 显式 `:dtype` |
| `vt-item` 对布尔返回 `T/NIL` | §17.8 | `compute-grad-norm`、`std-inv-from-var` 等 8 处 `vt-item` | 均作用于浮点归约结果 → 无影响（已逐个核对） | 记录于 §6.6 |
| 布尔下标被 `vt-take` 接受 | §17.10.7 | `embedding`、`tensor-gather` | **静默取第 1 行** | `assert-index-dtype` |
| `vt-==` 家族 / `vt-astype :bool` 真值化 | §17.1 | 无 | 无 | —— |
| 复数在 nn 函数上被拒绝 | §16.11 | `dense`/`conv` 的内置激活 | 报错（确定性） | 记录（§2.5） |
| 复数在 `matmul`/`sum`/`mean` 上被接受 | §16.11 | `dense`/`conv`/损失 | **静默复数** | G-4（未修，契约声明不支持） |
| `vt-concatenate` 拒绝 bool | §17.10 | `lstm`/`gru` 门拼接（浮点，不受影响） | 无 | G-8 记录 |
| `vt-zeros` 缺省 `:float64` | §6.4 | 优化器状态、梯度缓存、rnn 四支、embedding `dw`、bn running 统计量 | **dtype 漂移**（现在恒等于计算域，不再是问题） | 已修：状态改用 `vt-zeros-like`，其余直接受益于单一计算域 |
| `:uint32/:uint64` 为保留符号 | §3.1 | 无 | 无 | §2.1 记录 |
| `vt--` 对首个标量操作数做强提升（`(vt-- 1.0d0 x-f32)` → f64），其余算子走弱提升 | §16.31（NEP 50 弱标量） | `dense`/`activation`/`norm`/`rnn` 里全部 `1 − x` 写法 | **前向 f32、反向 f64** 的静默混合精度 | **上游已修**：clvt §16.31.7 把剩下十一个「交换顺序就换 dtype」的算子补齐，二元提升恢复可交换。clnn 侧只需修根因 ②（缓冲起点），见 G-11 |
| clvt `§16.31.7` 补齐弱标量（`vt--`/`vt-pow`/`vt-gcd`/`vt-lcm`/`vt-where`/`vt-atan2`/`vt-hypot`/`vt-logaddexp`/`vt-copysign`/`vt-nextafter`/`vt-float-power`） | §16.31 / §16.31.7 | 全部含标量的算子调用；clnn 直接用的是 `vt--`（激活导数、门控、`1−x`）与 `vt-*`/`vt-+`/`vt-/` | 二元提升恢复可交换，标量与张量同 dtype —— 单一计算域在标量参与时不会隐式加宽 | 保留 `dtype-contract-test` A11 组作为 clvt 语义基线（判据：`a ⊔ b == b ⊔ a`） |

**跟踪流程（建议写入 PR 模板）**：

1. clvt 升级后，读它的 `CHANGELOG.md`，找出「行为变化」条目；
2. 对每条填本表：受影响 clnn 文件、影响性质（无/报错/静默错值）、处理方式；
3. 若性质是「静默错值」，必须先加测试再改代码；
4. 更新 §2 的铁律与 §6 的禁令（若边界变了）；
5. 更新本文件头的 clvt 指纹与 §17 的复跑记录。

---

## 17. 验收探针

环境：SBCL 2.6.10 x86-64-linux，`--dynamic-space-size 4096`，clvt `f19d8e1`。
下表每一项都能在 `tests/` 里找到对应断言，全部在 `run-tests.sh` 的 `FAST_TESTS` 中，
`./run-tests.sh --fast` 一键复跑（18 个套件，约 40 秒）。

### 17.1 行为契约探针

| # | 验证点 | 结果 |
|---|---|---|
| T1 | dense 连续两次 backward：`dw` 精确等于「第二轮单独计算」值 | ✅ maxdiff = 0.0；且 ≠ 两次之和（非累加） |
| T2 | `zero-grad!` 置 `NIL`；`params/grads` 同构；`grads` 保留 NIL 占位 | ✅ 3 项 |
| T3 | 全 `NIL` 梯度时 `optimizer-step` 跳过、不报错、参数对象不变 | ✅ |
| T4 | `rnn-cell` 两次 backward **不**累加（覆盖写）；`reset-rnn-cell-state!` 清 state | ✅ 同输入两次 backward 逐位相同 |
| T5 | `embedding` 同一行三次命中 → `dw` = 3.0（scatter-add） | ✅ |
| T6 | `clear-forward-cache!` 后可继续 backward；`clear-step-caches!` 后缓存全 `NIL` | ✅ |
| T7 | 参数被 setter 替换为新对象；梯度槽保持旧对象 | ✅ |
| T8 | 标准六步循环（§3.1）60 步 CE + Adam：loss 1.247 → 0.220 | ✅ |
| T15 | 18 类层的 `cache-slots`/`grad-slots` 声明逐一导出 | ✅ 见 §4.1 表 |

### 17.2 clvt 语义探针

这些探针问的是 clvt 的底层行为（比较结果是什么 dtype、布尔索引怎么处理、
提升表怎么定结果类型），与 clnn 的策略无关 —— clvt 一旦变了，它们就是最早报警的地方。

| # | 探针 | 实测结果 | 用于 |
|---|---|---|---|
| D-A1 | `(vt-= …)` 的 dtype | `:bool`（值仍 0/1） | §2.4 |
| D-A2 | `(vt-* bool bool)` | `:bool`，逻辑与 | §2.4.2 |
| D-A3 | `(vt-- bool bool)` | 报错（numpy 同） | §6.5 |
| D-A4 | `(vt-scale bool 2.0)` | `:float64`，数值 2.0（**不是**逻辑运算） | §6.5 禁令 3 的替代写法 |
| D-A6/7 | `(vt-map f bool)` 返回 2 | `:bool`，真值化为 1 | §2.4.3 |
| D-A9 | `(vt-map (λ) int64)` 返回 0.5 | `:int64`，截断为 0 | §2.4.3 |
| D-A10 | `(vt-zeros '(2 2))` | `:float64` | §2.2 D2/D3 |
| D-B1/2 | `tensor-where` 收 `:bool` 与 `:int8` 条件 | 结果逐位一致 | §2.4.1 |
| D-C1…C4 | dropout 掩码 dtype（输入 f64/f32/int64/bool） | f64→f64，f32→f32，int64→**f64**，bool→**f64** | G-1 |
| D-C5 | p=0.3 整数输入下的保留位 | 1.4286（修复前为 1） | 已修 |
| D-D1 | f32 权重经一步 SGD 后的参数与状态 dtype | **f32 / f32**（修复前均 f64） | G-2 |
| D-D2/3 | **权重也 f32** 时 dense 的前向 / dX / dw dtype | 均 f32 | §2.6 |
| D-D6 | f32 的 mse 损失与梯度 | f32 | §2.6 |
| D-E1 | 复数输入 dense 前向 | **:complex128**（无守卫） | G-4 |
| D-E2/4 | 复数 `dw` 的 `global-grad-norm` / `compute-grad-norm` | 返回复数 | G-4 / §5.6 |
| D-E3 | 复数 `dw` 走 SGD | 参数变复数，无报错 | G-4 |
| D-E5 | 复数 logits 的 CE 损失 | 复数损失值 | G-4 |
| D-F1/2 | bool / complex 的 `%serialize-vt` roundtrip | dtype 与值均保持 | §13 |
| D-F2b | `(vt-concatenate bool bool)` | `TYPE-ERROR`（clvt 限制） | G-8 |
| D-G1/2 | CE 的浮点 / 布尔标签 | 报错（修复前文案指向 FIXNUM） | G-6 |
| D-G3 | CE 的 int64 标签 | 2.4076059644443806d0（与 clvt 的 logits 式 CE 一致） | §12.4 |
| D-G4/G6 | 布尔索引的 embedding / gather | **静默取第 1 行**（修复前）→ 现报错 | G-5 |
| D-G5 | 浮点索引 | 报错（clvt 口径 → 现 clnn 口径） | G-5 |
| D-CE | clnn 的 ce-loss 与 clvt 的 `vt-cross-entropy-logits` 对拍 | 双方均 `2.4076059644443806d0`；clnn 的 logits 梯度按行求和 ≈ 1e-17（数值上精确为 0） | §12.4 |

### 17.3 计算域验收探针

这组探针只问三件事：**规约是否发生**、**该不该规约的东西是否被误伤**、
**代价是否只付一次**（第一次 warn，之后静默）。
下表 15 行与 `dtype-contract-test` 的 I 组一一对应（编号 15 个，实际断言 27 条 ——
带 a/b/c 的行在测试里各含多条，I5/I6/I7 还带激活函数与层族的循环）。

| # | 探针 | 实测 | 说明 |
|---|---|---|---|
| I1 | f32 `x` → `dense` 的 `forward` | 输出 **f64** | 闸门基线 |
| I2 | 闸门是否正确性替换而非破坏性写入 | 调用方手里的 f32 张量仍是 **f32** | 返回新对象；`ensure-compute-dtype` 不改入参 |
| I3 | f32 `grad-output` → `backward` 的 dX | **f64** | 反向同样过闸门（×20 层同源） |
| I4 | f32 `x` → `dense(:tanh)`：fwd/dX/dw/db | 四项全 **f64** | 含 `1 − tanh²` 这类弱标量运算的层，反向也不漂移 |
| I5 | f32 `activation-layer(:tanh / :sigmoid)` 的 fwd/dX | **f64** | 导数 `1 − x²`、`x(1 − x)` 不再有窄路径 |
| I6a–d | f32 `batch-norm` 训练态 / 推理态 / running 统计量 / 推理态 dX | **f64** ×4 | buffer 不在 `params` 里，`normalize-params!` 碰不到 —— 只能靠闸门兜住，故单列 |
| I7a–d | f32 `rnn-cell` / `lstm` / `gru` / `rnn-sequence` 的 fwd/dX/全梯度槽 | **f64** | RNN 的状态缓冲与零 dX 全部落在计算域 |
| I8a–b | f32 `embedding` 的权重与 `dw` / 索引 dtype | 权重、`dw` **f64**；索引仍 **int64** | 一个探针里同时验「折了该折的」与「没折不该折的」 |
| I9a–b | f32 logits → `ce-loss` 的损失值 / `dP` | **f64** / **f64** | 目标分布由 logits 的 dtype 推导，两者恒同为 f64 |
| I10 | f32 `conv2d` | **f64** | 层归约；`%ensure-float64-data` 只管内核入口，不再承担「拒绝 f32 输入」的职责 |
| I11a–c | `:bool` 掩码 / `:int32` 索引 / `:float32` 计算域窄版本分别过闸门 | 前两者**原样通过**；第三者**被折** | 三域划分的守门断言 —— 一旦 `ensure-compute-dtype` 写错，这里立刻红 |
| I12a–b | 手工塞入模型的 f32 权重 / 调 `normalize-params!` 之后 | 归一前仍是 f32；归一后 2 个参数变 **f64** | 参数槽不走闸门（它们不是 `forward` 的入参），必须另有通道 |
| I13a | 同一层连续 5 次 f32 输入产生的警告条数 | **1** 条 | 每步都告警等于没有告警 |
| I13b | 三个不同的层各收一次 f32 产生的警告条数 | **1** 条 | 去重键是 dtype，**不含层名** —— 否则 20 层模型就是 20 条 5 行警告 |
| I13c | 清空 `*compute-dtype-warned*` 后再喂 f32 | 又报 **1** 条（共 2 条） | 留一条「想知道还有哪些层在收 f32」的逃生门 |

### 17.4 符号审计

| 项 | 结果 |
|---|---|
| `nn` 包内 `vt-*` 且 `fboundp` 的符号 | 106 个（去重后，实测） |
| 其中在 `clvt` 包内为 `:external` 且有函数绑定 | 94 个（无缺失、无未导出） |
| 其中为 clnn 自有（`nn::`） | 12 个（11 个导数/薄封装 + `vt-stop-gradient`） |
| 导入但未使用（死导入） | `vt-reduce`、`vt-do-each`、`vt-split` 三个在核心源码中只出现在 `nn-package.lisp` 的导入表里；`vt-tile`/`vt-repeat`/`vt-pad` 经 `tensor-tile`/`tensor-repeat`/`tensor-pad` 间接使用 |

口径：包内实测（`do-symbols` + `symbol-package` 判定归属，去重）—— 不是「源码里
出现过几次 `vt-*`」。结论：**没有任何一个导入符号在 clvt 里缺失或未导出**。
`contract-probes` 的 T15 逐条复核这张表。

## 附录 A：源文件地图

| 文件 | 职责 |
|---|---|
| `nn-package.lisp` | 包定义与 clvt 符号导入（94 个） |
| `nn-protocol.lisp` | `layer` 基类、协议泛型、训练模式、loss/optimizer 基类、`stop-gradient-node`、**dtype 契约工具（§2）** |
| `nn-layers-dense.lisp` | dense / activation-layer / flatten / residual、全部激活导数函数 |
| `nn-layers-conv.lisp` | im2col/col2im、conv2d、三种池化 |
| `nn-layers-norm.lisp` | dropout / batch-norm（2D+ND）/ layer-norm、`%vt-sum-over-axes` |
| `nn-layers-rnn.lisp` | rnn-cell / lstm / gru / rnn-sequence（BPTT） |
| `nn-layers-attention.lisp` | sdpa / multi-head-attention / transformer-block |
| `nn-layers-embedding.lisp` | embedding（max-norm / scale-grad-by-freq） |
| `nn-loss.lisp` | mse / bce / ce / huber / kl / smooth-l1 / cosine |
| `nn-optimizer.lisp` | 五个优化器、全局范数裁剪、配对断言、**状态 dtype 契约** |
| `nn-model.lisp` | sequential、训练工具、清零与缓存协议、序列化、copy-network、tensor-* 工具 |
| `nn-scheduler.lisp` | 学习率调度 |
| `tests/dtype-contract-test.lisp` | **本文档 §2/§6 的可执行对照**（70 断言，A–I 组） |
| `tests/manual-backward-contract-test.lisp` | **§1.5 手写架构协定 M1/M3/M4/M5/M7 的可执行对照**（16 断言） |
| `tests/contract-probes.lisp` | **§3/§4/§5/§15 的可执行对照**（T1–T15，58 项） |
| `tests/gradcheck-all-layers.lisp` | 全层有限差分对拍 |
| `tests/run-tests.sh` | 测试入口（14 个快速文件 + 2 个慢速文件） |

## 附录 B：常见错误速查

| 症状 | 根因 | 对照契约 |
|---|---|---|
| 训练 loss 不降、梯度越来越大 | 所有层都是覆盖写，**该症状不会由漏清零引起**，请查学习率/数据/梯度裁剪 | §5.2 |
| 部分参数永远不更新 | `params`/`grads` 项数不一致，或 backward 忘写梯度槽 | §4.4 / §5.3 |
| backward 拿到旧值/崩溃 | 一次 forward 两次 backward，或两次 forward 之间夹了 backward | §3.1 / §7.3 |
| 优化器状态错位 | 同一层对象放进两个容器（参数共享） | §7.2 |
| CE 报形状/类型错 | logits 非 `(N,C)`，或标签非整数 dtype | §12.3 |
| CE 报「标签必须是整数 dtype」 | 传了浮点/布尔标签（如误把掩码当标签） | §2.2 D4 |
| BN batch=1 训练统计异常 | `running-var` 仅在 batch>1 时更新 | §7.4 |
| dense 输入末维不匹配 | 多维输入未按「末维=特征维」组织 | §10.1 |
| **同一输入两次 `forward` 结果不同** | 模型还在训练态（dropout 生效）。`load-model` 已自动切推理态；手工构造的模型要 `(eval! model)` | §13.2 |
| **Adam 有效学习率偏小、grad-clip 表现怪** | 用逐层 `update!` 当成「一次更新」：每一步都推进了步号，偏差校正 `1/(1−β^t)` 一直偏小，裁剪也逐层各剪一次 | §3.5 |
| **mse/bce 训练比同类任务慢几倍** | `target` 是 f32 而 `predicted` 是 f64（已修：闸门现在也规约 target） | §5.7 |
| **训练越训越慢、dtype 自己变成 f64** | 优化器状态/零张量按 `vt-zeros` 造（G-2，已修） | §2.2 D2/D3 |
| **dropout 几乎不起作用、loss 只降一点** | 掩码被输入 dtype 传染，`1/(1-p)` 被截断成 1 | §2.4 / G-1 |
| **embedding 取到了错误的行** | 把布尔/浮点掩码当索引传进去 | §2.2 D4 / G-5 |
| **梯度范数算出复数、`>` 报类型错** | 用了复数参数（不支持） | §2.5 / G-4 |
| **布尔掩码相乘后值变成 0/1 而不是数值** | `vt-* bool bool` 走逻辑与 | §6.11 禁令 3 |
| `vt-concatenate` 报 `not of type REAL` | 拼了布尔张量 | G-8 |
| 自定义层 gradcheck 过但训练不动 | dX dtype 与输入不一致（用 `vt-zeros` 造零） | §2.2 D2 |

## 附录 C：术语

| 术语 | 含义 |
|---|---|
| 铁律 D1–D5 | §2.2 的五条 dtype 约束 |
| 协定 M1–M7 | §1.5.3 的七条手写求导约束 |
| 缺口 G-* | §2.8 的已知违背设计意图之处 |
| 探针 T-*/D-*/I-* | §17 的可复跑断言编号 |
| 计算域 | `+COMPUTE-DTYPE+`，当前恒为 `:float64`（§2.0） |
| 索引域 / 掩码域 | 整数（标签、下标）/ `:bool`（条件）；两者都不进 autodiff 链（§2.1） |
| 覆盖写 | backward 用 `setf` 整体替换梯度槽，与上一次无关（§5.1） |
| 优化步 | 一次参数更新；由 `update!` 划定边界（§3.5） |
| 逻辑视图 / 物理视图 | clvt 对布尔张量的两层读取语义（`T/NIL` vs `1/0`，§17.8） |
| a-prev | dense 反向里的左操作数，即 `input-cache` 的 cdr |
| 干跑（dry-run） | `build-model` 触发的试探性 forward，之后必须 `clear-step-caches!` |
