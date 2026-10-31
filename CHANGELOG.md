# 变更记录

本文件记录 clnn 的缺陷修复、性能改进与行为变更。面向使用者的用法说明见
README.md；本文件面向维护者，重点是「现象 → 成因 → 修复 → 量化效果」。

格式参考 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.1.0/)，
版本号遵循语义化版本（当前 `clnn.asd` 中为 `0.0.1`）。

---

## [未发布]

### 2026-11-01 · 三处静默失败：损失 target、优化步步号、加载后训练态

这一版没有新功能。修的是三个「不报错、数值上讲得通、效果悄悄变差」的洞 ——
它们的共同点是**违反时没有任何信号**，只能靠事后对拍或读源码才发现。

#### 1. 损失函数的 target 不过闸门

`compute-loss` / `compute-loss-gradient` 的 `:around` 只规约 `predicted`，
`target` 原样透传。对 CE 是对的（整数标签属索引域），但 mse / bce / huber /
cosine / kl 的 target 是**数值目标**，f32 目标与 f64 logits 混跑 —— 实测 mse
损失 160 → 765 ms（**4.8×**）、梯度 117 → 774 ms（**6.6×**），数值仍然正确，
所以不会有任何报错。

修法：target 也过 `ensure-compute-dtype`。它本身按域分流（整数与 `:bool` 原样
放行，只折 `:float32`），所以不需要按损失类型分派。测试 J1–J6。

#### 2. 优化步步号按 `optimizer-step` 调用次数计

`step-count` 的语义是「执行了多少个参数更新步」，实现却是「`optimizer-step` 被调
了几次」。整模型一次 `update!` 与逐层 N 次 `update!` 因此差 N 倍：

- Adam 的偏差校正 `1/(1−β^t)` 一路偏小 → 有效学习率偏低；
- Adagrad 的 `1/(1+λt)` 同理；SGD / RMSprop 的计数则一直是 0；
- 梯度裁剪每次按该层范数各剪一次，而不是全模型全局范数。

修法：步号所有权上移 —— `update!` 授予（动态变量 `*optimizer-step*`），
`optimizer-step` 只读。新增 `with-optimizer-step` 支持「分片更新算一步」；
`model-update!` 收敛为 `update!` 的别名。测试 K1–K7，其中 K7 用恒定梯度验证
第 1 步位移 ≈ lr（实得 0.0999999 ≈ 0.1）。

#### 3. `load-model` 不切推理态

`*TRAINING-MODE*` 默认 `T`，加载后 dropout 仍然生效、BN 用 batch 统计量 ——
save → load → predict 这个最常见流程给出随机结果，同一输入两次 `forward` 不同。

修法：`load-model` 末尾 `set-model-training! nil`；新增 `train!` / `eval!` 两个
单层入口（都返回 model 便于串联）。测试 L1–L5。

#### 4. examples 整理

| 文件 | 处理 | 行数 |
|---|---|---|
| `demo-last-bit-final.lisp` → `demo-last-bit.lisp` | 同一示例的三次迭代拼接（`take-last` 类定义三遍、加载即训练），重写为单一版本 | 429 → 154 |
| `resnet9-cifar.lisp` | 删除重复的前一份定义（10 个 duplicate definition 警告） | 623 → 323 |
| `cartpole-reinforce.lisp` | 删除 PPO v1 段（v3 覆盖同任务且带 KL 早停） | 2287 → 1953 |
| `conv2d-batched.lisp` | 末尾的 `(bench-conv2d-suite)` 改为注释 —— 加载示例不该自动跑基准 | — |
| 全部 10 个示例 | 17 处 `(optimizer-step opt (params m) (grads m))` 改为 `(update! m opt)`；`prove.lisp` 的「两子网络共用一个优化器」改用 `with-optimizer-step` 示范 | — |

10 个文件全部 `compile-file` 通过，**0 warning**（此前 4 个文件共 21 条）。

#### 5. 文档

契约文档 v2.2 → v2.3，1774 → 1584 行：

- **新增**：§3.5（优化步的边界）、§5.7（损失 target 的域）、§13.2（加载后的状态）
- **删除历史叙述**：§0.3 的三次翻案表、§17.4 的 F-1…F-23 修复项、§17.5/§17.6
  复跑记录、D-P1…D-P31 探针表。§17 从「按版本累积的日志」改为「验收探针」
- **修正三处与代码不符**：§3.3 的「累加层族」（早已统一为覆盖写）、§7.4 的
  「conv 显式拒绝 f32」（闸门会规约）、警告文案里的「16M 元素 ≈ 355 ms」
  （实测 46 ms，据此改为「1600 万元素约 50 ms」）

#### 测试

新增 `tests/test-runtime-contract.lisp`（26 断言，J/K/L 三组），
全量 18 个套件全绿，0 失败。

### 2026-10-31 · 单一计算域：把 float32 从「精度选项」降级为「被规约的输入」

动机：clnn 此前是「f32/f64 共存」的。但这条路在纯手写求导架构里站不住 —— 上一版
（v2.1）为了修 G-11，动了 `%rnn-zeros`、`value-float-dtype`、embedding `dw`、bn
running 统计量四类共二十多处，才勉强让 f32 「全链保住」；而下一次改动又要重来一遍。
本版把问题取消掉：**clnn 只有一个计算精度**。

#### 1. 决定与依据

`+COMPUTE-DTYPE+ = :float64`。参数、梯度、前向缓存、激活、损失、优化器状态、全部
内部缓冲恒为 f64。f32 不被拒绝，但会在按层的边界被规约一次，之后的任何中间量都不会
再出现窄 dtype。

两条依据：

- **架构**：clnn 没有 autograd，精度传播只能靠人肉追踪几十处缓冲（状态 8、累加器 12、
  running 统计量 2、零 dX 14、掩码/目标分布 5+）。多精度的隐性成本不是写代码，
  是每次改动都要重新证明这几十处仍然一致。
- **实测**（4000×4000，20 次）：f64⊔f64 852 ms、f32⊔f32 584 ms、**f32⊔f64 7165 ms**。
  慢的不是 f32（纯 f32 反而快 30%，带宽减半），是**混着跑**——每个算子都要定结果
  dtype、分配 f64 临时、逐元素转换。而 clnn 旧写法恰是最容易混的状态：参数 f64
  （初始化器决定）+ 输入可能是 f32（数据源决定）。另外 clvt 唯一的 AVX-512 GEMM
  微内核只发射 float64 指令。

#### 2. 实现：一个闸门

| 位置 | 机制 |
|---|---|
| 层的 `forward` / `backward` | `(defmethod forward :around ((l layer) input))` —— 一处覆盖全部 20 个层 |
| 损失 | `compute-loss` / `compute-loss-gradient` 的 `:around`，只规约 `predicted` |
| 初始化器 | `init-weight` / `init-bias` 的 `:around` —— 一处覆盖全部 10 个初始化器 |
| 外部导入 | 新增 `normalize-params!`；`load-model` 内部自动调用 |

用 `:around` 而不是在每个 `defmethod` 里加一行，是因为层有 20 个 × 两个入口 = 40 处，
逐处加必然漏一处，而漏掉的那处就是静默的混合精度。

新增/改动的函数（`nn-protocol.lisp`）：

| 名称 | 作用 |
|---|---|
| `+compute-dtype+` | 唯一的计算域定义 |
| `ensure-compute-dtype` | 规约单个张量；整数/bool/标量原样透传 |
| `ensure-compute-input` | 多输入版本（SDPA/MHA 的 Q/K/V 三元 list） |
| `compute-dtype-p` | 「是否已是计算域」 |
| `normalize-params!` | 外部权重的归一通道，返回改动个数 |
| `value-float-dtype` | **恒返回** `+compute-dtype+`（原为「f32 原样保留」） |

首遇 f32 时 `warn` 一次（每次会话一条，不刷屏），文案给出「请在数据管道里转换一次
后复用」的建议。**去重键是 dtype 本身，不含层名** —— 初版写成 `(dtype . 调用方)`，
结果是 20 层模型出 20 条 5 行警告，与「每次会话一条」的承诺不符；测试 I13b 现在
钉住这一点。

#### 3. 连带简化

单一计算域让上一版的一半补丁退化为多余防御：

| v2.1 的做法 | v2.2 |
|---|---|
| `%rnn-zeros` 按参照张量选 dtype | 退化为 `(vt-zeros shape :dtype +compute-dtype+)`，即普通 `vt-zeros` |
| embedding `dw` 显式传 `:dtype` | 不需要，缺省即计算域 |
| bn running 统计量惰性对齐 `gamma` | 不需要，它们本就是 f64 |
| `%ce-target` 按 `(vt-dtype predicted)` 造 | 恒等（logits 已被闸门规约） |
| `%ensure-float64-data` 对用户报「请先 astype」 | 改为**内部哨兵**：正常使用不可能触发；触发即说明某条路径绕过闸门，是框架 bug |

**哨兵刻意硬编码 `:float64`**，不跟 `+COMPUTE-DTYPE+` 走 —— 它守的是内核的物理能力
（`(SIMPLE-ARRAY DOUBLE-FLOAT (*))` 声明），不是框架的策略。若让它跟着常量走，改常量
就会放行 f32 并读到错误的位模式（静默错值）。

#### 4. 行为变更（用户可见）

| 场景 | v2.1 | v2.2 |
|---|---|---|
| f32 输入喂给 `conv2d`/`max-pool2d`/`avg-pool2d` | **报错**「仅支持 float64」 | 正常计算，输出 f64 |
| f32 输入喂给其它层 | 部分保持 f32、部分变 f64 | 全部规约到 f64 |
| `make-dense` 等构造器 | 参数永远 f64（初始化器决定） | 不变（但现在是设计意图，不是巧合） |
| 手工 `setf` f32 权重 | 后续 forward 每层逐次转换 | 同上；请调 `normalize-params!` 一次 |
| `update!` / 优化器 | 状态继承参数 dtype | 恒 f64 |

#### 5. 测试

| 文件 | 变化 |
|---|---|
| `tests/dtype-contract-test.lisp` | 70 → **77** 断言。**I 组全部反转**：原问「f32 能否全链保住」（I1–I11），现问「f32 是否被规约 + 索引/掩码是否没被误折 + 外部权重有没有归一通道」（I1–I13，I13 拆 a/b/c：同层只报一次 / 跨层也只报一次 / 清空记录可再报）。C 组的 `(:float32 :float32)` 改为 `(:float32 :float64)`；D 组从「跟随输入 dtype」改为「参数恒 f64」；F1 改为「零梯度为计算域」 |
| `tests/test-conv-im2col.lisp` | 第 6 节从「非 float64 必须被拒绝」改为「**层归约、内部内核哨兵拒绝**」两半：三个层断言「接受 f32 输入并输出 f64」，`im2col`/`col2im` 直调仍拒绝（它们不是层方法，不经闸门）。31 PASS / 0 FAIL |
| 其余 15 个套件 | 无需改动，全部通过 |

全量：**17 个套件全绿**，含 `gradcheck-all-layers`（全层有限差分）与真实 MNIST 收敛。

#### 6. 文档

《框架设计契约与接口契约文档》v2.1 → **v2.2**：

- **新增 §2.0**「单一计算域」：决定、两条依据、闸门机制表、`predicted`/`target` 的
  区别、三种方案对比、入口成本可以怎么消掉。
- **§2.1** 从「五个类型域」改为「三个可用域 + 两个禁区」，`float32` 明确不在任何域里。
- **§2.2** D1/D2/D3 从「提升结果」改为「恒为 `+COMPUTE-DTYPE+`」；说明提升表在 clnn
  热路径上退化为恒等。
- **§2.3** 从「f32 传播矩阵（26 行）」整节改为「各层的域归属与转换点」。
- **§2.6** 从「float32 支持等级」改为「计算域逐层验收」；删掉 §2.6.1 的全 f32 构造
  配方，换成「接入 f32 数据源的正确姿势」与「为什么不再提供全 f32 模型」。
- **§2.8** G-3 / G-7 / G-11 关闭（**不是修好，是前提消失**），新增 G-12（混跑 8.4×，
  已由本版消除）。
- **§7.4 / §16** 同步；**§0.3** 记录 v2.2 对 v2.1 的第三次翻案，并说明它与前两次的
  性质区别（前两次是「实测错了」，这次是「问题被取消了」）。
- **§17 实证记录改版**：新增读法说明（这一章是按版本累积的日志，不是现行规范）；
  §17.2 的 D-P1…D-P31 明确标注作废，新增 **§17.2.1**「闸门探针」15 条替换之；
  §17.4 按版本重排为 v2.2 → v2.1 → v2.0，新增 F-15…F-23；
  补上缺失的 §17.5 编号，并新增 **§17.6 v2.2 复跑记录**（含 float64 不变性四项证据）。
- **§1.5.1 修正一处失实**：原写「这不是暂时没做自动微分，而是架构选择」，但 git 历史
  `dd0ea79` 明写「删除 nn-autograd.lisp（clvt 无法完全支持 diff-tensor 自动微分）」。
  准确说法是**「试过、做不成、删掉了」**，不是主动选择。这条必须改：前者是偏好，
  后者要交代原因。

---

### 2026-10-31 · 纯手写求导架构注明 + G-11 收尾 + 梯度语义统一

动机：clvt 合并了弱标量补丁（`CONVENTIONS` §16.31.7：把剩下十一个「交换操作数
顺序就换 dtype」的算子补齐，二元提升恢复可交换）。clnn 的缺口 G-11 有两半 ——
上游的 `vt--` 不对称与 clnn 自己的 f64 零张量起点 —— 前者已由上游消除，后者
只能在这里修。顺手把「clnn 是纯手写求导」这个一直只写在 README 一句话里的前提
**上升为可审查的契约**，并据此修掉三处与该前提冲突的既有行为。

#### 1. 新增契约文档 §1.5「纯手写求导架构」（M1–M7）

clvt 是纯张量库：没有计算图、没有 `grad_fn`、没有磁带、没有伴随模式、没有对偶数。
它唯一的 `vt-gradient` 是**数值**梯度（对标 `numpy.gradient`），不能代替解析导数。
所以 clnn 的每一条反向公式都是人推手写的 —— 这不是「暂时没做自动微分」，而是架构
选择（个人研究级别实现，用可读/可预测/可逐位对拍换动态图灵活性）。

§1.5 把由此**必然推出**的约束写成七条协定，并给出「违反的后果」（都是静默算错）：

| 编号 | 协定 |
|---|---|
| M1 | 每层必须自缓存；`backward` 不得重跑前向 |
| M2 | 缓存是单次前向的快照；`backward` 只读不写 |
| M3 | `forward` 与 `backward` 一一配对 |
| M4 | 梯度槽语义统一为**覆盖写**（本次反向的梯度，不跨调用叠加） |
| M5 | 容器只调度，不夹带自己的数学 |
| M6 | 只提供一阶解析导；数值差分只在 `tests/` 里做对拍 |
| M7 | 参数更新必须显式两步：`grads ← backward`、`params ← optimizer-step`/`update!` |

配套给出「与 autograd 的责任划分表」与「扩展者自查清单（8 条）」，并把 M1/M3/M4/
M5/M7 落成新测试 `tests/manual-backward-contract-test.lisp`。

#### 2. G-11 根因②：状态/累加缓冲不再从 f64 起

修掉上游之后，rnn 四支仍全 f64 —— 因为状态缓冲写的是 `(vt-zeros (list batch hs))`，
缺省 f64，一旦参与 `(vt-+ 累加器 增量)` 就只升不降。新增 `%rnn-zeros (shape like)`
（按参照张量的浮点 dtype 造零），把 rnn-cell / lstm / gru / rnn-sequence 的状态、
累加器、`grad-input`、`dh/dc-next` 共 14 处改过来。

**参照量必须是真正的输入**（`x` 或 `grad-output`），不能是某个 f64 累加器 ——
继承后者等于没改。这个坑 v2.0 时代已经踩过（当时写死「rnn 的 8 处改了也是自欺」）。

同类修法还覆盖三处：

| 位置 | 现象 | 修法 |
|---|---|---|
| `embedding` 的 `dw` | 从 `(vt-zeros (list ne ed))` 起，恒 f64 | 按权重 dtype 造；`dX` 跟随 `grad-output` |
| `batch-norm` 的 `running-mean/var` | buffer 不在 `params` 里，astype 参数时不会跟着变；推理态 `(vt-- input rm)` 把 f32 输入拉成 f64 | 惰性对齐 `gamma`（无 affine 时对齐输入）的 dtype |
| `ce-loss` 的 `%ce-target` | 硬编码 `(make-array … :element-type 'double-float)`，梯度 `softmax − target` 恒 f64（G-7） | 接受 `dtype`，按 `logits` 造；常数先按 double 算再降型 |

效果（参数与输入全 f32）：

| 层 | 修复前 | 修复后 |
|---|---|---|
| `dense`（11 种激活） | `:tanh` 的 dX/dw 为 f64 | 全 f32 |
| `activation-layer`(`:tanh`/`:sigmoid`) | dX f64 | f32 |
| `batch-norm` / `layer-norm` | 前向即 f64 | 全 f32（含推理态） |
| `rnn-cell` / `lstm` / `gru` / `rnn-sequence` | 全 f64 | 全 f32 |
| `embedding` | `dw` f64 | f32 |
| `ce-loss` | 损失与 dP 均 f64 | f32 |
| `mha` / `transformer-block` / `residual` / `sdpa` / 其余损失 | 已是 f32 | 不变 |

`conv2d` / `max-pool2d` / `avg-pool2d` 仍只支持 f64（内核绑定 `double-float` 数组），
**遇 f32 显式报错**，不静默转换 —— 这是能力边界，不是缺陷。

#### 3. **行为变更**：梯度槽统一为覆盖写

这是本次唯一的**用户可见语义变化**，请留意。

修复前：`dense`/`conv2d`/norm/attention 是覆盖写，而 `lstm`/`gru`/`rnn-cell`/
`embedding` 是**跨调用累加**。于是「忘记 `zero-grad!`」在不同层上后果不同：

```
dense：梯度不变（覆盖写帮忙清了）
lstm ：梯度翻倍
```

同一张网络里一半层对、一半层错，观测到的只是「loss 降得比预期慢一点」——
比全错更难查。在纯手写架构里没有 autograd 来管梯度生命周期，所以这个语义必须由层
自己钉死，且必须一致（M4）。

修复后：**全部层族覆盖写**。

| 层 | 改法 |
|---|---|
| `lstm` / `gru` | 局部累加器 `dwih-acc` 沿时间步累加，最后 `(setf 槽 acc)`（原为 `(vt-+ 旧值 acc)`） |
| `embedding` | 复用旧槽但先 `(vt-fill dw 0)` 就地清零（省一次 `ne×ed` 分配），仍逐行 `+=` |
| `rnn-cell` | 默认覆盖写；被 `rnn-sequence` 复用时由后者绑定 `*rnn-accumulate-grads*` 并在循环前清空 cell 的槽 —— 跨时间步累加收进层内部 |

影响面：
- **首次 backward 的数值逐位不变**（`gradcheck-all-layers` 全层有限差分对拍 0 失败）；
- 连续两次 forward+backward 的梯度比值从 2.000 变为 1.000（10 个层族逐个验证）；
- 原先依赖「不清零就累加」做梯度累积的代码会失效 —— 契约文档 §7.1 给出了三种替代
  方案，推荐「在训练循环里自己攒」或「加一层累积容器」。

#### 4. `update!` 不再静默无效

`update!` 在 `nn-package.lisp` 里是导出符号，但库里只有 `(defmethod update! ((c t) opt))`
这个**空方法** —— 用户调 `(update! 某层 opt)` 会什么都不发生，且不报错。补上 `layer`
方法（`(optimizer-step opt (params c) (grads c))`，与 `model-update!` 等价但作用域可任意）。
必须放在 `optimizer-step` 的定义之后，否则编译期会报未定义函数。

#### 5. 测试

| 文件 | 变化 |
|---|---|
| `tests/dtype-contract-test.lisp` | 46 → **70** 断言：I 组从「记录缺陷」改为「验收全链 f32」（I1–I11，含 bn 推理态、rnn 四支、embedding、ce）；A 组新增 A11 弱标量对称性（8 个算子 + 2 条对照，判据 `a ⊔ b == b ⊔ a`） |
| `tests/manual-backward-contract-test.lisp` | **新增**，16 断言：M1（行为级「改参数后仍用旧激活」+ 反证 + 21 个 `backward` 方法体无 `forward` 调用的文本扫描）、M4（10 层族两次 backward 逐位一致）、M5（sequential dX 与手工倒序链对拍）、M7（`update!` 真的改参数） |
| `tests/contract-probes.lisp` | T4 语义反转：从「rnn-cell 跨调用累加」改为「同输入两次 backward 逐位相同」（原判据用随机输入比大小，本身就是弱断言） |
| `tests/pdf-bugs-verify.lisp` | BUG-08 从「不清零时梯度约 4 倍」改为「覆盖写，ratio ≈ 1」 |
| `tests/audit-verify.lisp` | 「LSTM BPTT 梯度累积（4x）」改为「不跨调用累积（1x）」 |
| `tests/run-tests.sh` | `FAST_TESTS` 加入 `manual-backward-contract-test`（15 个快速文件） |

全量：**17 个套件全绿**（含 `gradcheck-all-layers` 与真实 MNIST 的 `smoke-mnist-convergence`）。

#### 6. 文档

- 《框架设计契约与接口契约文档》v2.0 → **v2.1**：新增 §1.5；§2.2/§2.3/§2.6/§2.8、
  §5.1/§5.2、§6.5、§7.1/§7.3/§7.4、§8、§16、§17 全部按新结论改写；
  §0.3 记录 v2.1 对 v2.0 两处结论的**翻案**（f32「局部可用」→「整网可用」；
  「rnn/embedding 累加」→「全层覆盖写」）。
- `README.md`：设计目标里加「纯手写求导」一条；「已知限制」与 FAQ 的说明补全为
  「clvt 无 autograd」的因果链。

---

### 2026-10-31 · dtype 契约加固：布尔、复数与 float32

动机：clvt 0.4.20 把逻辑 dtype 扩到 11 种，新增 `:bool`（物理存储
`(unsigned-byte 8)`、只含 0/1）与 `:complex128` / `:complex64`。clnn 的层代码
写于「dtype 只有 float32/float64/int」的年代，其中若干处靠 `vt-map` /
`vt-zeros` 的**缺省 dtype** 而非显式声明来定类型。clvt 改缺省行为之后，这些
位置不再报错，而是静默产出错误数值或错误类型 —— 这是本次加固的主因。

审查方法：对 clnn 全部 107 个 `vt-*` 调用做符号审计（94 个是 clvt
`:external` 且 `fboundp`，13 个是 clnn 自有），逐类对照 clvt
`CONVENTIONS.md` §16（复数）/ §17（布尔）核验语义，再用探针脚本在真实张量上
复现。结论与缺口登记在《clnn-框架设计契约与接口契约文档》§2.4–2.8 与 §16，
本文只记改动本身。

#### 1. `vt-zeros` → `vt-zeros-like`：优化器状态与梯度累加器继承参数 dtype

`vt-zeros` 的缺省 dtype 是 `:float64`，与「按参数形状造一个同 dtype 的零张量」
这个意图不一致。凡是拿它当「某张量的同类零值」用的地方，float32 模型会在第一次
`optimizer-step` 里被静默升到 float64（参数本身不变，但状态张量变宽，反向传播
的 dtype 断言与 `assert-param-grad-alignment` 都看不出来）。

改为 `vt-zeros-like` 后 dtype 严格继承参照张量，共 14 处：

| 位置 | 处数 | 参照张量 |
|---|---:|---|
| `nn-optimizer.lisp` | 10 | 各优化器的动量 / 一阶矩 / 二阶矩 / `vmax` / adagrad 累加器 / rmsprop 缓冲 |
| `nn-layers-norm.lisp` | 2 | dropout 的 `p >= 1` 分支输入 |
| `nn-layers-dense.lisp` | 1 | residual backward 的 `grad-output` |
| `nn-protocol.lisp` | 1 | `stop-gradient-node` backward 的 `grad` |

**刻意未改的 8 处**：`nn-layers-rnn.lisp` 的 `dwih-acc` / `dwhh-acc` / `dbih-acc` /
`dbhh-acc`（LSTM 与 GRU 各 4）。它们写成 `(or (lstm-dweight-ih l) (vt-zeros (vt-shape dwih-acc)))`，
看着正是「继承参照张量」，但那个参照张量 `dwih-acc` 本身由 `(vt-zeros (list …))`
创建、恒为 `:float64` —— 改成 `vt-zeros-like` 得到的还是 f64，属于「改了个寂寞」。
真正要动的是累加器的**起点**，以及恒等式的另一边，见下面第 7 节。

#### 2. dropout 掩码 dtype 显式钉成浮点（两处）

`vt-map` 在 `:dtype` 缺省时**按输入提升**。dropout 的掩码 lambda 返回
`(if (< (random 1.0d0) p) 0.0d0 scale)`，其中 `scale = 1/(1-p)`：

* 整数输入 → 整数掩码，`1.428…` 被**截断为 1**，inverted dropout 退化为普通
  dropout（训练时静默少缩放）；
* `:bool` 输入 → 布尔掩码，`vt-*` 随之改派逻辑与，保留位从 `1/(1-p)` 塌成 1，
  且 `vt-multiply(mask, x)` 变成逻辑与。

现在统一传 `:dtype (value-float-dtype (vt-dtype input))`（`nn-layers-norm.lisp`
的 dropout 层、`nn-layers-attention.lisp` 的 sdpa dropout 各一处）。
`value-float-dtype` 把任何窄于 float32 的 dtype 抬到 `:float32`、其余抬到
`:float64`，保证掩码始终是可乘的实数。

#### 3. 新增 `assert-index-dtype`：索引 / 标签必须是整数

clvt 的 `vt-take` 对 `:bool` 索引不报错 —— 它把 0/1 当成合法下标，于是布尔掩码
误传进来会**静默取到第 1 行**；浮点索引则要进到 `vt-take` 内部才报 clvt 口径
的错误，文案不指向调用方。新建守卫（`nn-protocol.lisp`）在入口拦下非整数
dtype，并在错误信息里给出两条正确转换路径：

```lisp
(first (vt-nonzero mask))   ; 1-D 掩码 → 整数索引
(vt-extract mask x)         ; 展平语义取值
```

挂载点共 5 处：`embedding` 前向、`ce-loss` 标签、`tensor-one-hot`、
`tensor-gather`、`tensor-scatter`。`nn-loss.lisp` 里另外补了注释记录历史失败
点：旧实现一路走到 `(coerce (vt-ref …) 'fixnum)` 才炸，浮点标签报
`0.5d0 can't be converted to type FIXNUM`、布尔标签报
`T can't be converted to type FIXNUM`、复数标签报
`The value #C(1 0) is not of type REAL` —— 三处都在写了一半目标分布**之后**才
失败，留下半成品张量。

#### 4. `tensor-masked-fill` / `tensor-where` 收下布尔条件

两者的条件判定原来是 `(> mi 0.0d0)`，只对浮点掩码成立。clvt 引入 `:bool` 后，
比较函数（`vt->` / `vt-=` 等）缺省返回 `:bool`，`vt-map` 交给 lambda 的是**裸
字节 0/1**；若日后逻辑视图透到 `vt-map`，则会是 `T/NIL`，`(> 'T 0.0d0)` 当场抛
`T is not of type NUMBER`。新增 `%mask-true-p` 把三种表示（裸 0/1、`T/NIL`、
浮点）统一收为「非零为真」，并把这条语义写进 docstring：掩码只作**条件**消费，
需要乘性掩码请先 `vt-astype` 到浮点（布尔真⊗真 = 真，不是 1）。

#### 5. 新增 `tests/dtype-contract-test.lisp`（48 条断言）

按契约文档的章节分组，每组都能独立定位：

| 组 | 内容 |
|---|---|
| A | clvt 基线：11 种 dtype、`:bool` 存储与不变量 B1、`bool⊗bool` 逻辑与 |
| B | 掩码边界：`:bool` / 整数 / 浮点三种掩码喂给 `tensor-masked-fill` 与 `tensor-where` |
| C | dropout 掩码 dtype（含 `p >= 1` 与 `p = 0` 两个退化点） |
| D | 优化器状态 dtype（8 个优化器 × float32/float64） |
| E | 索引 / 标签守卫（5 个挂载点，浮点与布尔各一条） |
| F | 零张量 dtype（`vt-zeros-like` 在全部 11 种 dtype 上的继承） |
| G | 序列化往返不改变 dtype |
| H | 复数边界（`vt-relu` / `vt-sigmoid` 显式拒绝；`vt-matmul` / `vt-sum` 静默接受） |

接入 `tests/run-tests.sh` 的 `FAST_TESTS`。当前全套 **14 个快速文件 / 20 s / 0 FAIL**
（另有 `gradcheck-all-layers` 与 `smoke-mnist-convergence` 两个慢速套件，亦 0 失败）。

#### 6. 文档：《框架设计契约与接口契约文档》v1.0 → v2.0

原文档 519 行、写于 clvt 引入 `:bool` 之前。重写后 1278 行，新增四块此前完全
缺失的内容：

* **§2 类型契约**：五域划分（整型 / 浮点 / 布尔 / 复数 / 未知）、D1–D5 铁律、
  **实测**的 float32 传播矩阵、布尔的三类可用位置与三类禁止位置、复数契约、
  float32 支持等级（conv2d / max-pool2d / avg-pool2d **显式拒绝** f32）、
  缺口清单 **G-1…G-10**；
* **§6 `vt-*` 接口契约**：按用途分 10 类逐条列出前置条件与返回 dtype，末尾是
  **十二条禁令**与调用频次统计；
* **§16 clvt 语义变更跟踪矩阵**：13 行变更 × 5 步跟踪流程，供 clvt 下次改动时
  逐行核对；
* **§17 实证记录**：T1–T8/T15 的复核结果、各组探针表、符号审计明细、本次
  6 项修复（F-1…F-6）的对照表。

同时纠正了 v1.0 的四处失实描述（§0.3 有完整清单）：把「所有层输入输出 dtype
相同」改为「输入 ⊔ 参数的提升」；`vt-nonzero` 的返回是**逐轴索引列表**而非单个
索引张量；`nn::vt-relu-derivative` 等是 clnn 私有 defun 而非 clvt 公开 API
（登记为 G-10）；v1.0 声称的实证来源 `scripts/clnn-contract-smoke.lisp` 实际
不存在。

#### 7. float32 只在「窄路径」上成立（新增缺口 G-11）

> **本条已被 2026-10-31 的「G-11 收尾」取代**：clvt 修掉弱标量不对称、clnn 修掉
> f64 缓冲起点之后，f32 已整网可用（conv/pool 除外）。以下保留原文作为历史记录。

写文档时按「前向能跑」判断了 f32 支持等级，复核反向时发现结论不成立，遂把
**前向**与**反向**拆开重测（参数与输入全造 f32）：

| 层 | 前向 | dX | 参数梯度 |
|---|---|---|---|
| `dense(:none)` / `dense(:relu)` | f32 | f32 | f32 |
| `dense(:tanh)` | f32 | **f64** | **f64** |
| `activation-layer(:relu)` | f32 | f32 | — |
| `activation-layer(:tanh/:sigmoid)` | f32 | **f64** | — |
| `flatten` / `dropout` / `sdpa` / `mse` | f32 | f32 | — |
| `embedding` | f32 | f32（恒零） | **f64** |
| `batch-norm` / `layer-norm` | **f64** | **f64** | **f64** |
| `rnn-cell` / `lstm` / `gru` | **f64** | **f64** | **f64** |

根因只有一条，且在 clvt 侧：

```lisp
(vt-dtype (vt-- 1.0d0 x-f32))   ;=> :float64   ← 唯一的不对称
(vt-dtype (vt-- x-f32 1.0d0))   ;=> :float32
(vt-dtype (vt-+ 1.0d0 x-f32))   ;=> :float32
(vt-dtype (vt-* 2.0d0 x-f32))   ;=> :float32
```

`vt--` 把**首个**操作数无条件 `ensure-vt` 成 0 维张量（`src/elementwise.lisp`），
于是「double 标量 − f32 张量」走了强提升；其余算子把标量留在参数位、走 numpy 式
弱提升。而 `1 − x` 正是激活导数、LayerNorm 的 `1 − xhat²`、LSTM 门控的标准写法，
所以凡含「1 减」的层全部被这一条拖回 f64，形成「前向 f32、梯度 f64」的静默混合
精度 —— 训练一步后参数又变回 f64。

clnn 侧不做逐点改写（要把每一处 `1.0d0` 换成保 dtype 的写法，且需重跑全部
gradcheck 常数），改为：`nn-layers-rnn.lisp` 头部写明本文件为 float64-only、
契约文档 §2.6 给出实测矩阵与两种保 dtype 的替代写法、缺口 G-11 登记为上游问题。
测试侧新增 `dtype-contract-test` 的 I 组（当时 8 条断言，v2.1 起扩为 13 条）：既钉住「窄路径必须一直是
f32」（I4），也把当前的 f64 漂移标成 `[记录]`（I2/I5b/I6/I7）—— 谁修好了 G-11，
这几条会失败并提醒同步文档。

#### 8. 明确「不支持」而非「待修」的四项

审查中发现下列行为在复数 / 布尔输入下不正确，但改动面覆盖整个张量层，本版
**只声明契约、不改实现**，全部登记为缺口（详见契约文档 §2.8）：

| 缺口 | 现象 | 本版处置 |
|---|---|---|
| G-4 | 复数在 dense / conv / 损失 / 梯度范数上静默流通（`vt-astype complex→float64` 静默丢虚部） | 契约声明**不支持复数**；不加运行时守卫 |
| G-7 | `ce-loss` 恒返回 float64（`%ce-target` 内固定 `double-float`） | 记录；f32 训练时 loss 比 logits 宽一档 |
| G-8 | `vt-concatenate` 拒绝布尔（clvt 侧限制） | 记录；布尔拼接需先转整型 |
| G-9 | 31 处 `vt-map` 未显式 `:dtype` | 浮点输入下正确；逐处标注「输入若可为整数/布尔则失效」 |

#### 回归验证

* 全套 **14** 个测试文件、`--fast` 模式：**0 FAIL**（改动前 12 个文件 221 条
  断言亦为 0 FAIL；新增 `dtype-contract-test` 48 条、`contract-probes` 58 项）；
* 既有 float64 路径逐位等价：对同一输入的输出与改动前完全一致
  （`vt-zeros` 与 `vt-zeros-like` 在 float64 上是同一结果，`assert-index-dtype`
  对整数索引不触发，`%mask-true-p` 对浮点掩码与 `(> mi 0.0d0)` 等价）；
* 契约文档 §17 的 T1–T15 已固化成 `tests/contract-probes.lisp`（58 OK / 0 BAD），
  不再依赖一次性脚本。

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
* **不支持复数 dtype**：`dense` / `conv2d` / 损失函数 / 梯度范数都会让
  `:complex128`、`:complex64` 张量静默流通，`vt-astype` 降 float 时静默丢
  虚部。这是**契约层面的不支持**，不是待修缺陷，故不加运行时守卫（加守卫要
  覆盖整个张量层）。详见契约文档 §2.5 与 §2.8 的缺口 G-4。
* `ce-loss` 恒返回 float64，与 logits 的 dtype 无关（`%ce-target` 内固定
  `double-float`，缺口 G-7）。float32 训练时 loss 比模型宽一档。
* `conv2d` / `max-pool2d` / `avg-pool2d` **显式拒绝** float32（这三个核直接
  按 `(simple-array double-float (*))` 声明访问底层数组，SBCL 会信任声明并读
  错位模式，因此选择报错而非静默错值）。f32 模型不能带卷积层。
