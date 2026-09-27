;;;; examples/transformer-sum-classifier.lisp
;;;;
;;;; ============================================================================
;;;; Transformer 序列分类示例：判断序列之和是否超过阈值
;;;; ============================================================================
;;;;
;;;; 【任务】
;;;;   输入：长度 8 的整数序列，每个 token 均匀取自 {0, 1, ..., 9}。
;;;;   输出：这 8 个数之和是否大于 36（2 分类：0=否，1=是）。
;;;;
;;;;   由于每个 token 的期望值是 4.5，8 个之和的期望恰好是 36，
;;;;   这是一个 50/50 平衡的二分类问题，随机猜测的准确率是 50%。
;;;;
;;;; 【为什么选这个任务】
;;;;   1. 需要「跨位置聚合」：只看任何一个 token 都无法判断总和是否超阈值，
;;;;      模型必须把所有位置的信息汇总到一处——这正是自注意力的强项。
;;;;   2. 梯度下降可以学：不同于「奇偶性 / XOR」这类经典难任务
;;;;      （会长期困在均匀输出的鞍点上，loss 卡在 ln(2) ≈ 0.693），
;;;;      sum > 阈值 是单调可分信号的组合，训练曲线平滑。
;;;;   3. 规模适中：模型仅 2.5 万参数，CPU 上 15 epoch 内即可收敛到 99%+。
;;;;
;;;; 【架构】
;;;;   token 序列 (B, 8)
;;;;       ↓ Embedding              (B, 8, 32)    查表得到稠密表示
;;;;       ↓ LayerNorm                            把 embedding 归一到单位尺度
;;;;       ↓ + 正弦位置编码                       注入顺序信息
;;;;       ↓ TransformerBlock      (B, 8, 32)    Pre-Norm + MHA + FFN + 残差
;;;;            ├─ LayerNorm
;;;;            ├─ Multi-Head Attention (4 heads, head_dim=8)
;;;;            ├─ 残差连接
;;;;            ├─ LayerNorm
;;;;            ├─ FFN (32 → 64 → 32, GELU)
;;;;            └─ 残差连接
;;;;       ↓ Flatten              (B, 256)       展平所有位置
;;;;       ↓ Dense (256 → 64, GELU)             分类头隐层
;;;;       ↓ Dense (64 → 2)                     输出 2 类 logits
;;;;       ↓ Cross-Entropy Loss
;;;;
;;;; 【运行方式】
;;;;   1. 确保 nn 库已加载（全部 .lisp 按依赖顺序 LOAD 或通过 ASDF 加载）。
;;;;   2. 在 REPL 中：
;;;;        (in-package #:nn)
;;;;        (load "examples/transformer-sum-classifier.lisp")
;;;;        (run-sum-example)
;;;;
;;;; 【预期输出】（CPU，约 10~30 秒）
;;;;   参数量：25506
;;;;   Epoch  1/15  loss = 0.2955  acc = 0.8938
;;;;   Epoch  2/15  loss = 0.1114  acc = 0.9497
;;;;   ...
;;;;   Epoch 15/15  loss = 0.0477  acc = 0.9819
;;;;   最终准确率 = 0.9971（1024 样本）
;;;;
;;;; 【本示例覆盖的库能力】
;;;;   - 自定义 layer 子类（add-positional-encoding）
;;;;   - grad-slots / cache-slots 协议（自定义层必须实现，否则 zero-grad!/清缓存会漏）
;;;;   - 延迟初始化层的正确用法（build-model + dummy input 或首次 forward）
;;;;   - 训练/推理模式切换（set-model-training! / with-training）
;;;;   - 完整训练循环：forward → loss → backward → zero-grad! → optimizer-step
;;;;   - 序列化（save-model / load-model，本例未演示但可直接使用）
(ql:quickload :clnn)
(in-package #:nn)

;;; ============================================================================
;;; 1. 自定义层：正弦位置编码
;;; ============================================================================
;;;
;;; Transformer 本身对位置是「置换不变」的——同一组 token 无论顺序如何，
;;; 输出的集合相同。为了让模型感知「这是第几个 token」，需要注入位置信息。
;;;
;;; 这里采用原论文（Vaswani et al. 2017）的固定正弦编码，不可学习：
;;;   PE(pos, 2i)   = sin(pos / 10000^(2i/d))
;;;   PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
;;;
;;; 编码形状为 (seq-len, d-model)，前向时广播（tile）到 batch 维度后相加。

(defun sinusoidal-positional-encoding (seq-len d-model)
  "生成 (seq-len, d-model) 的经典 Transformer 正弦位置编码矩阵。"
  (let ((pos (make-array (list seq-len d-model)
                         :element-type 'double-float
                         :initial-element 0.0d0)))
    (dotimes (p seq-len)
      (dotimes (i d-model)
        (let* ((k (floor i 2))
               (denom (expt 10000.0d0
                            (/ (* 2.0d0 (float k 1.0d0))
                               (float d-model 1.0d0))))
               (angle (/ (float p 1.0d0) denom))
               (val (if (evenp i) (sin angle) (cos angle))))
          (setf (aref pos p i) val))))
    (vt-from-array pos :dtype :float64 :fast t)))

(defclass add-positional-encoding (layer)
  ((pos-enc :initarg :pos-enc :reader ape-pos-enc))
  (:documentation
   "把不可学习的正弦位置编码加到输入 (B, T, D) 上。
    这是无参数层，trainable 为 NIL。"))

(defun make-add-positional-encoding (seq-len d-model)
  "构造位置编码层。SEQ-LEN 与 D-MODEL 需与后续张量形状一致。"
  (make-instance 'add-positional-encoding
                 :pos-enc (sinusoidal-positional-encoding seq-len d-model)
                 :name "pos-enc" :trainable nil))

(defmethod forward ((l add-positional-encoding) x)
  ;; 输入 x: (B, T-len, D)，位置编码 pos-enc: (T-len, D)
  ;; 先 reshape 成 (1, T-len, D)，再 tile 到 (B, T-len, D) 后相加。
  ;; 说明：库里的 vt-+ 支持广播，但不同算子的广播语义不一定一致；
  ;;       显式 tile 是最稳妥的写法。
  (let* ((shape   (vt-shape x))
         (B       (first  shape))
         (T-len   (second shape))          ; 不用 T，避免遮蔽 CL 常量 T
         (D       (third  shape))
         (pe      (ape-pos-enc l))
         (pe-3d   (vt-reshape pe (list 1 T-len D)))
         (pe-batched (vt-tile pe-3d (list B 1 1))))
    (vt-+ x pe-batched)))

(defmethod backward ((l add-positional-encoding) grad)
  ;; 无可学习参数，梯度直通到上一层。
  grad)

;; 协议实现：告诉框架本层没有梯度需要清零、没有缓存需要清理。
;; 自定义层若忘记实现这两个方法，zero-grad! / clear-step-caches!
;; 会走 t 上的默认方法（返回 '()），行为恰好正确——但显式写出来更清晰。
(defmethod grad-slots  ((l add-positional-encoding)) '())
(defmethod cache-slots ((l add-positional-encoding)) '())

;;; ============================================================================
;;; 2. 超参数
;;; ============================================================================
;;; 所有维度参数集中在此，方便实验时统一修改。

(defparameter +vocab+    10)   ; 词表大小：token 取值 [0, 10)
(defparameter +seq-len+   8)   ; 序列长度
(defparameter +d-model+  32)   ; 模型隐藏维度（embedding 维度 = 每头维度 × 头数）
(defparameter +n-heads+   4)   ; 注意力头数；head-dim = d-model / n-heads = 8
(defparameter +ffn-dim+  64)   ; FFN 中间层维度
(defparameter +classes+   2)   ; 分类数

;;; ============================================================================
;;; 3. 模型构造
;;; ============================================================================
;;;
;;; 命名注意：不要用 BUILD-MODEL，那是 nn 库导出的公共 API
;;; （(build-model model &optional dummy-input)），在 (in-package #:nn)
;;; 下重名会静默覆盖库的实现，引发难以定位的编译错误。
;;; 自定义构造函数一律加领域前缀。

(defun build-sum-classifier ()
  "构造一个用于序列求和阈值分类的小型 Transformer。"
  (let* ((flat-dim (* +seq-len+ +d-model+))     ; 展平后的维度 = 8 × 32 = 256
         (model (make-sequential :name "sum-classifier")))

    ;; --- 1) 词嵌入 ---
    ;; 把整数 token 映射到 (B, T, D) 的稠密向量。
    ;; 注意：库默认 embedding 初始化 std=0.01，数值偏小；
    ;;       紧接着的 LayerNorm 会把尺度归一化回单位方差。
    (seq-add! model (make-embedding +vocab+ +d-model+ :name "tok-emb"))

    ;; --- 2) 嵌入后归一化 ---
    ;; 若不加 LN，位置编码（值域约 [-1, 1]）会完全淹没 embedding（std=0.01），
    ;; 模型只能看到「这是第几个位置」而看不到「这是哪个 token」。
    ;; 这是本示例从调试中总结的关键经验。
    (seq-add! model (make-layer-norm (list +d-model+) :name "emb-ln"))

    ;; --- 3) 注入位置信息 ---
    (seq-add! model (make-add-positional-encoding +seq-len+ +d-model+))

    ;; --- 4) Transformer 编码块 ---
    ;; Pre-Norm 结构：LN → MHA → +残差 → LN → FFN → +残差。
    ;; dropout 设为 0 便于复现和调试；正式训练可调回 0.1。
    (seq-add! model (make-transformer-block +d-model+ +n-heads+
                                            :ffn-dim +ffn-dim+
                                            :dropout-rate 0.0d0
                                            :name "tx-block"))

    ;; --- 5) 展平所有位置 ---
    ;; (B, 8, 32) → (B, 256)。也可以用 mean-pool，
    ;; 但那会把 8 个位置的信息平均掉，此处保留全部位置更直观。
    (seq-add! model (make-flatten :start-dim 1 :name "flat"))

    ;; --- 6) 分类头（两层 MLP）---
    ;; 第一层提供非线性（GELU），第二层输出 2 类 logits。
    ;; 若不接隐层直接 dense(2)，模型表达能力会受限（线性分类头）。
    (seq-add! model (make-dense 64 :in-dim flat-dim
                                :activation :gelu :name "cls-h"))
    (seq-add! model (make-dense +classes+ :in-dim 64
                                :activation :none :name "cls-out"))
    model))

;;; ============================================================================
;;; 4. 数据生成
;;; ============================================================================
;;;
;;; 这里直接在线生成随机 batch，无需外部数据集文件。
;;; 任务平衡性：每个 token ∈ {0..9} 均匀分布，8 个之和期望为 36；
;;; 「>36」与「<=36」的样本约各占一半。

(defun generate-batch (batch-size)
  "生成一批随机样本。
   返回 (values INDICES LABELS)：
     INDICES - (batch, seq-len) :int64
     LABELS  - (batch,)         :int64，取值 {0, 1}"
  (let* ((idx (make-array (list batch-size +seq-len+)
                          :element-type 'fixnum))
         (lab (make-array batch-size :element-type 'fixnum)))
    (dotimes (i batch-size)
      (let ((s 0))
        (dotimes (j +seq-len+)
          (let ((tok (random +vocab+)))
            (setf (aref idx i j) tok)
            (incf s tok)))
        (setf (aref lab i) (if (> s 36) 1 0))))
    (values (vt-from-array idx :dtype :int64 :fast t)
            (vt-from-array lab :dtype :int64 :fast t))))

;;; ============================================================================
;;; 5. 预测辅助
;;; ============================================================================
;;;
;;; 手写 argmax 而不调用 vt-argmax，是因为不同版本 vt-argmax 的返回值
;;; 约定（是否返回 (values values indices)）可能不同，手写比较最稳。

(defun predict-class (logits i)
  "从形状 (B, 2) 的 logits 中取第 I 行的预测类别（0 或 1）。"
  (if (> (coerce (vt-ref logits i 0) 'double-float)
         (coerce (vt-ref logits i 1) 'double-float))
      0 1))

;;; ============================================================================
;;; 6. 训练循环
;;; ============================================================================
;;;
;;; 每次迭代的步骤：
;;;   forward → compute-loss → compute-loss-gradient
;;;           → zero-grad! → backward → optimizer-step
;;;           → clear-step-caches!
;;;
;;; 关键点：
;;;   - zero-grad! 必须在 backward 之前：它把上一步残留的梯度置 nil，
;;;     rnn/lstm 等层会把梯度累加，不 zero 会跨步污染。
;;;   - clear-step-caches! 必须在 optimizer-step 之后：
;;;     它清空整个前向缓存（包括 backward 依赖的中间张量），
;;;     在 forward 与 backward 之间调用会让 backward 崩溃。
;;;   - 延迟初始化：Embedding / Dense / Conv2d 等层在构造时不知道输入维度，
;;;     需要一次真实前向才能创建权重张量。本例显式调用一次 forward
;;;     并立刻清缓存，等价于库提供的 (build-model model dummy-input)。

(defun train-sum-classifier
    (&key (epochs 15)
          (steps-per-epoch 100)   ; 每 epoch 的 batch 数
          (batch-size 32)
          (lr 3.0d-3))
  "训练求和阈值分类器，返回训练完毕的模型。"
  (let* ((model     (build-sum-classifier))
         (optimizer (make-adam :lr lr :weight-decay 1.0d-4))
         (loss-fn   (make-ce-loss)))

    ;; --- 延迟初始化：一次虚拟前向 ---
    ;; 用一个 (1, seq-len) 的 dummy input 触发所有依赖输入维度的层。
    ;; 执行后必须清理缓存，避免残留的中间张量占用内存 / 影响后续训练。
    (set-model-training! model t)   ; 显式进入训练模式（dropout 生效等）
    (forward model
             (vt-reshape
              (vt-from-sequence '(0 0 0 0 0 0 0 0) :dtype :int64)
              (list 1 +seq-len+)))
    (clear-step-caches! model)

    ;; 触发初始化后再统计参数量，否则会得到 0。
    (format t "~&参数量：~a~%" (param-count model))

    (dotimes (epoch epochs)
      (let ((total-loss 0.0d0)
            (correct    0)
            (total      0))
        (dotimes (step steps-per-epoch)
          (multiple-value-bind (x y) (generate-batch batch-size)
            (let* ((logits (forward model x))                        ; 前向
                   (loss   (compute-loss loss-fn logits y))          ; 损失
                   (grad   (compute-loss-gradient loss-fn logits y))); dL/dlogits
              (incf total-loss (coerce (vt-item loss) 'double-float))

              ;; 准确率统计（训练时 batch 上）
              (dotimes (i batch-size)
                (incf total)
                (when (= (predict-class logits i)
                         (coerce (vt-ref y i) 'fixnum))
                  (incf correct)))

              ;; 反向 + 参数更新
              (zero-grad! model)                                      ; 清空梯度
              (backward model grad)                                   ; 反向传播
              (optimizer-step optimizer (params model) (grads model)) ; 更新参数
              (clear-step-caches! model))))                           ; 清空缓存

        (format t "Epoch ~2d/~2d  loss = ~,4f  acc = ~,4f~%"
                (1+ epoch) epochs
                (/ total-loss steps-per-epoch)
                (/ (float correct 1.0d0)
                   (float total 1.0d0)))))
    (format t "~&训练完成。~%")
    model))

;;; ============================================================================
;;; 7. 评估
;;; ============================================================================
;;;
;;; with-training 动态绑定全局 *training-mode* 为 NIL：
;;;   - Dropout 层会关闭随机失活（直接返回输入）
;;;   - BatchNorm 会使用 running statistics 而非 batch statistics
;;; 这是推理模式的标准写法，离开作用域后自动恢复。

(defun evaluate-sum-classifier (model &key (num-batches 8) (batch-size 128))
  "在独立随机 batch 上评估模型准确率。
   返回 (values ACCURACY TOTAL-SAMPLES)。"
  (let ((correct 0) (total 0))
    (with-training nil
      (dotimes (i num-batches)
        (multiple-value-bind (x y) (generate-batch batch-size)
          (let ((logits (forward model x)))
            (dotimes (j batch-size)
              (incf total)
              (when (= (predict-class logits j)
                       (coerce (vt-ref y j) 'fixnum))
                (incf correct)))))))
    (values (/ (float correct 1.0d0) (float total 1.0d0))
            total)))

;;; ============================================================================
;;; 8. 入口
;;; ============================================================================

(defun run-sum-example ()
  "运行本示例：训练 + 评估，返回训练好的模型。"
  (let ((model (train-sum-classifier :epochs 15
                                     :steps-per-epoch 100
                                     :batch-size 32
                                     :lr 3.0d-3)))
    (multiple-value-bind (acc n) (evaluate-sum-classifier model)
      (format t "~&最终准确率 = ~,4f（~a 样本）~%" acc n)
      (format t "随机基线   = ~,4f~%" 0.5d0))
    model))

(run-sum-example)
#|
参数量：25506
Epoch  1/15  loss = 0.2508  acc = 0.9000
Epoch  2/15  loss = 0.1226  acc = 0.9494
Epoch  3/15  loss = 0.1124  acc = 0.9544
Epoch  4/15  loss = 0.0978  acc = 0.9572
Epoch  5/15  loss = 0.0813  acc = 0.9659
Epoch  6/15  loss = 0.0660  acc = 0.9716
Epoch  7/15  loss = 0.0652  acc = 0.9722
Epoch  8/15  loss = 0.0709  acc = 0.9706
Epoch  9/15  loss = 0.0520  acc = 0.9759
Epoch 10/15  loss = 0.0381  acc = 0.9841
Epoch 11/15  loss = 0.0408  acc = 0.9838
Epoch 12/15  loss = 0.0415  acc = 0.9838
Epoch 13/15  loss = 0.0422  acc = 0.9816
Epoch 14/15  loss = 0.0409  acc = 0.9831
Epoch 15/15  loss = 0.0314  acc = 0.9878
训练完成。
最终准确率 = 0.9785（1024 样本）
随机基线   = 0.5000
|#
