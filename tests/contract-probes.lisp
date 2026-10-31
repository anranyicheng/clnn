;;;; contract-probes.lisp
;;;;
;;;; 契约复核探针（对应《框架设计契约与接口契约文档》§17.1 的 T1–T15）。
;;;;
;;;; 与 dtype-contract-test.lisp 的分工：
;;;;   dtype-contract-test —— 钉 **dtype** 行为（bool / complex / f32 / 索引）
;;;;   contract-probes      —— 钉 **运行时行为**（梯度覆盖写 vs 累加、缓存生命
;;;;                           周期、参数替换、优化器不清零、槽位声明导出……）
;;;;
;;;; 这些条目原本散落在人工探针里，只写进文档而不可复跑；本文件的目的是让
;;;; 「文档声称的行为」与「代码的实际行为」能被同一条命令对照，从而在
;;;; 重构把覆盖写改成累加、或把某层的 cache-slots 漏改时立刻失败。
;;;;
;;;; 用法（需先加载系统）:
;;;;   sbcl --dynamic-space-size 4096 --non-interactive \
;;;;        --eval '(require :asdf)' --eval '(asdf:load-system :clnn)' \
;;;;        --load tests/contract-probes.lisp
;;;;
;;;; 加载即执行；有任何 BAD 项时以非 0 退出。

(in-package :clnn)

(defvar *ok* 0)
(defvar *bad* 0)
(defvar *bad-labels* '())

(defun note-bad (label detail)
  (incf *bad*)
  (push label *bad-labels*)
  (format t "  [FAIL] ~a~@[ => ~a~]~%" label detail))

(defmacro probe (label form)
  "断言 FORM 为真。LABEL 同时用作报告行。"
  `(handler-case
       (let ((v ,form))
         (if v
             (progn (incf *ok*) (format t "  [PASS] ~a~%" ,label))
             (note-bad ,label "返回 NIL")))
     (error (e) (note-bad ,label e))))

(defun show (label value)
  (format t "  [INFO] ~a: ~a~%" label value))

(defun probe-equal (label actual expected)
  "结构相等断言。列表逐项按字符串比较，便于报告「哪个槽名不对」。"
  (let ((a (mapcar #'string actual))
        (b (mapcar #'string expected)))
    (if (equal a b)
        (progn (incf *ok*) (format t "  [PASS] ~a~%" label))
        (note-bad label (format nil "实际 ~s，期望 ~s" a b)))))

(set-global-training! nil)

;;; ================================================================
;;; T1  dense 的梯度是「覆盖写」而非「累加」
;;; ================================================================
;;; 契约（§5.2）：单层 backward 只反映本步梯度，调用方负责在 optimizer-step
;;; 之后 zero-grad!。若实现变成累加，两次 backward 后的 dw 会是两步之和，
;;; 训练循环忘记 zero-grad! 将不再是「能跑但没正则」而是「梯度翻倍」。
(format t "~&~%── T1 dense 覆盖写 ──~%")
(let* ((d (make-dense 3 :in-dim 4 :activation :none))
       (x1 (vt-random-normal '(2 4)))
       (x2 (vt-random-normal '(2 4)))
       (g1 (vt-random-normal '(2 3)))
       (g2 (vt-random-normal '(2 3))))
  (forward d x1)
  (backward d g1)
  (let ((dw1 (dense-dw d)))
    (forward d x2)
    (backward d g2)
    (let* ((dw2 (dense-dw d))
           ;; 干净副本上单独跑第二轮，作为「正确值」的参照
           (d-solo (make-instance 'dense :out-dim 3 :in-dim 4 :activation :none
                                  :weights (vt-copy (dense-weights d))
                                  :bias (vt-copy (dense-bias d))))
           (dw-solo (progn (forward d-solo x2) (backward d-solo g2)
                           (dense-dw d-solo)))
           (diff-solo (vt-item (vt-amax (vt-abs (vt-- dw2 dw-solo)))))
           (diff-sum (vt-item (vt-amax (vt-abs (vt-- dw2 (vt-+ dw1 dw-solo)))))))
      (show "max|dw2 - solo|" diff-solo)
      (show "max|dw2 - (dw1 + solo)|" diff-sum)
      (probe "T1  两次 backward 后 dw 精确等于本轮单独计算的值" (zerop diff-solo))
      (probe "T1b 且不等于两步之和（确认不是累加）" (plusp diff-sum)))))

;;; ================================================================
;;; T2  zero-grad! 与 params/grads 同构
;;; ================================================================
;;; 契约（§5.3）：zero-grad! 把梯度槽置 NIL（不造零张量），params 与 grads
;;; 严格同构，且 grads 保留 NIL 占位（不省略项）—— 省略会让 optimizer 里
;;; 的 (mapcar ...) 按位置对错参数。
(format t "~&~%── T2 zero-grad! 与同构 ──~%")
(let ((d (make-dense 2 :in-dim 3 :activation :none)))
  (forward d (vt-random-normal '(1 3)))
  (backward d (vt-random-normal '(1 2)))
  (probe "T2a backward 后 dw/db 均非 NIL" (and (dense-dw d) (dense-db d)))
  (zero-grad! d)
  (probe "T2b zero-grad! 置 NIL（而非零张量）" (and (null (dense-dw d))
                                                     (null (dense-db d))))
  (probe "T2c params/grads 长度相同" (= (length (params d)) (length (grads d))))
  (probe "T2d grads 保留 NIL 占位（共 2 项）" (and (= 2 (length (grads d)))
                                                   (null (cdr (first (grads d)))))))

;;; ================================================================
;;; T3  全 NIL 梯度时 optimizer-step 跳过
;;; ================================================================
;;; 契约（§11.1）：梯度为 NIL 的参数跳过更新，不报错、不替换参数对象。
;;; 这条保证「模型里有冻结层 / 未参与前向的分支」时可以直接 step。
(format t "~&~%── T3 全 NIL 梯度 ──~%")
(let* ((d (make-dense 2 :in-dim 3 :activation :none))
       (opt (make-sgd :lr 0.1d0))
       (w-before (dense-weights d))
       (b-before (dense-bias d)))
  (zero-grad! d)
  (probe "T3 optimizer-step 全 NIL 梯度不报错" (progn (optimizer-step opt (params d) (grads d)) t))
  (probe "T3b 权重对象未被替换" (eq w-before (dense-weights d)))
  (probe "T3c 偏置对象未被替换" (eq b-before (dense-bias d))))

;;; ================================================================
;;; T4  rnn-cell 的梯度**不**跨调用累积（覆盖写，§5.1）
;;; ================================================================
;;; 契约（§5.1 推广到全部层）：一次 forward + 一次 backward 之后，grads 给出的
;;; 是「这一次反向算出的梯度」。重复 backward 不得叠加 —— 否则忘记 zero-grad!
;;; 时 dense/conv 不受影响、rnn/embedding 却翻倍，同一模型两种语义（§7.3）。
;;;
;;; BPTT 的「跨时间步累加」是 RNN-SEQUENCE 内部的事：它复用同一个 cell 走完整条
;;; 序列，靠动态变量 *RNN-ACCUMULATE-GRADS* 在层内部累加，对外仍是覆盖写。
;;;
;;; 注意本探针的两次 forward 必须用**同一输入**且先 reset 隐状态：rnn-cell 的
;;; STATE 会跨 forward 保留（链式展开的设计），且用随机输入比大小本身没有判据。
(format t "~&~%── T4 rnn-cell 覆盖写 ──~%")
(let ((c (make-rnn-cell 3 4))
      (x (vt-random-normal '(1 3)))
      (g (vt-random-normal '(1 4))))
  (forward c x)
  (backward c g)
  (let ((g1 (vt-copy (rnn-dwih c))))
    (reset-rnn-cell-state! c)
    (forward c x)
    (backward c g)
    (let ((diff (vt-item (vt-amax (vt-abs (vt-- g1 (rnn-dwih c)))))))
      (show "同输入两次 backward 的 max|Δdwih|" diff)
      (probe "T4a rnn-cell 权重梯度不跨调用累积（覆盖写）" (< diff 1d-9))))
  (probe "T4b reset-rnn-cell-state! 后 state 为 NIL"
         (progn (reset-rnn-cell-state! c) (null (rnn-state c)))))

;;; ================================================================
;;; T5  embedding 的 dW 是 scatter-add
;;; ================================================================
;;; 契约（§10.3）：同一行被命中 n 次则梯度为 n（累加而非覆盖），
;;; 且 dX 恒为零张量（整数索引不可导）。
(format t "~&~%── T5 embedding scatter-add ──~%")
(let ((e (make-embedding 5 3)))
  (forward e (vt-from-sequence '(1 1 1) :dtype :int64))
  (let ((dx (backward e (vt-ones '(3 3))))
        (dw (emb-dw e)))
    (show "dW 第 1 行" (vt-to-list (vt-slice dw (list 1) (list :all))))
    (probe "T5a 三次命中 → dW[1] = 3.0" (= 3.0d0 (vt-ref dw 1 0)))
    (probe "T5b 未命中行 → dW[0] = 0.0" (zerop (vt-ref dw 0 0)))
    (probe "T5c dX 为全零（整数索引不可导）"
           (zerop (vt-item (vt-amax (vt-abs dx)))))))

;;; ================================================================
;;; T6  缓存生命周期
;;; ================================================================
;;; 契约（§15）：clear-forward-cache! 只清前向中间量（backward 所需的
;;; 输入副本保留），clear-step-caches! 清全部步内缓存。
(format t "~&~%── T6 缓存生命周期 ──~%")
(let ((d (make-dense 2 :in-dim 3 :activation :relu)))
  (forward d (vt-random-normal '(1 3)))
  (clear-forward-cache! d)
  (probe "T6a clear-forward-cache! 后仍可 backward"
         (vt-p (backward d (vt-random-normal '(1 2)))))
  (clear-step-caches! d)
  (probe "T6b clear-step-caches! 后 input/z/a 缓存全 NIL"
         (and (null (dense-input-cache d))
              (null (dense-z-cache d))
              (null (dense-a-cache d)))))

;;; ================================================================
;;; T7  优化器替换参数对象但不清零梯度槽
;;; ================================================================
;;; 契约（§11.2）：参数被 setter 换成新张量（原地改会让持有旧引用的
;;; 视图失效），梯度槽保持原对象且内容不变（清零是 zero-grad! 的职责）。
(format t "~&~%── T7 参数替换 / 梯度槽保持 ──~%")
(let* ((d (make-dense 2 :in-dim 3 :activation :none))
       (opt (make-sgd :lr 0.1d0))
       (w-before (dense-weights d)))
  (forward d (vt-random-normal '(1 3)))
  (backward d (vt-ones '(1 2)))
  (let ((dw-before (dense-dw d)))
    (optimizer-step opt (params d) (grads d))
    (probe "T7a 参数被替换为新对象" (not (eq w-before (dense-weights d))))
    (probe "T7b 梯度槽对象不变（优化器不清零）"
           (eq dw-before (dense-dw d)))))

;;; ================================================================
;;; T8  标准六步训练循环可收敛
;;; ================================================================
;;; 契约（§3.1）：zero-grad! → forward → loss → loss-gradient → backward →
;;; optimizer-step → clear-step-caches!。这条是「全部契约合起来能跑」的
;;; 冒烟证据：任何一环的语义变了，loss 都不会降到这个水平。
(format t "~&~%── T8 六步循环收敛 ──~%")
(let* ((m (make-sequential))
       (d1 (make-dense 8 :in-dim 4 :activation :relu))
       (d2 (make-dense 3 :in-dim 8 :activation :none))
       (loss (make-ce-loss))
       (opt (make-adam :lr 0.05d0)))
  (seq-add! m (list d1 d2))
  (set-global-training! t)
  (let ((x (vt-random-normal '(32 4)))
        (y (vt-from-sequence (loop repeat 32 collect (random 3)) :dtype :int64))
        (first nil)
        (last nil))
    (dotimes (step 60)
      (zero-grad! m)
      (let* ((pred (forward m x))
             (lv (vt-item (compute-loss loss pred y)))
             (g (compute-loss-gradient loss pred y)))
        (backward m g)
        (optimizer-step opt (params m) (grads m))
        (clear-step-caches! m)
        (when (= step 0) (setf first lv))
        (when (= step 59) (setf last lv))))
    (show "CE loss 第 1 / 第 60 步" (list first last))
    (probe "T8a loss 单调下降（第 60 步 < 第 1 步）" (< last first))
    (probe "T8b 降幅超过 50%（真的在学，不是抖动）" (< last (* 0.5d0 first))))
  (set-global-training! nil))

;;; ================================================================
;;; T15 槽位声明导出（与文档 §4.1 表逐行对照）
;;; ================================================================
;;; 这张表的价值：cache-slots / grad-slots 决定了 clear-step-caches! 和
;;; zero-grad! 触及哪些槽。新增层若漏声明，缓存会跨步泄漏（显存/内存增长）
;;; 而测试全绿 —— 因为它只是「没清干净」。因此把声明钉成表。
(format t "~&~%── T15 cache-slots / grad-slots 声明 ──~%")

(defun probe-slots (label layer cache-expected grad-expected)
  (probe-equal (format nil "~a cache-slots" label) (cache-slots layer) cache-expected)
  (probe-equal (format nil "~a grad-slots" label) (grad-slots layer) grad-expected))

(probe-slots "dense" (make-dense 3 :in-dim 4)
             '(input-cache z-cache a-cache) '(db dw))
(probe-slots "activation-layer" (make-activation-layer :relu)
             '(z-cache a-cache) '())
(probe-slots "flatten" (make-flatten)
             '(cache) '())
(probe-slots "dropout" (make-dropout 0.3d0)
             '(mask-cache) '())
(probe-slots "batch-norm" (make-batch-norm 4)
             '(input-cache xhat-cache std-inv-cache) '(dgamma dbeta))
(probe-slots "layer-norm" (make-layer-norm 4)
             '(input-cache xhat-cache std-inv-cache) '(dgamma dbeta))
(probe-slots "conv2d" (make-conv2d 4 3 :in-channels 2)
             '(input-cache col-cache output-shape-cache) '(db dw))
(probe-slots "max-pool2d" (make-max-pool2d 2)
             '(cache) '())
(probe-slots "avg-pool2d" (make-avg-pool2d 2)
             '(input-cache count-cache) '())
(probe-slots "global-avg-pool2d" (make-global-avg-pool2d)
             '(cache-shape) '())
(probe-slots "embedding" (make-embedding 5 3)
             '(indices-cache norm-scale) '(dw))
(probe-slots "rnn-cell" (make-rnn-cell 3 4)
             '(input-cache h-prev-cache h-cache) '(dwih dwhh dbih))
(probe-slots "lstm" (make-lstm 3 4)
             '(cache) '(dweight-ih dweight-hh dbias-ih dbias-hh))
(probe-slots "gru" (make-gru 3 4)
             '(cache) '(dweight-ih dweight-hh dbias-ih dbias-hh))
(probe-slots "sdpa" (make-scaled-dot-product-attention)
             '(cache) '())
(probe-slots "mha" (make-multi-head-attention 4 2)
             '(cache) '(dw-q dw-k dw-v dw-o db-q db-k db-v db-o))
(probe-slots "transformer-block" (make-transformer-block 4 2 :ffn-dim 8)
             '() '())
(probe-slots "residual" (make-residual (make-dense 3 :in-dim 3))
             '() '())
(probe-slots "stop-gradient" (make-stop-gradient)
             '() '())

;;; ================================================================
;;; 汇总
;;; ================================================================
(format t "~&~%================================================================================~%")
(format t "契约探针汇总: ~a OK / ~a BAD~%" *ok* *bad*)
(when (plusp *bad*)
  (format t "失败项:~%")
  (dolist (l (reverse *bad-labels*))
    (format t "  - ~a~%" l))
  (error "契约探针失败 ~a 项" *bad*))
