;;;; demo-last-bit.lisp
;;;;
;;;; 序列级二分类：Embedding → LSTM → TakeLastTimeStep → Dense
;;;; 任务：给一个长度 10 的 token 序列（词表 10），判断其中是否出现过 token 7。
;;;; 预期：15 epoch 内 test-acc → 100%。
;;;;
;;;; 这个示例的价值在两处：
;;;;   1. TAKE-LAST 层 —— 从 (batch, seq, dim) 里只取最后时间步，是「用 LSTM
;;;;      做序列分类」的最小必要组件。它演示了自定义层要实现的全部方法：
;;;;      forward / backward / params / grads / grad-slots / cache-slots。
;;;;   2. 整条链路都是纯手写求导：backward 里手工把梯度放回最后时间步，其余
;;;;      时间步为零 —— 没有 autograd 会替你补这个洞。
;;;;
;;;; 用法：
;;;;   (in-package :clnn)
;;;;   (load "examples/demo-last-bit.lisp")
;;;;   (demo-last-bit)                 ; 默认 4000 训练 / 1000 测试
;;;;   (demo-last-bit :epochs 3)       ; 快速跑通

(in-package :clnn)

;;; ============================================================
;;; 1. 自定义层：取最后时间步
;;; ============================================================

(defclass take-last (layer)
  ((input-shape :initform nil :accessor tlt-input-shape))
  (:documentation
   "从 (batch, seq, dim) 取最后时间步，返回连续张量 (batch, dim)。"))

(defun make-take-last ()
  (make-instance 'take-last :name "take-last" :trainable nil))

(defmethod forward ((l take-last) x)
  (let* ((s (vt-shape x))
         (seq-len (second s)))
    (setf (tlt-input-shape l) s)
    ;; vt-contiguous 保证输出连续，避免下游 matmul 处理非连续视图
    (vt-contiguous
     (vt-slice x (list :all) (list (1- seq-len)) (list :all)))))

(defmethod backward ((l take-last) grad)
  (let* ((s (tlt-input-shape l))
         (seq-len (second s))
         (dx (vt-zeros s)))
    ;; dx 连续；单元素整数索引压缩维度，grad 与 (vt-slice dx ...) 形状对齐
    (setf (vt-slice dx (list :all) (list (1- seq-len)) (list :all)) grad)
    dx))

(defmethod params ((l take-last)) '())
(defmethod grads  ((l take-last)) '())
(defmethod grad-slots ((l take-last)) '())
(defmethod cache-slots ((l take-last)) '(input-shape))

;;; ============================================================
;;; 2. 数据
;;; ============================================================

(defun gen-token-presence-data (n seq-len vocab target-token)
  "生成 N 个长度 SEQ-LEN 的序列，label = 序列中是否包含 TARGET-TOKEN。"
  (let ((xs '()) (ys '()))
    (dotimes (i n)
      (let* ((has-target (zerop (random 2)))
             (seq (if has-target
                      (let ((s (loop for j from 0 below seq-len
                                     collect (random vocab))))
                        ;; 随机把一个位置改成 target-token
                        (setf (nth (random seq-len) s) target-token)
                        s)
                      ;; 确保不包含 target-token
                      (loop for j from 0 below seq-len
                            collect (let ((v (random vocab)))
                                      (if (= v target-token)
                                          (mod (1+ v) vocab)
                                          v))))))
        (dolist (tok seq) (push tok xs))
        (push (if has-target 1 0) ys)))
    (values
     (vt-reshape (vt-from-sequence (nreverse xs) :dtype :int64)
                 (list n seq-len))
     (vt-from-sequence (nreverse ys) :dtype :int64))))

(defun compute-acc (pred y)
  "两分类准确率。PRED: (batch, 2) logits；Y: (batch,) 整数标签。"
  (let* ((batch (first (vt-shape pred)))
         (p-data (vt-data pred))
         (p-off (vt-offset pred))
         (p-rs (first (vt-strides pred)))
         (p-cs (second (vt-strides pred)))
         (y-data (vt-data y))
         (y-off (vt-offset y))
         (y-rs (first (vt-strides y)))
         (correct 0))
    (dotimes (i batch)
      (let* ((base (+ p-off (* i p-rs)))
             (v0 (aref p-data base))
             (v1 (aref p-data (+ base p-cs)))
             (pred-class (if (> v0 v1) 0 1))
             (true-class (aref y-data (+ y-off (* i y-rs)))))
        (when (= pred-class true-class)
          (incf correct))))
    correct))

;;; ============================================================
;;; 3. 训练入口
;;; ============================================================

(defun demo-last-bit (&key (n-train 4000) (n-test 1000) (seq-len 10)
                           (vocab 10) (target 7) (emb 16) (hidden 32)
                           (epochs 15) (batch-size 64) (lr 1e-3))
  (format t "~%=== 判断 token ~a 是否出现在序列中 ===~%" target)
  (format t "模型: Embedding(~a→~a) → LSTM(~a→~a) → TakeLastTimeStep → Dense(~a→2)~%~%"
          vocab emb emb hidden hidden)
  (multiple-value-bind (x-train y-train)
      (gen-token-presence-data n-train seq-len vocab target)
    (multiple-value-bind (x-test y-test)
        (gen-token-presence-data n-test seq-len vocab target)
      (let* ((model (make-sequential))
             (loss-fn (make-ce-loss :reduction :mean))
             (opt (make-adam :lr lr))
             (n-batches (floor n-train batch-size)))
        (seq-add! model (make-embedding vocab emb))
        (seq-add! model (make-lstm emb hidden))
        (seq-add! model (make-take-last))
        (seq-add! model (make-dense 2 :activation :none))
        (dotimes (epoch epochs)
          (let ((epoch-loss 0.0d0) (epoch-correct 0))
            (dotimes (b n-batches)
              (let* ((start (* b batch-size))
                     (end (+ start batch-size))
                     (x (vt-slice x-train (list start end) (list :all)))
                     (y (vt-slice y-train (list start end))))
                (zero-grad! model)
                (let* ((pred (forward model x))
                       (loss (compute-loss loss-fn pred y))
                       (grad (compute-loss-gradient loss-fn pred y)))
                  (backward model grad)
                  (update! model opt)
                  (incf epoch-loss (vt-item loss))
                  (incf epoch-correct (compute-acc pred y)))))
            (let ((test-correct 0) (test-total 0) (start 0))
              (loop while (< start n-test) do
                (let* ((end (min (+ start batch-size) n-test))
                       (x (vt-slice x-test (list start end) (list :all)))
                       (y (vt-slice y-test (list start end)))
                       (pred (forward model x)))
                  (incf test-correct (compute-acc pred y))
                  (incf test-total (- end start))
                  (setf start end)))
              (format t "Epoch ~2a/~a  loss=~,4f  train=~,3f  test=~,3f~%"
                      (1+ epoch) epochs
                      (/ epoch-loss n-batches)
                      (/ epoch-correct n-train)
                      (/ test-correct test-total)))))))))
