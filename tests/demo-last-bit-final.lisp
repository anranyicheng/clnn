;;;; demo-last-bit-final.lisp
;;;; 任务：输入长度 10 的 0/1 序列，输出最后一位的值。
;;;; 模型：LSTM(1→16) → TakeLastTimeStep → Dense(16→2)
;;;; 预期：test acc → 100%（3–5 epoch 内）
;;;;
;;;; 用法：
;;;;   (in-package :nn)
;;;;   (load "demo-last-bit-final.lisp")
;;;;   (demo-last-bit-final)

(in-package :nn)

;;; ============================================================
;;; 1. TakeLastTimeStep 层
;;;    forward : (batch, seq, dim) → (batch, dim)，强制连续
;;;    backward: 梯度只回传到最后时间步，其余填 0
;;; ============================================================
(defclass take-last (layer)
  ((input-shape :initform nil :accessor tlt-input-shape))
  (:documentation "从 (batch, seq, dim) 取最后时间步，返回连续张量 (batch, dim)。"))

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
;;; 2. 数据生成
;;;    返回 (values X Y)：
;;;      X 形状 (n, seq-len, 1)，float64
;;;      Y 形状 (n,)，int64
;;; ============================================================
(defun gen-last-bit-data (n seq-len)
  (let ((xs '()) (ys '()))
    (dotimes (i n)
      (let ((last 0))
        (dotimes (j seq-len)
          (let ((b (random 2)))
            (push (coerce b 'double-float) xs)
            (when (= j (1- seq-len)) (setf last b))))
        (push last ys)))
    (values
     (vt-reshape (vt-from-sequence (nreverse xs) :dtype :float64)
                 (list n seq-len 1))
     (vt-from-sequence (nreverse ys) :dtype :int64))))

;;; ============================================================
;;; 3. 准确率（内联，无外部依赖）
;;; ============================================================
(defun compute-acc (pred y)
  "PRED 形状 (batch, 2)，Y 形状 (batch,) int64。返回 batch 内正确数。"
  (let* ((batch  (first (vt-shape pred)))
         (p-data (vt-data pred))
         (p-off  (vt-offset pred))
         (p-rs   (first  (vt-strides pred)))
         (p-cs   (second (vt-strides pred)))
         (y-data (vt-data y))
         (y-off  (vt-offset y))
         (y-rs   (first (vt-strides y)))
         (correct 0))
    (dotimes (i batch)
      (let* ((base (+ p-off (* i p-rs)))
             (v0 (aref p-data base))
             (v1 (aref p-data (+ base p-cs)))
             (pred-class (if (> v0 v1) 0 1))
             (true-class (aref y-data (+ y-off (* i y-rs)))))
        (when (= pred-class true-class) (incf correct))))
    correct))

;;; ============================================================
;;; 4. 主函数
;;; ============================================================
(defun demo-last-bit-final (&key (n-train 2000) (n-test 500)
                                   (seq-len 10) (hidden 16)
                                   (epochs 20) (batch-size 64) (lr 1e-3))
  (format t "~%=== Demo: 最后一位复制 ===~%")
  (format t "任务: 输入长度 ~a 的 0/1 序列，输出最后一位的值~%" seq-len)
  (format t "模型: LSTM(1→~a) → TakeLastTimeStep → Dense(~a→2)~%" hidden hidden)
  (format t "n-train=~a  n-test=~a  batch=~a  lr=~a~%~%"
          n-train n-test batch-size lr)

  (multiple-value-bind (x-train y-train) (gen-last-bit-data n-train seq-len)
    (multiple-value-bind (x-test y-test) (gen-last-bit-data n-test seq-len)
      (let* ((model    (make-sequential))
             (loss-fn  (make-ce-loss :reduction :mean))
             (opt      (make-adam :lr lr))
             (n-batches (floor n-train batch-size)))
        (seq-add! model (make-lstm 1 hidden))
        (seq-add! model (make-take-last))
        (seq-add! model (make-dense 2 :activation :none))

        (dotimes (epoch epochs)
          (let ((epoch-loss 0.0d0)
                (epoch-correct 0))
            ;; ---- 训练 ----
            (dotimes (b n-batches)
              (let* ((start (* b batch-size))
                     (end   (+ start batch-size))
                     (x (vt-slice x-train (list start end) (list :all) (list :all)))
                     (y (vt-slice y-train (list start end))))
                (zero-grad! model)
                (let* ((pred (forward model x))
                       (loss (compute-loss loss-fn pred y))
                       (grad (compute-loss-gradient loss-fn pred y)))
                  (backward model grad)
                  (optimizer-step opt (params model) (grads model))
                  (incf epoch-loss (vt-item loss))
                  (incf epoch-correct (compute-acc pred y)))))
            ;; ---- 测试 ----
            (let ((test-correct 0) (test-total 0) (start 0))
              (loop while (< start n-test) do
                (let* ((end (min (+ start batch-size) n-test))
                       (x (vt-slice x-test (list start end) (list :all) (list :all)))
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

(demo-last-bit-final)


;;;; demo-A1-last-bit-gru.lisp
(in-package :nn)

;; take-last 层（同 demo-last-bit-final 的定义）
(defclass take-last (layer)
  ((input-shape :initform nil :accessor tlt-input-shape)))
(defun make-take-last ()
  (make-instance 'take-last :name "take-last" :trainable nil))
(defmethod forward ((l take-last) x)
  (let* ((s (vt-shape x)) (seq-len (second s)))
    (setf (tlt-input-shape l) s)
    (vt-contiguous (vt-slice x (list :all) (list (1- seq-len)) (list :all)))))
(defmethod backward ((l take-last) grad)
  (let* ((s (tlt-input-shape l)) (seq-len (second s))
         (dx (vt-zeros s)))
    (setf (vt-slice dx (list :all) (list (1- seq-len)) (list :all)) grad)
    dx))
(defmethod params ((l take-last)) '())
(defmethod grads  ((l take-last)) '())
(defmethod grad-slots ((l take-last)) '())
(defmethod cache-slots ((l take-last)) '(input-shape))

(defun gen-last-bit-data (n seq-len)
  (let ((xs '()) (ys '()))
    (dotimes (i n)
      (let ((last 0))
        (dotimes (j seq-len)
          (let ((b (random 2)))
            (push (coerce b 'double-float) xs)
            (when (= j (1- seq-len)) (setf last b))))
        (push last ys)))
    (values (vt-reshape (vt-from-sequence (nreverse xs) :dtype :float64)
                        (list n seq-len 1))
            (vt-from-sequence (nreverse ys) :dtype :int64))))

(defun compute-acc (pred y)
  (let* ((batch (first (vt-shape pred)))
         (p-data (vt-data pred)) (p-off (vt-offset pred))
         (p-rs (first (vt-strides pred))) (p-cs (second (vt-strides pred)))
         (y-data (vt-data y)) (y-off (vt-offset y))
         (y-rs (first (vt-strides y))) (correct 0))
    (dotimes (i batch)
      (let* ((base (+ p-off (* i p-rs)))
             (v0 (aref p-data base))
             (v1 (aref p-data (+ base p-cs)))
             (pred-class (if (> v0 v1) 0 1))
             (true-class (aref y-data (+ y-off (* i y-rs)))))
        (when (= pred-class true-class) (incf correct))))
    correct))

(defun demo-A1 (&key (n-train 2000) (n-test 500) (seq-len 10)
                      (hidden 16) (epochs 20) (batch-size 64) (lr 1e-3))
  (format t "~%=== A1: GRU 版最后一位复制 ===~%")
  (format t "模型: GRU(1→~a) → TakeLastTimeStep → Dense(~a→2)~%~%" hidden hidden)
  (multiple-value-bind (x-train y-train) (gen-last-bit-data n-train seq-len)
    (multiple-value-bind (x-test y-test) (gen-last-bit-data n-test seq-len)
      (let* ((model (make-sequential))
             (loss-fn (make-ce-loss :reduction :mean))
             (opt (make-adam :lr lr))
             (n-batches (floor n-train batch-size)))
        (seq-add! model (make-gru 1 hidden))
        (seq-add! model (make-take-last))
        (seq-add! model (make-dense 2 :activation :none))
        (dotimes (epoch epochs)
          (let ((epoch-loss 0.0d0) (epoch-correct 0))
            (dotimes (b n-batches)
              (let* ((start (* b batch-size)) (end (+ start batch-size))
                     (x (vt-slice x-train (list start end) (list :all) (list :all)))
                     (y (vt-slice y-train (list start end))))
                (zero-grad! model)
                (let* ((pred (forward model x))
                       (loss (compute-loss loss-fn pred y))
                       (grad (compute-loss-gradient loss-fn pred y)))
                  (backward model grad)
                  (optimizer-step opt (params model) (grads model))
                  (incf epoch-loss (vt-item loss))
                  (incf epoch-correct (compute-acc pred y)))))
            (let ((test-correct 0) (test-total 0) (start 0))
              (loop while (< start n-test) do
                (let* ((end (min (+ start batch-size) n-test))
                       (x (vt-slice x-test (list start end) (list :all) (list :all)))
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

(demo-A1)

;;;; demo-A2-last-bit-rnn-seq.lisp
(in-package :nn)

;; 复用上面的 take-last / gen-last-bit-data / compute-acc
;; （若已在同一 REPL 会话，直接调用即可；否则把上面的定义粘贴过来）

(defun demo-A2 (&key (n-train 2000) (n-test 500) (seq-len 10)
                      (hidden 16) (epochs 20) (batch-size 64) (lr 1e-3))
  (format t "~%=== A2: RNN-Sequence 版最后一位复制 ===~%")
  (format t "模型: RNN-Sequence(1→~a) → TakeLastTimeStep → Dense(~a→2)~%~%" hidden hidden)
  (multiple-value-bind (x-train y-train) (gen-last-bit-data n-train seq-len)
    (multiple-value-bind (x-test y-test) (gen-last-bit-data n-test seq-len)
      (let* ((model (make-sequential))
             (loss-fn (make-ce-loss :reduction :mean))
             (opt (make-adam :lr lr))
             (n-batches (floor n-train batch-size)))
        ;; 唯一改动：把 LSTM 换成 rnn-sequence
        (seq-add! model (make-rnn-sequence 1 hidden :activation :tanh))
        (seq-add! model (make-take-last))
        (seq-add! model (make-dense 2 :activation :none))
        (dotimes (epoch epochs)
          (let ((epoch-loss 0.0d0) (epoch-correct 0))
            (dotimes (b n-batches)
              (let* ((start (* b batch-size)) (end (+ start batch-size))
                     (x (vt-slice x-train (list start end) (list :all) (list :all)))
                     (y (vt-slice y-train (list start end))))
                (zero-grad! model)
                (let* ((pred (forward model x))
                       (loss (compute-loss loss-fn pred y))
                       (grad (compute-loss-gradient loss-fn pred y)))
                  (backward model grad)
                  (optimizer-step opt (params model) (grads model))
                  (incf epoch-loss (vt-item loss))
                  (incf epoch-correct (compute-acc pred y)))))
            (let ((test-correct 0) (test-total 0) (start 0))
              (loop while (< start n-test) do
                (let* ((end (min (+ start batch-size) n-test))
                       (x (vt-slice x-test (list start end) (list :all) (list :all)))
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

(demo-A2)

;;;; demo-A3-token-presence.lisp
;;;; 任务：长度 10 的 token 序列（词表 10），判断序列中是否包含 token 7。
;;;; 模型：Embedding(10→16) → LSTM(16→32) → TakeLastTimeStep → Dense(32→2)
;;;; 为什么选这个任务：
;;;;   - 需要读入整个序列（不能只看首尾 token）→ 验证 Embedding + LSTM 的整合
;;;;   - 是"全序列聚合"任务，LSTM 的 last hidden 足以编码结果
;;;;   - 比回文简单，但同样能验证 Embedding 的反向传播（梯度回传到索引行）
(in-package :nn)

;; take-last 层（同前）
(defclass take-last (layer)
  ((input-shape :initform nil :accessor tlt-input-shape)))
(defun make-take-last ()
  (make-instance 'take-last :name "take-last" :trainable nil))
(defmethod forward ((l take-last) x)
  (let* ((s (vt-shape x)) (seq-len (second s)))
    (setf (tlt-input-shape l) s)
    (vt-contiguous (vt-slice x (list :all) (list (1- seq-len)) (list :all)))))
(defmethod backward ((l take-last) grad)
  (let* ((s (tlt-input-shape l)) (seq-len (second s))
         (dx (vt-zeros s)))
    (setf (vt-slice dx (list :all) (list (1- seq-len)) (list :all)) grad)
    dx))
(defmethod params ((l take-last)) '())
(defmethod grads  ((l take-last)) '())
(defmethod grad-slots ((l take-last)) '())
(defmethod cache-slots ((l take-last)) '(input-shape))

(defun gen-token-presence-data (n seq-len vocab target-token)
  "生成 N 个长度 SEQ-LEN 的序列，label = 序列中是否包含 TARGET-TOKEN。"
  (let ((xs '()) (ys '()))
    (dotimes (i n)
      (let* ((has-target (zerop (random 2)))
             (seq (if has-target
                      (let ((s (loop for j from 0 below seq-len collect (random vocab))))
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
  (let* ((batch (first (vt-shape pred)))
         (p-data (vt-data pred)) (p-off (vt-offset pred))
         (p-rs (first (vt-strides pred))) (p-cs (second (vt-strides pred)))
         (y-data (vt-data y)) (y-off (vt-offset y))
         (y-rs (first (vt-strides y))) (correct 0))
    (dotimes (i batch)
      (let* ((base (+ p-off (* i p-rs)))
             (v0 (aref p-data base))
             (v1 (aref p-data (+ base p-cs)))
             (pred-class (if (> v0 v1) 0 1))
             (true-class (aref y-data (+ y-off (* i y-rs)))))
        (when (= pred-class true-class) (incf correct))))
    correct))

(defun demo-A3 (&key (n-train 4000) (n-test 1000) (seq-len 10)
                      (vocab 10) (target 7) (emb 16) (hidden 32)
                      (epochs 15) (batch-size 64) (lr 1e-3))
  (format t "~%=== A3: Embedding + LSTM 判断 token ~a 是否出现 ===~%" target)
  (format t "模型: Embedding(~a→~a) → LSTM(~a→~a) → TakeLastTimeStep → Dense(~a→2)~%~%"
          vocab emb emb hidden hidden)
  (multiple-value-bind (x-train y-train) (gen-token-presence-data n-train seq-len vocab target)
    (multiple-value-bind (x-test y-test) (gen-token-presence-data n-test seq-len vocab target)
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
              (let* ((start (* b batch-size)) (end (+ start batch-size))
                     (x (vt-slice x-train (list start end) (list :all)))
                     (y (vt-slice y-train (list start end))))
                (zero-grad! model)
                (let* ((pred (forward model x))
                       (loss (compute-loss loss-fn pred y))
                       (grad (compute-loss-gradient loss-fn pred y)))
                  (backward model grad)
                  (optimizer-step opt (params model) (grads model))
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

(demo-A3)
