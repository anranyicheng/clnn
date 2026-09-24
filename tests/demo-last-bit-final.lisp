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
