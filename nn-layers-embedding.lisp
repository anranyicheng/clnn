(in-package #:nn)

(defclass embedding (layer)
  ((num-embeddings :initarg :num-embeddings
		   :initform 0
		   :reader emb-num-embeddings :type fixnum
		   :documentation "词汇表大小")
   (embedding-dim :initarg :embedding-dim
		  :initform 0
		  :reader emb-embedding-dim :type fixnum
		  :documentation "嵌入维度")
   (max-norm :initarg :max-norm
	     :initform nil
	     :accessor emb-max-norm
	     :type (or null double-float))
   (scale-grad-by-freq :initarg :scale-grad-by-freq
		       :initform nil
		       :accessor emb-scale-grad-by-freq)

   (weight :initarg :weight
	   :initform nil
	   :accessor emb-weight :type (or null vt))
   ;; 梯度
   (dw :initarg :dw
       :initform nil
       :accessor emb-dw
       :type (or null vt))
   ;; 缓存
   (indices-cache :initarg :indices-cache
		  :accessor emb-indices-cache)
   ;; max-norm 重归一化的逐行缩放系数，形状 (num-embeddings,)；
   ;; 未启用 max-norm 时为 NIL。反向传播需要它来还原梯度尺度。
   (norm-scale :initarg :norm-scale
               :initform nil
               :accessor emb-norm-scale))
  (:documentation "词嵌入层."))

(defun make-embedding
    (num-embeddings embedding-dim
     &key max-norm scale-grad-by-freq
       (name "embedding") (trainable t))
  (make-instance 'embedding
		 :num-embeddings num-embeddings
		 :embedding-dim embedding-dim
		 :max-norm max-norm
		 :scale-grad-by-freq scale-grad-by-freq
		 :name name :trainable trainable))

(defun renorm-embedding-weights! (l)
  "把整张 (V, D) 嵌入表里 L2 范数超过 MAX-NORM 的行缩放到 MAX-NORM。
   这是「对全表做一次」的显式工具（例如训练结束后的整理）。

   注意：FORWARD 不走这条路径 —— 它只重归一化本次前向真正用到的行
   （见 RENORM-EMBEDDING-ROWS!）。整表重归一化是 O(V·D)，
   在 V=50000 时每次前向都是几千万次元素访问，无法接受。

   实现已向量化：行范数由 VT-SUM 沿 axis=1 一次算出，
   不再逐元素 VT-REF 扫描全表。"
  (let* ((w (emb-weight l))
         (ne (emb-num-embeddings l))
         (ed (emb-embedding-dim l))
         (max-norm (coerce (emb-max-norm l) 'double-float))
         (norms (vt-sqrt (vt-sum (vt-square w) :axis 1)))  ; (V,)
         (scale (vt-ones (list ne))))
    (dotimes (i ne)
      (let ((norm (coerce (vt-ref norms i) 'double-float)))
        (when (and (> norm max-norm) (plusp norm))
          (let ((s (/ max-norm norm)))
            (setf (vt-ref scale i) s)
            (dotimes (j ed)
              (setf (vt-ref w i j)
                    (* s (coerce (vt-ref w i j) 'double-float))))))))
    (setf (emb-norm-scale l) scale)))

(defun %embedding-unique-rows (indices ne)
  "返回本次前向引用到的、位于 [0, NE) 内的唯一行号列表（升序）。"
  (let* ((flat (vt-flatten indices))
         (n (reduce #'* (vt-shape flat)))
         (seen (make-hash-table :test #'eql))
         (rows '()))
    (dotimes (i n)
      (let ((idx (coerce (vt-ref flat i) 'fixnum)))
        (unless (or (minusp idx) (>= idx ne) (gethash idx seen))
          (setf (gethash idx seen) t)
          (push idx rows))))
    (nreverse rows)))

(defun renorm-embedding-rows! (l indices)
  "只对 INDICES 中真正用到的行做 max-norm 重归一化（PyTorch 语义）。

   复杂度 O(unique(indices) × D)：行范数用一次向量化 VT-SUM 算出，
   只有确实超限的行才逐元素缩放。与词表大小 V 无关。

   若没有任何行超限（训练收敛后的常态），直接把 EMB-NORM-SCALE 置 NIL，
   连 (V,) 的缩放向量都不分配。"
  (let* ((w (emb-weight l))
         (ne (emb-num-embeddings l))
         (ed (emb-embedding-dim l))
         (max-norm (coerce (emb-max-norm l) 'double-float))
         (rows (%embedding-unique-rows indices ne)))
    (if (null rows)
        (setf (emb-norm-scale l) nil)
        (let* ((idx-array (make-array (length rows) :element-type '(signed-byte 64)))
               (norms nil)
               (over '()))
          (loop for r in rows for i from 0
                do (setf (aref idx-array i) r))
          (setf norms (vt-sqrt
                       (vt-sum (vt-square (vt-take w (vt-from-array idx-array
                                                                    :dtype :int64)
                                                   :axis 0))
                               :axis 1)))
          (loop for r in rows for i from 0
                do (let ((norm (coerce (vt-ref norms i) 'double-float)))
                     (when (and (> norm max-norm) (plusp norm))
                       (push (cons r (/ max-norm norm)) over))))
          (if (null over)
              (setf (emb-norm-scale l) nil)
              (let ((scale (vt-ones (list ne))))
                (dolist (pair over)
                  (destructuring-bind (row . s) pair
                    (setf (vt-ref scale row) s)
                    (dotimes (j ed)
                      (setf (vt-ref w row j)
                            (* s (coerce (vt-ref w row j) 'double-float))))))
                (setf (emb-norm-scale l) scale)))))))

(defmethod forward ((l embedding) indices)
  "前向传播: 查表获取嵌入向量."
  ;; 延迟初始化权重
  (unless (emb-weight l)
    (setf (emb-weight l)
          (vt-scale
           (vt-random-normal
            (list (emb-num-embeddings l)
                  (emb-embedding-dim l)))
           0.01d0)))
  ;; 先缓存 indices
  (setf (emb-indices-cache l) indices)
  ;; max-norm 重归一化：只处理本次用到的行，并缓存逐行系数
  (if (emb-max-norm l)
      (renorm-embedding-rows! l indices)
      ;; 关闭 max-norm 时必须清掉上一轮的缩放系数，
      ;; 否则 BACKWARD 会拿陈旧系数去缩放梯度。
      (setf (emb-norm-scale l) nil))
  (let* ((weights (emb-weight l))
         (edim (emb-embedding-dim l))
         (idx-shape (vt-shape indices))
         ;; 沿第 0 轴 (词表维度) 取值
         (taken (vt-take weights indices :axis 0))
         ;; 恢复 indices 的原始形状，追加 embedding 维度
         (out-shape (append idx-shape (list edim))))
    ;; 确保将计算出的张量作为结果返回
    (vt-reshape taken out-shape)))


(defmethod backward ((l embedding) grad-output)
  "梯度累积到嵌入矩阵的对应行。
   当 scale-grad-by-freq 为 T 时，每个嵌入向量的梯度除以该索引在批次中出现的次数。"
  (let* ((indices (emb-indices-cache l))
         (ne (emb-num-embeddings l))
         (ed (emb-embedding-dim l))
         (idx-shape (vt-shape indices))
         (flat-size (reduce #'* idx-shape))
         (flat-idx (vt-reshape indices (list flat-size)))
         (flat-go  (vt-reshape grad-output (list flat-size ed)))
         (freq (when (emb-scale-grad-by-freq l)
                 (let ((ht (make-hash-table :test #'eql)))
                   (dotimes (i flat-size)
                     (let ((idx (coerce (vt-ref flat-idx i) 'fixnum)))
                       (incf (gethash idx ht 0))))
                   ht)))
         (dw (or (emb-dw l)
                 (vt-zeros (list ne ed))))
         (norm-scale (emb-norm-scale l)))
    (dotimes (i flat-size)
      (let* ((idx-val (coerce (vt-ref flat-idx i) 'fixnum))
             (freq-scale (if freq
                             (/ 1.0d0 (coerce (gethash idx-val freq) 'double-float))
                             1.0d0))
             ;; max-norm 生效时，前向把该行缩放了 s 倍，
             ;; 反向必须乘回 s 才是正确的梯度尺度。
             (renorm-scale (if (and norm-scale
                                    (>= idx-val 0)
                                    (< idx-val ne))
                               (coerce (vt-ref norm-scale idx-val) 'double-float)
                               1.0d0))
             (scale (* freq-scale renorm-scale)))
        (dotimes (j ed)
          (setf (vt-ref dw idx-val j)
                (+ (vt-ref dw idx-val j)
                   (* scale (coerce (vt-ref flat-go i j) 'double-float)))))))
    (setf (emb-dw l) dw)
    (vt-zeros idx-shape)))

(defmethod params ((l embedding))
  (when (emb-weight l)
    (list (list l "weight" (emb-weight l)
                #'(lambda (v)
                    (setf (emb-weight l) v))))))

(defmethod grads ((l embedding))
  ;; 与 PARAMS 同构（PARAMS 以 emb-weight 是否存在为准）。
  (when (emb-weight l)
    (list (cons "weight" (emb-dw l)))))

;; ---- grad-slots ----
(defmethod grad-slots ((l embedding)) '(dw))

(defmethod cache-slots ((l embedding))
  '(indices-cache norm-scale))
