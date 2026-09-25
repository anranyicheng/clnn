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
  "按 PyTorch 语义就地重归一化嵌入矩阵：
   对 L2 范数超过 MAX-NORM 的行缩放到 MAX-NORM，
   并把逐行缩放系数缓存到 EMB-NORM-SCALE，供反向传播还原梯度尺度。"
  (let* ((w (emb-weight l))
         (ne (emb-num-embeddings l))
         (ed (emb-embedding-dim l))
         (max-norm (coerce (emb-max-norm l) 'double-float))
         (scale (vt-ones (list ne))))
    (dotimes (i ne)
      (let ((sq 0.0d0))
        (dotimes (j ed)
          (let ((v (coerce (vt-ref w i j) 'double-float)))
            (incf sq (* v v))))
        (let ((norm (sqrt sq)))
          (when (and (> norm max-norm) (plusp norm))
            (let ((s (/ max-norm norm)))
              (setf (vt-ref scale i) s)
              (dotimes (j ed)
                (setf (vt-ref w i j)
                      (* s (coerce (vt-ref w i j) 'double-float)))))))))
    (setf (emb-norm-scale l) scale)))

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
  ;; max-norm 重归一化（就地缩放超限行，并缓存逐行系数）
  (when (emb-max-norm l)
    (renorm-embedding-weights! l))
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
