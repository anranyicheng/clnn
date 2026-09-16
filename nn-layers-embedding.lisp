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
		  :accessor emb-indices-cache))
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
                 (vt-zeros (list ne ed)))))
    (dotimes (i flat-size)
      (let* ((idx-val (coerce (vt-ref flat-idx i) 'fixnum))
             (scale (if freq
                        (/ 1.0d0 (coerce (gethash idx-val freq) 'double-float))
                        1.0d0)))
        (dotimes (j ed)
          (setf (vt-ref dw idx-val j)
                (+ (vt-ref dw idx-val j)
                   (* scale (coerce (vt-ref flat-go i j) 'double-float)))))))
    (setf (emb-dw l) dw)
    nil))

(defmethod params ((l embedding))
  (when (emb-weight l)
    (list (list l "weight" (emb-weight l)
                #'(lambda (v)
                    (setf (emb-weight l) v))))))

(defmethod grads ((l embedding))
  (when (emb-dw l)
    (list (cons "weight" (emb-dw l)))))

;; ---- grad-slots ----
(defmethod grad-slots ((l embedding)) '(dw))

(defmethod cache-slots ((l embedding))
  '(indices-cache))
