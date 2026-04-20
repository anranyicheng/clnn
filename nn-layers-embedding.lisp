(in-package #:nn)

(defclass embedding (layer)
  ((num-embeddings :initarg :num-embeddings
    :reader emb-num-embeddings :type fixnum
    :documentation "词汇表大小")
   (embedding-dim :initarg :embedding-dim
    :reader emb-embedding-dim :type fixnum
    :documentation "嵌入维度")
   (max-norm :initarg :max-norm :initform nil
    :accessor emb-max-norm
    :type (or null double-float))
   (scale-grad-by-freq :initarg :scale-grad-by-freq
    :initform nil
    :accessor emb-scale-grad-by-freq)

   (weight :accessor emb-weight :type (or null vt))
   ;; 梯度
   (dw :accessor emb-dw :type (or null vt))
   ;; 缓存
   (indices-cache :accessor emb-indices-cache))
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
  "梯度累积到嵌入矩阵的对应行."
  (let* ((indices (emb-indices-cache l))
         (ne (emb-num-embeddings l))
         (ed (emb-embedding-dim l))
         (idx-shape (vt-shape indices))
         (flat-size (reduce #'* idx-shape))
         ;; 显式展平 indices，保证绝对是一维索引
         (flat-idx
           (vt-reshape indices (list flat-size)))
         ;; 显式展平 grad-output 为 2D，防止高维越界
         (flat-go
           (vt-reshape grad-output (list flat-size ed)))
         ;; 初始化梯度为全零
         (dw-data
           (make-array (list ne ed)
                       :element-type 'double-float
                       :initial-element 0.0d0)))
    (dotimes (i flat-size)
      (let ((idx-val
              (coerce
                (vt-ref flat-idx (list i))
                'fixnum)))
        (dotimes (j ed)
          (incf (aref dw-data idx-val j)
                (coerce
                  (vt-ref flat-go (list i j))
                  'double-float)))))
    (setf (emb-dw l)
          (vt-reshape
            (vt-from-2d-array dw-data)
            (list ne ed)))))

(defmethod params ((l embedding))
  (when (emb-weight l)
    (list (list "weight" (emb-weight l)
                #'(lambda (v)
                    (setf (emb-weight l) v))))))

(defmethod grads ((l embedding))
  (when (emb-dw l)
    (list (cons "weight" (emb-dw l)))))
