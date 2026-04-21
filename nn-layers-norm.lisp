(in-package #:nn)

(defun vt-std-inv-from-var (var eps)
  "计算标准差的倒数: 1 / sqrt(var + eps)"
  (vt-map
   (lambda (v) (/ 1.0d0 (sqrt (+ v eps)))) var))


(defclass dropout (layer)
  ((p :initarg :p :initform 0.5d0
      :accessor dropout-p :type double-float)
   (mask-cache :initarg :mask-cache
	       :initform nil
	       :accessor dropout-mask-cache)
   (inverted :initform t
	     :initarg :inverted
	     :accessor dropout-inverted-p :type boolean))
  (:documentation "Dropout 正则化层."))

(defun make-dropout
    (p &key (name "dropout") (inverted t))
  (assert (<= 0.0d0 p 1.0d0) (p)
          "Dropout probability must be in [0, 1]")
  (make-instance 'dropout
		 :p p :inverted inverted
		 :name name :trainable nil))


(defmethod forward ((l dropout) input)
  (if (training-p l)
      (let ((p (dropout-p l)))
        (if (>= p 1.0d0)
            ;; p=1.0 时直接返回全 0，避免 1/(1-1) 除零
            (progn
              (setf (dropout-mask-cache l)
                    (vt-zeros (vt-shape input)))
              (vt-zeros (vt-shape input)))
            (let* ((scale
                     (if (dropout-inverted-p l)
                         (/ 1.0d0 (- 1.0d0 p))
                         1.0d0))
                   (mask
                     (vt-map
                      (lambda (x)
                        (declare (ignore x))
                        (if (< (random 1.0d0) p)
                            0.0d0 scale))
                      input)))
              (setf (dropout-mask-cache l) mask)
              (vt-* input mask))))
      input))

(defmethod backward ((l dropout) grad-output)
  (if (training-p l)
      (vt-* grad-output (dropout-mask-cache l))
      grad-output))

(defclass batch-norm (layer)
  ((num-features :initarg :num-features
                 :reader bn-num-features :type fixnum)
   (eps :initarg :eps :initform 1.0d-5
        :accessor bn-eps :type double-float)
   (momentum :initarg :momentum :initform 0.1d0
             :accessor bn-momentum :type double-float)
   (affine :initarg :affine :initform t
           :reader bn-affine-p :type boolean)
   
   ;; --- 可学习参数 ---
   (gamma :initarg :gamma :initform nil
          :accessor bn-gamma :type (or null vt))
   (beta :initarg :beta :initform nil
         :accessor bn-beta :type (or null vt))
   
   ;; --- 运行时统计量 ---
   (running-mean :initarg :running-mean :initform nil
                 :accessor bn-running-mean :type (or null vt))
   (running-var :initarg :running-var :initform nil
                :accessor bn-running-var :type (or null vt))
   
   ;; --- 梯度 ---
   (dgamma :initarg :dgamma :initform nil
           :accessor bn-dgamma :type (or null vt))
   (dbeta :initarg :dbeta :initform nil
          :accessor bn-dbeta :type (or null vt))
   
   ;; --- 前向缓存 ---
   (input-cache :initarg :input-cache :initform nil
                :accessor bn-input-cache :type (or null vt))
   (xhat-cache :initarg :xhat-cache :initform nil
               :accessor bn-xhat-cache :type (or null vt))
   (std-inv-cache :initarg :std-inv-cache :initform nil
                  :accessor bn-std-inv-cache :type (or null vt))
   
   ;; --- 状态变量 ---
   ;; 注意: 类型改为 (or null fixnum) 以兼容 :initform nil
   (batch-size :initarg :batch-size :initform nil
               :accessor bn-batch-size :type (or null fixnum)))
  (:documentation "批归一化层."))


(defun make-batch-norm
    (num-features &key eps momentum affine
                    (name "batch-norm") (trainable t))
  (make-instance 'batch-norm
		 :num-features num-features
		 :eps (or eps 1.0d-5)
		 :momentum (or momentum 0.1d0)
		 :affine (if affine t nil)
		 :name name :trainable trainable))

(defmethod forward ((l batch-norm) input)
  (let* ((nf (bn-num-features l))
         (eps (bn-eps l))
         (shape (vt-shape input))
         (batch-size (first shape)))
    ;; 延迟初始化
    (unless (bn-gamma l)
      (setf (bn-gamma l) (vt-ones (list nf)))
      (setf (bn-beta l) (vt-zeros (list nf)))
      (setf (bn-running-mean l) (vt-zeros (list nf)))
      (setf (bn-running-var l) (vt-ones (list nf))))
    (setf (bn-batch-size l) batch-size)
    (setf (bn-input-cache l) input)
    (if (training-p l)
        ;; 训练模式
        (let* ((mean
                 (vt-mean input :axis 0 :keepdims nil))
               (diff (vt-- input mean))
               (var-biased
                 (vt-mean (vt-square diff) :axis 0))
               (std-inv
                 (vt-std-inv-from-var var-biased eps))
               (xhat (vt-* diff std-inv))
               (n-1 (1- batch-size))
               (n-val batch-size)
               (var-unbiased
                 (vt-scale var-biased (/ n-val n-1))))
          (setf (bn-xhat-cache l) xhat)
          (setf (bn-std-inv-cache l) std-inv)
          ;; 更新运行统计量
          (let ((m (bn-momentum l)))
            (setf (bn-running-mean l)
                  (vt-+
                   (vt-scale (bn-running-mean l)
                             (- 1.0d0 m))
                   (vt-scale mean m)))
            (setf (bn-running-var l)
                  (vt-+
                   (vt-scale (bn-running-var l)
                             (- 1.0d0 m))
                   (vt-scale var-unbiased m))))
          (if (bn-affine-p l)
              (vt-+ (vt-* (bn-gamma l) xhat)
                    (bn-beta l))
              xhat))
        ;; 推理模式
        (let* ((std-inv
                 (vt-std-inv-from-var
                  (bn-running-var l) eps))
               (xhat
                 (vt-* (vt-- input (bn-running-mean l))
                       std-inv)))
          (if (bn-affine-p l)
              (vt-+ (vt-* (bn-gamma l) xhat)
                    (bn-beta l))
              xhat)))))

(defmethod backward ((l batch-norm) grad-output)
  "BatchNorm 反向传播."
  (let* ((n (bn-batch-size l))
         (xhat (bn-xhat-cache l))
         (std-inv (bn-std-inv-cache l))
         (dxhat
           (if (bn-affine-p l)
               (vt-* grad-output (bn-gamma l))
               grad-output)))
    (when (bn-affine-p l)
      (setf (bn-dgamma l)
            (vt-sum (vt-* grad-output xhat)
                    :axis 0))
      (setf (bn-dbeta l)
            (vt-sum grad-output :axis 0)))
    (let* ((sum-dxhat
             (vt-sum dxhat :axis 0 :keepdims t))
           (sum-dxhat-xhat
             (vt-sum (vt-* dxhat xhat)
                     :axis 0 :keepdims t))
           (dx
             (vt-*
              std-inv
              (vt-scale
               (vt--
                (vt-- (vt-scale dxhat n) sum-dxhat)
                (vt-* xhat sum-dxhat-xhat))
               (/ 1.0d0 n)))))
      dx)))


(defmethod params ((l batch-norm))
  (if (bn-affine-p l)
      (list
       (list "gamma" (bn-gamma l)
             #'(lambda (v) (setf (bn-gamma l) v)))
       (list "beta" (bn-beta l)
             #'(lambda (v) (setf (bn-beta l) v))))
      '()))

(defmethod grads ((l batch-norm))
  (if (bn-affine-p l)
      (list (cons "gamma" (bn-dgamma l))
            (cons "beta" (bn-dbeta l)))
      '()))

(defclass layer-norm (layer)
  ((normalized-shape :initarg :normalized-shape
                     :reader ln-normalized-shape :type list)
   (eps :initarg :eps :initform 1.0d-5
        :accessor ln-eps :type double-float)
   (affine :initarg :affine :initform t
           :reader ln-affine-p)   
   ;; --- 可学习参数 ---
   (gamma :initarg :gamma :initform nil
          :accessor ln-gamma :type (or null vt))
   (beta :initarg :beta :initform nil
         :accessor ln-beta :type (or null vt))   
   ;; --- 梯度 ---
   (dgamma :initarg :dgamma :initform nil
           :accessor ln-dgamma :type (or null vt))
   (dbeta :initarg :dbeta :initform nil
          :accessor ln-dbeta :type (or null vt))   
   ;; --- 前向缓存 ---
   (input-cache :initarg :input-cache :initform nil
                :accessor ln-input-cache :type (or null vt))
   (xhat-cache :initarg :xhat-cache :initform nil
               :accessor ln-xhat-cache :type (or null vt))
   (std-inv-cache :initarg :std-inv-cache :initform nil
                  :accessor ln-std-inv-cache :type (or null vt))   
   ;; --- 状态变量 ---
   ;; 同样改为 (or null fixnum) 以兼容 :initform nil
   (norm-size :initarg :norm-size :initform nil
              :accessor ln-norm-size :type (or null fixnum)))
  (:documentation "层归一化."))


(defun make-layer-norm
    (normalized-shape &key eps affine
			(name "layer-norm") (trainable t))
  (make-instance 'layer-norm
		 :normalized-shape
		 (if (listp normalized-shape)
		     normalized-shape
		     (list normalized-shape))
		 :eps (or eps 1.0d-5)
		 :affine (if affine t nil)
		 :name name :trainable trainable))

(defmethod forward ((l layer-norm) input)
  (let* ((shape (vt-shape input))
         (rank (length shape))
         (norm-dims (ln-normalized-shape l))
         (norm-rank (length norm-dims))
         (eps (ln-eps l))
         (start-axis (- rank norm-rank))
         (norm-size (reduce #'* norm-dims)))
    (setf (ln-norm-size l) norm-size)
    (setf (ln-input-cache l) input)
    (unless (ln-gamma l)
      (setf (ln-gamma l) (vt-ones norm-dims))
      (setf (ln-beta l) (vt-zeros norm-dims)))
    (let* ((flat-shape
             (append (subseq shape 0 start-axis)
                     (list norm-size)))
           (input-flat (vt-reshape input flat-shape))
           (mean-flat
             (vt-mean input-flat
                      :axis -1 :keepdims t))
           (diff-flat (vt-- input-flat mean-flat))
           (var-flat
             (vt-mean (vt-square diff-flat)
                      :axis -1 :keepdims t))
           ;; 使用提取的公共函数
           (std-inv-flat
             (vt-std-inv-from-var var-flat eps))
           (xhat-flat (vt-* diff-flat std-inv-flat))
           (xhat (vt-reshape xhat-flat shape))
           (std-inv
             (vt-reshape
              std-inv-flat
              (append
               (subseq shape 0 start-axis)
               (make-list norm-rank
                          :initial-element 1)))))
      (setf (ln-xhat-cache l) xhat)
      (setf (ln-std-inv-cache l) std-inv)
      (if (ln-affine-p l)
          (vt-+ (vt-* (ln-gamma l) xhat)
                (ln-beta l))
          xhat))))

(defmethod backward ((l layer-norm) grad-output)
  "LayerNorm 反向传播."
  (let* ((shape (vt-shape (ln-input-cache l)))
         (rank (length shape))
         (norm-dims (ln-normalized-shape l))
         (norm-rank (length norm-dims))
         (start-axis (- rank norm-rank))
         (d (ln-norm-size l))
         (xhat (ln-xhat-cache l))
         (std-inv (ln-std-inv-cache l))
         (dxhat
           (if (ln-affine-p l)
               (vt-* grad-output (ln-gamma l))
               grad-output))
         (flat-shape
           (append (subseq shape 0 start-axis)
                   (list d))))
    (when (ln-affine-p l)
      (let* ((dxhat-flat (vt-reshape dxhat flat-shape))
             (xhat-flat (vt-reshape xhat flat-shape))
             (dgamma-flat
               (vt-sum (vt-* dxhat-flat xhat-flat)
                       :axis -1 :keepdims nil))
             (dbeta-flat
               (vt-sum dxhat-flat
                       :axis -1 :keepdims nil)))
        (setf (ln-dgamma l)
              (if (> start-axis 0)
                  (vt-sum dgamma-flat
                          :axis 0 :keepdims nil)
                  dgamma-flat))
        (setf (ln-dbeta l)
              (if (> start-axis 0)
                  (vt-sum dbeta-flat
                          :axis 0 :keepdims nil)
                  dbeta-flat))))

    (let* ((dxhat-flat (vt-reshape dxhat flat-shape))
           (xhat-flat (vt-reshape xhat flat-shape))
           (sum-dxhat
             (vt-sum dxhat-flat
                     :axis -1 :keepdims t))
           (sum-dxhat-xhat
             (vt-sum (vt-* dxhat-flat xhat-flat)
                     :axis -1 :keepdims t))
           (dx-flat
             (vt-*
              std-inv
              (vt-scale
               (vt--
                (vt-- (vt-scale dxhat-flat d)
                      sum-dxhat)
                (vt-* xhat-flat sum-dxhat-xhat))
               (/ 1.0d0 d))))
           (dx (vt-reshape dx-flat shape)))
      dx)))


(defmethod params ((l layer-norm))
  (if (ln-affine-p l)
      (list
       (list "gamma" (ln-gamma l)
             #'(lambda (v) (setf (ln-gamma l) v)))
       (list "beta" (ln-beta l)
             #'(lambda (v) (setf (ln-beta l) v))))
      '()))

(defmethod grads ((l layer-norm))
  (if (ln-affine-p l)
      (list (cons "gamma" (ln-dgamma l))
            (cons "beta" (ln-dbeta l)))
      '()))
