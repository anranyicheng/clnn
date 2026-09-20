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
  ;; P1-7: 依据「前向是否真的生成了 mask」分支，而不是反向时的 training-p。
  ;; 前后向模式不一致（with-training / set-training! 交叉）时，
  ;; 原实现会漏乘或错乘 mask。
  (let ((mask (dropout-mask-cache l)))
    (if mask
        (vt-* grad-output mask)
        grad-output)))

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
    (num-features &key eps momentum (affine t)
                    (name "batch-norm") (trainable t))
  (make-instance 'batch-norm
		 :num-features num-features
		 :eps (or eps 1.0d-5)
		 :momentum (or momentum 0.1d0)
		 :affine (if affine t nil)
		 :name name :trainable trainable))

(defun bn-permute-to-channel-last (shape)
  "Permutation to move axis=1 (C) to last: (N,C,H,W) -> (N,H,W,C), i.e. perm (0,2,3,...,1)."
  (append (list 0)
	  (loop for i from 2 below (length shape)
		collect i)
	  (list 1)))

(defun bn-invert-perm (perm)
  (let* ((n (length perm))
         (inv (make-list n)))
    (dotimes (i n inv)
      (setf (nth (nth i perm) inv) i))))

(defmethod forward ((l batch-norm) input)
  (let* ((nf (bn-num-features l))
         (eps (bn-eps l))
         (shape (vt-shape input))
         (rank (length shape)))
    (unless (bn-gamma l)
      (setf (bn-gamma l) (vt-ones (list nf)))
      (setf (bn-beta l) (vt-zeros (list nf)))
      (setf (bn-running-mean l) (vt-zeros (list nf)))
      (setf (bn-running-var l) (vt-ones (list nf))))
    (if (= rank 2)
        (let* ((batch (first shape)))
          (setf (bn-batch-size l) batch)
          (setf (bn-input-cache l) input)
          (if (training-p l)
              (let* ((mean-r (vt-mean input :axis 0 :keepdims t))
                     (diff (vt-- input mean-r))
                     (var-r (vt-mean (vt-square diff) :axis 0 :keepdims t))
                     (std-inv (vt-map (lambda (v)
					(/ 1.0d0 (sqrt (+ v eps))))
				      var-r))
                     (xhat (vt-* diff std-inv)))
                (setf (bn-xhat-cache l) xhat)
                (setf (bn-std-inv-cache l) std-inv)
                (let ((m (bn-momentum l))
                      ;; Bessel 校正: 无偏方差 = 有偏方差 * N/(N-1)
                      (bessel (if (> batch 1)
				  (/ (coerce batch 'double-float)
                                     (coerce (1- batch) 'double-float))
				  1.0d0)))
                  (setf (bn-running-mean l)
			(vt-+ (vt-scale (bn-running-mean l) (- 1.0d0 m))
			      (vt-scale (vt-reshape mean-r (list nf)) m)))
                  (setf (bn-running-var l)
			(vt-+ (vt-scale (bn-running-var l) (- 1.0d0 m))
			      (vt-scale (vt-scale (vt-reshape var-r (list nf)) bessel) m))))
                (if (bn-affine-p l) (vt-+ (vt-* (bn-gamma l) xhat) (bn-beta l)) xhat))
              (let* ((rm (vt-reshape (bn-running-mean l) (list 1 nf)))
                     (rv (vt-reshape (bn-running-var l) (list 1 nf)))
                     (std-inv (vt-map (lambda (v) (/ 1.0d0 (sqrt (+ v eps)))) rv))
                     (xhat (vt-* (vt-- input rm) std-inv)))
                (if (bn-affine-p l) (vt-+ (vt-* (bn-gamma l) xhat) (bn-beta l)) xhat))))
        ;; ND case: permute C to last, flatten to 2D
        (let* ((perm (bn-permute-to-channel-last shape))
               (inv-perm (bn-invert-perm perm))
               (trans (vt-transpose input perm))
               (tshape (vt-shape trans))
               (spatial (reduce #'* (butlast tshape)))
               (two-d (vt-reshape trans (list spatial nf))))
          (setf (bn-batch-size l) spatial)
          (setf (bn-input-cache l) (list input inv-perm tshape))
          (if (training-p l)
              (let* ((mean-r (vt-mean two-d :axis 0 :keepdims t))
                     (diff (vt-- two-d mean-r))
                     (var-r (vt-mean (vt-square diff) :axis 0 :keepdims t))
                     (std-inv-r (vt-map (lambda (v)
					  (/ 1.0d0 (sqrt (+ v eps))))
					var-r))
                     (xhat-2d (vt-* diff std-inv-r))
                     (mean-v (vt-reshape mean-r (list nf)))
                     (var-v (vt-reshape var-r (list nf))))
                (setf (bn-xhat-cache l) xhat-2d)
                (setf (bn-std-inv-cache l) std-inv-r)
                (let ((m (bn-momentum l))
                      ;; Bessel 校正: 无偏方差 = 有偏方差 * N/(N-1)
                      (bessel (if (> spatial 1)
				  (/ (coerce spatial 'double-float)
                                     (coerce (1- spatial) 'double-float))
				  1.0d0)))
                  (setf (bn-running-mean l)
			(vt-+ (vt-scale (bn-running-mean l) (- 1.0d0 m))
			      (vt-scale mean-v m)))
                  (setf (bn-running-var l)
			(vt-+ (vt-scale (bn-running-var l) (- 1.0d0 m))
			      (vt-scale (vt-scale var-v bessel) m))))
                (let* ((y-2d (if (bn-affine-p l)
				 (vt-+ (vt-* xhat-2d (bn-gamma l)) (bn-beta l))
				 xhat-2d))
                       (y-t (vt-reshape y-2d tshape))
                       (y (vt-transpose y-t inv-perm))) y))
              (let* ((mean-v (vt-reshape (bn-running-mean l) (list 1 nf)))
                     (var-v (vt-reshape (bn-running-var l) (list 1 nf)))
                     (std-inv (vt-map (lambda (v) (/ 1.0d0 (sqrt (+ v eps)))) var-v))
                     (xhat-2d (vt-* (vt-- two-d mean-v) std-inv))
                     (y-2d (if (bn-affine-p l)
			       (vt-+ (vt-* xhat-2d (bn-gamma l)) (bn-beta l))
			       xhat-2d))
                     (y-t (vt-reshape y-2d tshape))
                     (y (vt-transpose y-t inv-perm))) y))))))

(defmethod backward ((l batch-norm) grad-output)
  (let ((cache (bn-input-cache l)))
    (if (vt-p cache)
        (let* ((n (coerce (bn-batch-size l) 'double-float))
               (xhat (bn-xhat-cache l))
               (std-inv (bn-std-inv-cache l))
               (dxhat (if (bn-affine-p l) (vt-* grad-output (bn-gamma l)) grad-output)))
          (when (bn-affine-p l)
            (setf (bn-dgamma l) (vt-sum (vt-* grad-output xhat) :axis 0))
            (setf (bn-dbeta l) (vt-sum grad-output :axis 0)))
          (let* ((sum-dxhat (vt-sum dxhat :axis 0 :keepdims t))
                 (sum-dxhat-xhat (vt-sum (vt-* dxhat xhat) :axis 0 :keepdims t))
                 (dx (vt-* std-inv (vt-scale (vt-- (vt-- (vt-scale dxhat n) sum-dxhat)
						   (vt-* xhat sum-dxhat-xhat))
					     (/ 1.0d0 n)))))
            dx))
        (destructuring-bind (orig-input inv-perm tshape) cache
          (declare (ignore orig-input))
          (let* ((nf (bn-num-features l))
                 (spatial (reduce #'* (butlast tshape)))
                 (fwd-perm (bn-invert-perm inv-perm))
                 (grad-t (vt-transpose grad-output fwd-perm))
                 (g2d (vt-reshape grad-t (list spatial nf)))
                 (n (coerce (bn-batch-size l) 'double-float))
                 (xhat (bn-xhat-cache l))
                 (std-inv-r (bn-std-inv-cache l))
                 (dxhat (if (bn-affine-p l)
			    (vt-* g2d (vt-reshape (bn-gamma l) (list 1 nf))) g2d)))
            (when (bn-affine-p l)
              (setf (bn-dgamma l) (vt-sum (vt-* g2d xhat) :axis 0))
              (setf (bn-dbeta l) (vt-sum g2d :axis 0)))
            (let* ((sum-dxhat (vt-sum dxhat :axis 0 :keepdims t))
                   (sum-dxhat-xhat (vt-sum (vt-* dxhat xhat) :axis 0 :keepdims t))
                   (dx2d (vt-* std-inv-r (vt-scale (vt-- (vt-- (vt-scale dxhat n) sum-dxhat)
							 (vt-* xhat sum-dxhat-xhat))
						   (/ 1.0d0 n))))
                   (dx-t (vt-reshape dx2d tshape))
                   (dx (vt-transpose dx-t inv-perm))) dx))))))


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
    (normalized-shape &key eps (affine t)
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
         (norm-size (reduce #'* norm-dims))
         (aligned-shape
           (append (make-list start-axis :initial-element 1)
                   norm-dims)))
    (labels ((align (param)
               (if (equal (vt-shape param) aligned-shape)
                   param
                   (vt-reshape param aligned-shape))))
      (setf (ln-norm-size l) norm-size)
      (setf (ln-input-cache l) input)
      (unless (ln-gamma l)
        (setf (ln-gamma l) (vt-ones norm-dims))
        (setf (ln-beta l) (vt-zeros norm-dims)))
      (let* ((flat-shape
               (append (subseq shape 0 start-axis)
                       (list norm-size)))
             (input-flat
               (if (equal shape flat-shape)
                   input
                   (vt-reshape input flat-shape)))
             (mean-flat (vt-mean input-flat :axis -1 :keepdims t))
             (diff-flat (vt-- input-flat mean-flat))
             (var-flat
               (vt-mean (vt-square diff-flat) :axis -1 :keepdims t))
             (std-inv-flat (vt-std-inv-from-var var-flat eps))
             (xhat-flat (vt-* diff-flat std-inv-flat))
             (xhat
               (if (equal (vt-shape xhat-flat) shape)
                   xhat-flat
                   (vt-reshape xhat-flat shape)))
             (std-inv-target
               (append (subseq shape 0 start-axis)
                       (make-list norm-rank :initial-element 1)))
             (std-inv
               (if (equal (vt-shape std-inv-flat) std-inv-target)
                   std-inv-flat
                   (vt-reshape std-inv-flat std-inv-target))))
        (setf (ln-xhat-cache l) xhat)
        (setf (ln-std-inv-cache l) std-inv)
        (if (ln-affine-p l)
            (vt-+ (vt-* (align (ln-gamma l)) xhat)
                  (align (ln-beta l)))
            xhat)))))

(defun %vt-sum-over-axes (tensor axes &key keepdims)
  "沿多个轴依次求和。
   AXES 为整数列表（可为空，此时原样返回 TENSOR）。
   从大到小排序：当 KEEPDIMS=NIL 时，先求和的轴会被削掉，
   若从小到大求和，低轴索引会漂移，因此必须从高轴开始。

   本函数存在的意义：不能用
     (apply #'vt-sum tensor :keepdims nil :axis axes)
   因为 APPLY 会把最后一个列表参数展开成散列实参，
   变成 (vt-sum tensor :keepdims nil :axis 0 1) 这种
   『:axis 0 1』关键字不成对的调用，直接抛
   SB-INT:SIMPLE-PROGRAM-ERROR: odd number of &KEY arguments。"
  (if (null axes)
      tensor
      (let ((result tensor)
            (sorted (sort (copy-list axes) #'>)))
        (dolist (ax sorted result)
          (setf result (vt-sum result :axis ax :keepdims keepdims))))))

(defmethod backward ((l layer-norm) grad-output)
  (let* ((shape (vt-shape (ln-input-cache l)))
         (rank (length shape))
         (norm-dims (ln-normalized-shape l))
         (norm-rank (length norm-dims))
         (start-axis (- rank norm-rank))
         (d (ln-norm-size l))
         (xhat (ln-xhat-cache l))
         (std-inv (ln-std-inv-cache l))
         (aligned-shape
           (append (make-list start-axis :initial-element 1)
                   norm-dims)))
    (labels ((align (param)
               (if (equal (vt-shape param) aligned-shape)
                   param
                   (vt-reshape param aligned-shape))))
      (let* ((dxhat (if (ln-affine-p l)
                        (vt-* grad-output (align (ln-gamma l)))
                        grad-output))
             (batch-axes (loop for i below start-axis collect i))
             (norm-axes  (loop for i from start-axis below rank collect i)))
        (when (ln-affine-p l)
          (setf (ln-dgamma l)
                (%vt-sum-over-axes (vt-* dxhat xhat)
                                   batch-axes :keepdims nil))
          (setf (ln-dbeta l)
                (%vt-sum-over-axes dxhat
                                   batch-axes :keepdims nil)))
        (let* ((sum-dxhat
                 (%vt-sum-over-axes dxhat norm-axes :keepdims t))
               (sum-dxhat-xhat
                 (%vt-sum-over-axes (vt-* dxhat xhat) norm-axes :keepdims t))
               (dx (vt-* std-inv
                         (vt-scale
                          (vt-- (vt-- (vt-scale dxhat d) sum-dxhat)
                                (vt-* xhat sum-dxhat-xhat))
                          (/ 1.0d0 d)))))
          dx)))))

(defmethod params ((l layer-norm))
  (if (ln-affine-p l)
      (list
       (list l "gamma" (ln-gamma l)
             #'(lambda (v) (setf (ln-gamma l) v)))
       (list l "beta" (ln-beta l)
             #'(lambda (v) (setf (ln-beta l) v))))
      '()))

(defmethod grads ((l layer-norm))
  (if (ln-affine-p l)
      (list (cons "gamma" (ln-dgamma l))
            (cons "beta" (ln-dbeta l)))
      '()))

;; ---- grad-slots ----
(defmethod grad-slots ((l dropout)) '())

(defmethod grad-slots ((l batch-norm))
  (if (bn-affine-p l) '(dgamma dbeta) '()))

(defmethod grad-slots ((l layer-norm))
  (if (ln-affine-p l) '(dgamma dbeta) '()))

(defmethod params ((l batch-norm))
  (if (bn-affine-p l)
      (list (list l "gamma" (bn-gamma l)
                  #'(lambda (v) (setf (bn-gamma l) v)))
            (list l "beta" (bn-beta l)
                  #'(lambda (v) (setf (bn-beta l) v))))
      '()))

(defmethod grads ((l batch-norm))
  (if (bn-affine-p l)
      (list (cons "gamma" (bn-dgamma l))
            (cons "beta"  (bn-dbeta  l)))
      '()))

(defmethod cache-slots ((l dropout))
  '(mask-cache))

(defmethod cache-slots ((l batch-norm))
  ;; 注意：不含 batch-size / running-mean / running-var
  '(input-cache xhat-cache std-inv-cache))

(defmethod cache-slots ((l layer-norm))
  ;; 注意：不含 norm-size
  '(input-cache xhat-cache std-inv-cache))
