(in-package #:nn)

(defun vt-mean-axis (tensor axis &key keepdims)
  "沿指定轴求均值."
  (vt-mean tensor :axis axis :keepdims keepdims))

(defun vt-sum-axis (tensor axis &key keepdims)
  "沿指定轴求和."
  (vt-sum tensor :axis axis :keepdims keepdims))

(defun vt-relu-derivative (z)
  "ReLU 导数: z > 0 → 1, 否则 → 0"
  (vt-map
   (lambda (x) (if (> x 0.0d0) 1.0d0 0.0d0)) z))

(defun vt-leaky-relu-derivative (z &key (alpha 0.01d0))
  "Leaky ReLU 导数."
  (vt-map
   (lambda (x) (if (> x 0.0d0) 1.0d0 alpha)) z))

(defun vt-sigmoid-derivative (a)
  "Sigmoid 导数: a * (1 - a)"
  (vt-* a (vt-- 1.0d0 a)))

(defun vt-tanh-derivative (a)
  "Tanh 导数: 1 - a^2"
  (vt-- 1.0d0 (vt-* a a)))

(defun vt-gelu-derivative (x)
  "GELU 导数（近似）."
  (let* ((c (sqrt (/ 2.0d0 pi)))
         (x3 (vt-expt x 3.0d0))
         (inner (vt-+ x (vt-scale x3 0.044715d0)))
         (tanh-val (vt-tanh (vt-scale inner c)))
         (sec2 (vt-- 1.0d0 (vt-* tanh-val tanh-val)))
	 (dtanh (vt-scale sec2
			  (vt-scale
			   (vt-+ 1.0d0
				 (vt-scale
				  (vt-expt x 2.0d0)
				  (* 3.0d0 0.044715d0)))
			   c)))
         (phi (vt-scale (vt-+ 1.0d0 tanh-val) 0.5d0))
         (dphi (vt-scale dtanh 0.5d0)))
    (vt-+ phi (vt-* x dphi))))

(defun vt-swish-derivative (x)
  "Swish 导数."
  (let ((sig (vt-sigmoid x)))
    (vt-+ sig (vt-* x (vt-* sig (vt-- 1.0d0 sig))))))

(defun vt-hard-sigmoid-derivative (x)
  "Hard Sigmoid 导数."
  (vt-map
   (lambda (v)
     (if (and (>= v -2.5d0) (<= v 2.5d0)) 0.2d0 0.0d0))
   x))

(defun vt-mish-derivative (x)
  "mish 的精确导数."
  (let* ((sp (vt-softplus x))
         (tsp (vt-tanh sp))
         (sech2 (vt-- 1.0d0 (vt-* tsp tsp)))
         (sig (vt-sigmoid x)))
    (vt-+ tsp (vt-* x (vt-* sech2 sig)))))

(defclass dense (layer)
  ((in-dim :initarg :in-dim
	   :initform nil
	   :reader dense-in-dim
           :type (or null fixnum))
   (out-dim :initarg :out-dim
	    :initform nil
	    :reader dense-out-dim
            :type (or null fixnum))
   (weight-init :initarg :weight-init
		:initform nil
                :accessor dense-weight-init)
   (bias-init :initarg :bias-init
	      :initform nil
              :accessor dense-bias-init)
   (use-bias :initarg :use-bias
	     :initform t
             :reader dense-use-bias-p)
   (activation :initarg :activation
	       :initform :none
               :accessor dense-activation
               :type (member :none :relu :leaky-relu :sigmoid
                             :tanh :gelu :swish :mish
                             :softplus :hard-tanh
                             :hard-sigmoid :linear))
   (weights :initarg :weights
	    :initform nil
	    :accessor dense-weights
	    :type (or null vt))
   (bias :initarg :bias
	 :initform nil
	 :accessor dense-bias
	 :type (or null vt))
   (dw :initarg :dw
       :initform nil
       :accessor dense-dw
       :type (or null vt))
   (db :initarg :db
       :initform nil
       :accessor dense-db
       :type (or null vt))
   (input-cache :initarg :input-cache
		:initform nil
		:accessor dense-input-cache
                :type (or null vt))
   (z-cache :initarg :z-cache
	    :initform nil
	    :accessor dense-z-cache
	    :type (or null vt))
   (a-cache :initarg :a-cache
	    :initform nil
	    :accessor dense-a-cache
	    :type (or null vt))
   (leaky-alpha :initarg :leaky-alpha
		:initform 0.01d0
                :accessor dense-leaky-alpha
                :type double-float))
  (:documentation "全连接层: y = activation(x · W + b)"))

(defun make-dense
    (out-dim &key (in-dim nil) (activation :relu) (use-bias t)
               weight-init (bias-init (make-zeros-init))
	       leaky-alpha (name "dense") (trainable t))
  "构造全连接层."
  (make-instance 'dense
		 :out-dim out-dim
		 :in-dim in-dim
		 :activation activation
		 :use-bias use-bias
		 :weight-init weight-init
		 :bias-init bias-init
		 :leaky-alpha (or leaky-alpha 0.01d0)
		 :name name
		 :trainable trainable))


(defmethod forward ((l dense) input)
  (let* ((in-shape (vt-shape input))
         (rank (length in-shape))
         ;; 符合 PyTorch 标准，只认最后一个维度为特征维度
         (in-dim (or (dense-in-dim l)
                     (car (last in-shape))))
         ;; 前面的维度全部相乘作为 Batch
         (batch-size (if (= rank 1)
                         1
                         (reduce #'* (butlast in-shape))))
         ;; 延迟初始化权重
         (w (or (dense-weights l)
                (let* ((fan-in in-dim)
                       (fan-out (dense-out-dim l))
                       (w-init (or (dense-weight-init l)
                                   (make-he-normal))))
                  (setf (slot-value l 'in-dim) in-dim)
                  (setf (dense-weights l)
                        (init-weight w-init
                                     (list in-dim fan-out)
                                     :fan-in fan-in
                                     :fan-out fan-out)))))
         ;; 只有当不是标准 2D 时才展平
         (x-flat (if (and (= rank 2)
                          (= (second in-shape) in-dim))
                     input
                     (vt-reshape input
                                 (list batch-size in-dim))))
         ;; 线性变换 + 偏置
         (z (if (dense-use-bias-p l)
                (let ((b (or (dense-bias l)
                             (setf (dense-bias l)
                                   (init-bias
                                    (or (dense-bias-init l)
                                        (make-zeros-init))
                                    (list (dense-out-dim l)))))))
                  (vt-+ (vt-matmul x-flat w) b))
                (vt-matmul x-flat w)))
         ;; 激活函数
         (a (ecase (dense-activation l)
              ((:none :linear) z)
              ((:relu relu) (vt-relu z))
              ((:leaky-relu leaky-relu)
               (vt-leaky-relu z :alpha (dense-leaky-alpha l)))
              ((:sigmoid sigmoid)
	       (vt-sigmoid z))
              ((:tanh tanh) (vt-tanh z))
              ((:gelu gelu) (vt-gelu z))
              ((:swish swish) (vt-swish z))
              ((:mish mish) (vt-mish z))
              ((:softplus softplus) (vt-softplus z))
              ((:hard-tanh hard-tanh) (vt-hard-tanh z))
              ((:hard-sigmoid hard-sigmoid)
               (vt-hard-sigmoid z)))))
    ;; 巧妙利用 cons 打包，同时缓存原始形状和展平输入
    (setf (dense-input-cache l) (cons in-shape x-flat))
    (setf (dense-z-cache l) z)
    (setf (dense-a-cache l) a)
    ;; 根据原始形状决定是否恢复多维
    (if (or (> rank 2)
            (not (and (= rank 2)
                      (= (second in-shape) in-dim))))
        (vt-reshape a (append (butlast in-shape)
                              (list (dense-out-dim l))))
        a)))

(defmethod backward ((l dense) grad-output)
  (let* ((cached (dense-input-cache l))
         (orig-shape (first cached))  ;; 取出原始形状
         (a-prev (rest cached))       ;; 取出展平后的 2D 矩阵
         (w (dense-weights l))
         (act-kind (dense-activation l))
         (rank (length orig-shape))
         ;; 如果原本是多维输入，把梯度也展平成 2D 去算矩阵乘法
	 (grad-flat
	   (cond
	     ((= rank 1)
	      (vt-reshape grad-output (list 1 (dense-out-dim l))))
	     ((= rank 2) grad-output)
	     (t (vt-reshape grad-output
			    (list (first (vt-shape a-prev))
				  (dense-out-dim l))))))
         ;; 计算激活函数的导数
         (d-activation
           (ecase act-kind
             ((:none :linear) grad-flat)
             ((:relu relu)
              (vt-* grad-flat
                    (vt-relu-derivative (dense-z-cache l))))
             ((:leaky-relu leaky-relu)
              (vt-* grad-flat
                    (vt-leaky-relu-derivative
                     (dense-z-cache l)
                     :alpha (dense-leaky-alpha l))))
             ((:sigmoid sigmoid)
              (vt-* grad-flat
                    (vt-sigmoid-derivative (dense-a-cache l))))
             ((:tanh tanh)
              (vt-* grad-flat
                    (vt-tanh-derivative (dense-a-cache l))))
             ((:gelu gelu)
              (vt-* grad-flat
                    (vt-gelu-derivative (dense-z-cache l))))
             ((:swish swish)
              (vt-* grad-flat
                    (vt-swish-derivative (dense-z-cache l))))
             ((:mish mish)
              (vt-* grad-flat
                    (vt-mish-derivative (dense-z-cache l))))
             ((:softplus softplus)
              (vt-* grad-flat
                    (vt-sigmoid (dense-z-cache l))))
             ((:hard-tanh hard-tanh)
              (let ((z (dense-z-cache l)))
                (vt-* grad-flat
                      (vt-map (lambda (x)
                                (if (and (>= x -1.0d0)
                                         (<= x 1.0d0))
                                    1.0d0 0.0d0))
                              z))))
             ((:hard-sigmoid hard-sigmoid)
              (vt-* grad-flat
                    (vt-hard-sigmoid-derivative
                     (dense-z-cache l))))))
         ;; 计算权重梯度
         (dw (vt-matmul (vt-transpose a-prev)
                        d-activation)))
    (setf (dense-dw l) dw)
    (when (dense-use-bias-p l)
      (setf (dense-db l)
	    (vt-sum d-activation :axis 0 :keepdims nil)))
    ;; 计算输入梯度并还原形状
    (let ((d-x-flat (vt-matmul d-activation
                               (vt-transpose w))))
      (cond
	((= rank 1) (vt-reshape d-x-flat orig-shape))     ; (1, d) -> (d,)
	((= rank 2) d-x-flat)                              ; 形状本就匹配
	(t          (vt-reshape d-x-flat orig-shape))))))


(defmethod params ((l dense))
  (let ((result '()))
    (when (dense-weights l)
      (push (list l "weights" (dense-weights l)
                  #'(lambda (v)
                      (setf (dense-weights l) v)))
            result))
    (when (and (dense-use-bias-p l) (dense-bias l))
      (push (list l "bias" (dense-bias l)
                  #'(lambda (v)
                      (setf (dense-bias l) v)))
            result))
    (nreverse result)))

(defmethod grads ((l dense))
  (let ((result '()))
    ;; 与 PARAMS 严格同构 —— 只要 PARAMS 会返回该项，这里就返回一项
    ;; （梯度尚未算出时 tensor 为 NIL）。否则两个列表长度不等，
    ;; 优化器按位置配对时会用别的层的梯度更新本层参数。
    (when (dense-weights l)
      (push (cons "weights" (dense-dw l)) result))
    (when (and (dense-use-bias-p l) (dense-bias l))
      (push (cons "bias" (dense-db l)) result))
    (nreverse result)))


(defclass activation-layer (layer)
  ((kind :initarg :kind
	 :initform :relu
         :accessor activation-kind
         :type (member :relu :leaky-relu :sigmoid :tanh
                       :gelu :swish :mish :softplus
                       :hard-tanh :hard-sigmoid :linear
			     :softmax :log-softmax))
   (leaky-alpha :initarg :leaky-alpha
		:initform 0.01d0
                :accessor act-leaky-alpha)
   ;; 新增：分别缓存原始输入 (z) 和激活输出 (a)
   (z-cache :initarg :z-cache
            :initform nil
            :accessor act-z-cache)
   (a-cache :initarg :a-cache
            :initform nil
            :accessor act-a-cache)))

(defun make-activation-layer
    (kind &key leaky-alpha
            (name "activation") (trainable nil))
  (make-instance 'activation-layer
		 :kind kind
		 :leaky-alpha (or leaky-alpha 0.01d0)
		 :name name :trainable trainable))

(defmethod forward ((l activation-layer) input)
  (let ((out
          (ecase (activation-kind l)
            ((:relu relu)               (vt-relu input))
            ((:leaky-relu leaky-relu)
             (vt-leaky-relu input :alpha (act-leaky-alpha l)))
            ((:sigmoid sigmoid)         (vt-sigmoid input))
            ((:tanh tanh)               (vt-tanh input))
            ((:gelu gelu)               (vt-gelu input))
            ((:swish swish)             (vt-swish input))
            ((:mish mish)               (vt-mish input))
            ((:softplus softplus)       (vt-softplus input))
            ((:hard-tanh hard-tanh)     (vt-hard-tanh input))
            ((:hard-sigmoid hard-sigmoid) (vt-hard-sigmoid input))
            ((:linear :none)             input)
            (:softmax
             (setf (act-z-cache l) input)
             (vt-softmax input))
            (:log-softmax
             (setf (act-z-cache l) input)
             (vt-log-softmax input)))))
    ;; 根据激活类型，缓存反向传播所需的值
    (case (activation-kind l)
      ((:relu :leaky-relu :gelu :swish :mish :softplus :hard-tanh :hard-sigmoid)
       (setf (act-z-cache l) input)
       (setf (act-a-cache l) nil))
      ((:sigmoid :tanh)
       (setf (act-a-cache l) out)
       (setf (act-z-cache l) nil))
      ((:softmax :log-softmax)
       (setf (act-a-cache l) out))
      (t (setf (act-z-cache l) nil)
         (setf (act-a-cache l) nil)))
    out))

(defmethod backward ((l activation-layer) grad-output)
  (let ((kind (activation-kind l)))
    (ecase kind
      ((:relu relu)
       (vt-* grad-output (vt-relu-derivative (act-z-cache l))))
      ((:leaky-relu leaky-relu)
       (vt-* grad-output (vt-leaky-relu-derivative (act-z-cache l) :alpha (act-leaky-alpha l))))
      ((:sigmoid sigmoid)
       (vt-* grad-output (vt-sigmoid-derivative (act-a-cache l))))
      ((:tanh tanh)
       (vt-* grad-output (vt-tanh-derivative (act-a-cache l))))
      ((:gelu gelu)
       (vt-* grad-output (vt-gelu-derivative (act-z-cache l))))
      ((:swish swish)
       (vt-* grad-output (vt-swish-derivative (act-z-cache l))))
      ((:mish mish)
       (vt-* grad-output (vt-mish-derivative (act-z-cache l))))
      ((:softplus softplus)
       (vt-* grad-output (vt-sigmoid (act-z-cache l))))
      ((:hard-tanh hard-tanh)
       (vt-* grad-output
             (vt-map (lambda (x)
                       (if (and (>= x -1.0d0) (<= x 1.0d0))
                           1.0d0 0.0d0))
                     (act-z-cache l))))
      ((:hard-sigmoid hard-sigmoid)
       (vt-* grad-output (vt-hard-sigmoid-derivative (act-z-cache l))))
      ((:linear :none) grad-output)
      (:softmax
       (let* ((y (act-a-cache l))
              (gy grad-output)
              (sum-gy-y (vt-sum (vt-* gy y) :axis -1 :keepdims t))
              (gz (vt-* y (vt-- gy sum-gy-y))))
         gz))
      (:log-softmax
       (let* ((z (act-z-cache l))
              (s (vt-softmax z))
              (gy grad-output)
              (sum-gy (vt-sum gy :axis -1 :keepdims t))
              (gz (vt-- gy (vt-* s sum-gy))))
         gz)))))

(defclass flatten (layer)
  ((start-dim :initarg :start-dim
	      :initform 1
              :reader flatten-start-dim)
   (cache :initform nil :accessor flatten-cache))
  (:documentation
   "将 (batch, d1, d2, ...) 展平为 (batch, d1*d2*...)."))

(defun make-flatten (&key (start-dim 1) (name "flatten"))
  (make-instance 'flatten
		 :start-dim start-dim :name name :trainable nil))

(defmethod forward ((l flatten) input)
  (let* ((shape (vt-shape input))
         (start (flatten-start-dim l))
         (pre (subseq shape 0 start))
         (post (subseq shape start))
         (post-dim (if post (reduce #'* post) 1)))
    (setf (flatten-cache l) shape)
    (vt-reshape
     input
     (append (if (plusp start) pre '())
             (list post-dim)))))

(defmethod backward ((l flatten) grad-output)
  "把展平后的梯度 reshape 回前向输入形状。"
  (let ((shape (flatten-cache l)))
    (if shape
        (vt-reshape grad-output shape)
        grad-output)))

(defclass residual (layer)
  ((block :initarg :block
	  :initform nil
	  :reader residual-block))
  (:documentation
   "残差连接: output = input + block(input)."))

(defun make-residual (block &key (name "residual"))
  (make-instance 'residual
		 :block block :name name
		 :trainable (layer-trainable-p block)))

(defmethod forward ((l residual) input)
  (let ((out (forward (residual-block l) input)))
    (vt-+ input out)))

(defmethod backward ((l residual) grad-output)
  (let ((grad-block
          (backward (residual-block l) grad-output)))
    (vt-+ grad-output
	  (or grad-block
	      (vt-zeros (vt-shape grad-output))))))

(defmethod params ((l residual))
  (params (residual-block l)))

(defmethod grads ((l residual))
  (grads (residual-block l)))

(defmethod set-training! ((l residual) mode)
  (call-next-method)
  (when (residual-block l)
    (set-training! (residual-block l) mode)))

;; ---- grad-slots (zero-grad! 基础) ----
(defmethod grad-slots ((l dense))
  (let ((slots '(dw)))
    (when (dense-use-bias-p l) (push 'db slots))
    slots))

(defmethod grad-slots ((l activation-layer)) '())

(defmethod grad-slots ((l flatten)) '())

;; ---- cache-slots (clear-forward-cache! 基础) ----
(defmethod cache-slots ((l dense))
  '(input-cache z-cache a-cache))

(defmethod cache-slots ((l activation-layer))
  '(z-cache a-cache))

(defmethod cache-slots ((l flatten))
  '(cache))
