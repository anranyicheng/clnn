(in-package #:nn)

(defun vt-square (x)
  "逐元素平方: x^2"
  (vt-* x x))

(defun vt-ones-like (x)
  "创建与 x 同形状的全 1 张量."
  (vt-ones (vt-shape x)))

(defun vt-zeros-like (x)
  "创建与 x 同形状的全 0 张量."
  (vt-zeros (vt-shape x)))

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

(defun vt-leaky-relu-derivative (z &optional (alpha 0.01d0))
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
  (let* ((c (sqrt (/ 2.0d0 (coerce pi 'double-float))))
         (x3 (vt-expt x 3.0d0))
         (inner (vt-+ x (vt-scale x3 0.044715d0)))
         (tanh-val (vt-tanh (vt-scale inner c)))
         (sec2 (vt-- 1.0d0 (vt-* tanh-val tanh-val)))
         (dtanh
           (vt-scale
             sec2
             (* c (vt-+ 1.0d0
                        (vt-scale
                          (vt-expt x 2.0d0)
                          (* 3.0d0 0.044715d0))))))
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

(defun vt-softmax-derivative (s)
  "Softmax 导数简化近似."
  (vt-* s (vt-- 1.0d0 s)))

(defun vt-mish-derivative (x)
  "mish 的精确导数."
  (let* ((sp (vt-softplus x))
         (tsp (vt-tanh sp))
         (sech2 (vt-- 1.0d0 (vt-* tsp tsp)))
         (sig (vt-sigmoid x)))
    (vt-+ tsp (vt-* x (vt-* sech2 sig)))))

(defclass dense (layer)
  ((in-dim :initarg :in-dim :reader dense-in-dim
           :type fixnum)
   (out-dim :initarg :out-dim :reader dense-out-dim
            :type fixnum)
   (weight-init :initarg :weight-init :initform nil
                :accessor dense-weight-init)
   (bias-init :initarg :bias-init :initform nil
              :accessor dense-bias-init)
   (use-bias :initarg :use-bias :initform t
             :reader dense-use-bias-p)
   (activation :initarg :activation :initform :none
               :accessor dense-activation
               :type (member :none :relu :leaky-relu :sigmoid
                             :tanh :gelu :swish :mish
                             :softplus :hard-tanh
                             :hard-sigmoid :linear))
   (weights :accessor dense-weights :type (or null vt))
   (bias :accessor dense-bias :type (or null vt))
   (dw :accessor dense-dw :type (or null vt))
   (db :accessor dense-db :type (or null vt))
   (input-cache :accessor dense-input-cache
                 :type (or null vt))
   (z-cache :accessor dense-z-cache :type (or null vt))
   (a-cache :accessor dense-a-cache :type (or null vt))
   (leaky-alpha :initarg :leaky-alpha :initform 0.01d0
                :accessor dense-leaky-alpha
                :type double-float))
  (:documentation "全连接层: y = activation(x · W + b)"))

(defun make-dense
    (out-dim &key (in-dim nil) activation use-bias
              weight-init bias-init leaky-alpha
              (name "dense") (trainable t))
  "构造全连接层."
  (make-instance 'dense
    :out-dim out-dim :in-dim in-dim
    :activation (or activation :none)
    :use-bias use-bias
    :weight-init weight-init
    :bias-init bias-init
    :leaky-alpha (or leaky-alpha 0.01d0)
    :name name :trainable trainable))

(defmethod forward ((l dense) input)
  (let* ((in-shape (vt-shape input))
         (batch-size
           (if (= (length in-shape) 1)
               1 (first in-shape)))
         (in-dim
           (or (dense-in-dim l)
               (if (= (length in-shape) 1)
                   (first in-shape)
                   (reduce #'* (rest in-shape)))))
         (w
           (or (dense-weights l)
               (let* ((fan-in in-dim)
                      (fan-out (dense-out-dim l))
                      (w-init
                        (or (dense-weight-init l)
                            (make-he-normal))))
                 (setf (slot-value l 'in-dim) in-dim)
                 (setf (dense-weights l)
                       (init-weight
                         w-init
                         (list in-dim fan-out)
                         :fan-in fan-in
                         :fan-out fan-out)))))
         (x-flat
           (if (= (length in-shape) 2)
               input
               (vt-reshape input
                           (list batch-size in-dim))))
         (z
           (if (dense-use-bias-p l)
               (let ((b
                       (or (dense-bias l)
                           (setf (dense-bias l)
                                 (init-bias
                                   (or (dense-bias-init l)
                                       (make-zeros-init))
                                   (list (dense-out-dim l)))))))
                 (vt-+ (vt-matmul x-flat w) b))
               (vt-matmul x-flat w)))
         (a
           (ecase (dense-activation l)
             ((:none :linear)             z)
             ((:relu relu)                (vt-relu z))
             ((:leaky-relu leaky-relu)    (vt-leaky-relu
					   z (dense-leaky-alpha l)))
             ((:sigmoid sigmoid)          (vt-sigmoid z))
             ((:tanh tanh)                (vt-tanh z))
             ((:gelu gelu)                (vt-gelu z))
             ((:swish swish)              (vt-swish z))
             ((:mish mish)                (vt-mish z))
             ((:softplus softplus)        (vt-softplus z))
             ((:hard-tanh hard-tanh)      (vt-hard-tanh z))
             ((:hard-sigmoid had-sigmoid) (vt-hard-sigmoid z)))))
    (setf (dense-input-cache l) x-flat)
    (setf (dense-z-cache l) z)
    (setf (dense-a-cache l) a)
    a))

(defmethod backward ((l dense) grad-output)
  (let* ((a-prev (dense-input-cache l))
         (w (dense-weights l))
         (act-kind (dense-activation l))
         (d-activation
           (ecase act-kind
             ((:none :linear) grad-output)
             ((:relu relu)  
               (vt-* grad-output
                     (vt-relu-derivative (dense-z-cache l))))
             ((:leaky-relu leaky-relu)
               (vt-* grad-output
                     (vt-leaky-relu-derivative
                       (dense-z-cache l)
                       (dense-leaky-alpha l))))
             ((:sigmoid sigmoid)
               (vt-* grad-output
                     (vt-sigmoid-derivative
                       (dense-a-cache l))))
             ((:tanh tanh)
               (vt-* grad-output
                     (vt-tanh-derivative (dense-a-cache l))))
             ((:gelu gelu)  
               (vt-* grad-output
                     (vt-gelu-derivative (dense-z-cache l))))
             ((:swish swish) 
               (vt-* grad-output
                     (vt-swish-derivative (dense-z-cache l))))
             ((:mish mish) 
               (vt-* grad-output
                     (vt-mish-derivative (dense-z-cache l))))
             ((:softplus softplus)  
               (vt-* grad-output
                     (vt-sigmoid (dense-z-cache l))))
             ((:hard-tanh hard-tanh)
               (let ((z (dense-z-cache l)))
                 (vt-* grad-output
                       (vt-map
                         (lambda (x)
                           (if (and (>= x -1.0d0)
                                    (<= x 1.0d0))
                               1.0d0 0.0d0))
                         z))))
             ((:hard-sigmoid had-sigmoid)
               (vt-* grad-output
                     (vt-hard-sigmoid-derivative
                       (dense-z-cache l))))))
         (dw
           (vt-matmul
             (vt-transpose a-prev) d-activation)))
    (setf (dense-dw l) dw)
    (when (dense-use-bias-p l)
      (setf (dense-db l)
            (vt-sum d-activation :axis 0)))
    (vt-matmul d-activation (vt-transpose w))))


(defmethod params ((l dense))
  (let ((result '()))
    (when (dense-weights l)
      (push (list "weights" (dense-weights l)
                  #'(lambda (v)
                      (setf (dense-weights l) v)))
            result))
    (when (and (dense-use-bias-p l) (dense-bias l))
      (push (list "bias" (dense-bias l)
                  #'(lambda (v)
                      (setf (dense-bias l) v)))
            result))
    (nreverse result)))

(defmethod grads ((l dense))
  (let ((result '()))
    (when (dense-dw l)
      (push (cons "weights" (dense-dw l)) result))
    (when (and (dense-use-bias-p l) (dense-db l))
      (push (cons "bias" (dense-db l)) result))
    (nreverse result)))


(defclass activation-layer (layer)
  ((kind :initarg :kind :initform :relu
         :accessor activation-kind
         :type (member :relu :leaky-relu :sigmoid :tanh
                       :gelu :swish :mish :softplus
                       :hard-tanh :hard-sigmoid :linear
                       :softmax :log-softmax))
   (leaky-alpha :initarg :leaky-alpha :initform 0.01d0
                :accessor act-leaky-alpha)
   (cache :initform nil :accessor act-cache)))

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
            ((:leaky-relu               leaky-relu)
              (vt-leaky-relu input      (act-leaky-alpha l)))
            ((:sigmoid sigmoid)         (vt-sigmoid input))
            ((:tanh tanh)               (vt-tanh input))
            ((:gelu gelu)               (vt-gelu input))
            ((:swish swish)             (vt-swish input))
            ((:mish mish)               (vt-mish input))
            ((:softplus softplus)       (vt-softplus input))
            ((:hard-tanh hard-tanh)     (vt-hard-tanh input))
            ((:hard-sigmoid hard-sigmoid) (vt-hard-sigmoid input))
            ((:linear :none)             input)
            ((:softmax softmax)         (vt-softmax input))
            ((:log-softmax log-softmax) (vt-log-softmax input)))))
    (setf (act-cache l) input)
    out))

(defmethod backward ((l activation-layer) grad-output)
  (let ((input (act-cache l)))
    (ecase (activation-kind l)
      ((:relu relu)
        (vt-* grad-output
              (vt-relu-derivative input)))
      ((:leaky-relu leaky-relu)
        (vt-* grad-output
              (vt-leaky-relu-derivative
                input (act-leaky-alpha l))))
      ((:sigmoid sigmoid)
        (vt-* grad-output
              (vt-sigmoid-derivative (vt-sigmoid input))))
      ((:tanh tanh)
        (vt-* grad-output
              (vt-tanh-derivative (vt-tanh input))))
      ((:gelu gelu)
        (vt-* grad-output (vt-gelu-derivative input)))
      ((:swish swish)
        (vt-* grad-output (vt-swish-derivative input)))
      ((:mish mish)
        (vt-* grad-output (vt-mish-derivative input)))
      ((:softplus softplus)
        (vt-* grad-output (vt-sigmoid input)))
      ((:hard-tanh hard-tanh)
        (vt-* grad-output
              (vt-map
                (lambda (x)
                  (if (and (>= x -1.0d0) (<= x 1.0d0))
                      1.0d0 0.0d0))
                input)))
      ((:hard-sigmoid hard-sigmoid)
        (vt-* grad-output
              (vt-hard-sigmoid-derivative input)))
      ((:linear :none) grad-output)
      ((:softmax softmax)
        (vt-* grad-output
              (vt-softmax-derivative (vt-softmax input))))
      ((:log-softmax log-softmax)
        (let* ((s (vt-softmax input))
               (sum-dy
                 (vt-sum grad-output
                         :axis -1 :keepdims t)))
          (vt-- grad-output (vt-* s sum-dy)))))))


(defclass flatten (layer)
  ((start-dim :initarg :start-dim :initform 1
              :reader flatten-start-dim))
  (:documentation
    "将 (batch, d1, d2, ...) 展平为 (batch, d1*d2*...)."))

(defun make-flatten (&key (start-dim 1) (name "flatten"))
  (make-instance 'flatten
    :start-dim start-dim :name name :trainable nil))

(defmethod forward ((l flatten) input)
  (let* ((shape (vt-shape input))
         (start (flatten-start-dim l))
         (pre-dim (reduce #'* (subseq shape 0 start)))
         (post-dim (reduce #'* (subseq shape start))))
    (vt-reshape
      input
      (list (if (> start 0) pre-dim 1) post-dim))))

(defclass residual (layer)
  ((block :initarg :block :reader residual-block))
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
    (vt-+ grad-output grad-block)))

(defmethod params ((l residual))
  (params (residual-block l)))

(defmethod grads ((l residual))
  (grads (residual-block l)))

(defmethod set-training! ((l residual) mode)
  (call-next-method)
  (when (residual-block l)
    (set-training! (residual-block l) mode)))
