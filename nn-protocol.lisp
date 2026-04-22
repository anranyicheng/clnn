(in-package #:nn)

(defvar *training-mode* t
  "全局训练/推理模式开关. 推理时 Dropout/BatchNorm 行为不同.")

(defgeneric set-training! (component mode)
  (:documentation "设置组件的训练/推理模式.")
  (:method ((c t) (mode t)) (declare (ignore c mode))))

(defgeneric training-p (component)
  (:documentation "查询组件是否处于训练模式.")
  (:method ((c t)) *training-mode*))


(defclass layer ()
  ((name :initarg :name
	 :initform ""
	 :accessor layer-name
	 :type string
	 :documentation "层名称")
   (trainable :initarg :trainable
	      :initform t
	      :accessor layer-trainable-p
	      :type boolean
	      :documentation "是否参与训练")
   (training :initform *training-mode*
	     :accessor layer-training
	     :type boolean :documentation "内部训练状态"))
  (:documentation "所有神经网络层的基类."))

(defun make-layer (&rest args)
  "通用层构造函数."
  (apply #'make-instance 'layer args))

(defmethod set-training! ((l layer) mode)
  (setf (layer-training l) mode)
  (setf *training-mode* mode))

(defmethod training-p ((l layer)) (layer-training l))

(defgeneric forward (component input)
  (:documentation "前向传播.")
  (:method ((c t) input) input))

(defgeneric backward (component grad-output)
  (:documentation "反向传播.")
  (:method ((c t) grad-output) grad-output))

(defgeneric params (component)
  (:documentation "返回可训练参数及 setter.")
  (:method ((c t)) '()))

(defgeneric grads (component)
  (:documentation "返回梯度列表.")
  (:method ((c t)) '()))

(defgeneric update! (component optimizer)
  (:documentation "使用优化器更新参数.")
  (:method ((c t) optimizer) (declare (ignore optimizer))))


(defclass loss ()
  ((name :initarg :name
	 :initform "loss"
	 :accessor loss-name
	 :type string)
   (reduction :initarg :reduction
	      :initform :mean
	      :accessor loss-reduction
	      :type (member :mean :sum :none)
	      :documentation "归约方式: :mean / :sum / :none"))
  (:documentation "损失函数基类."))

(defgeneric compute-loss (loss-fn predicted target)
  (:documentation "计算损失值（标量张量）."))

(defgeneric compute-loss-gradient (loss-fn predicted target)
  (:documentation "计算损失关于 predicted 的梯度."))


(defclass optimizer ()
  ((lr :initarg :lr
       :initform 1e-3
       :accessor optimizer-lr
       :type double-float
       :documentation "学习率")
   (grad-clip :initarg :grad-clip
	      :initform 0.0d0
	      :accessor optimizer-grad-clip
	      :type double-float
	      :documentation "梯度裁剪阈值 (0=不裁剪)")
   (weight-decay :initarg :weight-decay
		 :initform 0.0d0
		 :accessor optimizer-weight-decay
		 :type double-float
		 :documentation "L2 正则化系数")
   (step-count :initform 0
	       :accessor optimizer-step-count
	       :type fixnum
	       :documentation "已执行步数")
   (state-registry :initform (make-hash-table :test #'equal)
		   :initarg :state-registry
		   :accessor optimizer-state-registry))
  (:documentation "优化器基类."))

(defun make-optimizer (&rest args)
  (apply #'make-instance 'optimizer args))

(defgeneric optimizer-step (opt param-list grad-list)
  (:documentation "对参数列表执行一步优化更新.
PARAM-LIST = ((name tensor setter-fn) ...)
GRAD-LIST = ((name . tensor) ...)"))

(defgeneric optimizer-zero-grad! (opt)
  (:documentation "清空所有梯度缓存.")
  (:method ((o optimizer)) (values)))


(defclass initializer ()
  ((name :initarg :name
	 :initform "init"
	 :accessor initializer-name
	 :type string))
  (:documentation "参数初始化器基类."))

(defgeneric init-weight (init shape &key fan-in fan-out)
  (:documentation "根据 shape 初始化权重张量."))

(defgeneric init-bias (init shape)
  (:documentation "初始化偏置张量.")
  (:method ((init initializer) shape)
    (declare (ignore init)) (vt-zeros shape)))


(defclass regularizer ()
  ((name :initarg :name
	 :initform "regularizer"
	 :accessor regularizer-name))
  (:documentation "参数正则化器基类."))

(defgeneric regularizer-penalty (reg param-list)
  (:documentation "计算正则化惩罚项（标量）."))
