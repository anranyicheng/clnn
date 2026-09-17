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
   (training :initarg :training
             :accessor layer-training
             :type (or null boolean)
	     :documentation "内部训练状态"))
  (:documentation "所有神经网络层的基类."))

(defun make-layer (&rest args)
  "通用层构造函数."
  (apply #'make-instance 'layer args))

(defmethod set-training! ((l layer) mode)
  "仅设置本层的局部 training 状态, 不污染全局 *training-mode*."
  (setf (layer-training l) (and mode t)))

(defun set-global-training! (mode)
  "显式设置全局训练/推理开关 (影响默认 training-p 方法)."
  (setf *training-mode* mode))

(defmacro with-training (mode &body body)
  "在 BODY 执行期间动态绑定 *training-mode*=MODE, 离开作用域自动恢复."
  `(let ((*training-mode* ,mode))
     ,@body))

(defmethod training-p ((l layer))
  "查询本层是否处于训练模式。
   语义：
     - 若本层从未 set-training! 过（training 槽 unbound），
       回退到全局 *training-mode*；因此：
         (with-training nil (training-p some-layer))  => NIL
     - 若本层已 set-training! 过，返回局部值，
       不再受 *training-mode* 影响。"
  (if (slot-boundp l 'training)
      (layer-training l)
      *training-mode*))

(defun reset-training! (layer)
  "清除本层的显式 training 覆盖，使其重新跟随全局 *training-mode*。
   常用于在 set-model-training! 之后，重新让 with-training 生效。"
  (when (slot-boundp layer 'training)
    (slot-makunbound layer 'training))
  layer)

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

(defgeneric grad-slots (component)
  (:documentation "返回组件自身的梯度 slot 名符号列表（每个元素是一个符号，
即该组件中存储梯度张量的 slot 名，如 DW、DB）。
容器层（sequential/residual/transformer-block）返回 nil，
通过 zero-grad-children 递归子层；叶子层（dense/conv2d/lstm/...）
返回自己的梯度 slot 名列表。zero-grad! 据此把对应 slot 置 nil。
用户自定义新层只需实现此方法即可被正确清零，无需修改白名单。")
  (:method ((c t)) '()))

(defgeneric cache-slots (component)
  (:documentation "返回组件自身需要在前向传播结束后清空的缓存 slot 名符号列表。
设计原则与 grad-slots 完全同构：
  - 容器层（sequential / residual / transformer-block）返回 '()，
    由 clear-forward-cache! 负责递归子层；
  - 叶子层（dense / conv2d / lstm / ...）显式列出自己持有的前向缓存 slot；
  - 用户自定义新层只需实现此方法即可被正确清理，无需修改任何字符串白名单。
注意：只列出『前向结束后可以丢弃的中间缓存』。
     反向传播依赖的整型/配置状态（如 batch-norm 的 BATCH-SIZE、
     layer-norm 的 NORM-SIZE、rnn-cell 的 STATE）不属于缓存，不要列入。")
  (:method ((c t)) '()))

(defgeneric update! (component optimizer)
  (:documentation "使用优化器更新参数.")
  (:method ((c t) optimizer)
    (declare (ignore optimizer))))


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
       :initform 1d-3
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

(defgeneric init-weight (init shape &key fan-in fan-out layout)
  (:documentation "根据 shape 初始化权重张量。
   FAN-IN / FAN-OUT 显式提供时优先使用 (跳过自动推断)。
   LAYOUT 仅在 shape 为 2D 时生效, 指定维度的语义:
     :in-out (默认) -> shape = (in, out)    [PyTorch Linear / Dense]
     :out-in         -> shape = (out, in)   [PyTorch Conv2d 权重展平后]
   对 1D 和 ND (rank >= 3) shape, LAYOUT 无影响。
   ND 始终按卷积 (out, in, k1, k2, ...) 处理。"))

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

(defclass l1-regularizer (regularizer)
  ((lambda :initarg :lambda
           :initform 1.0d-4
           :accessor l1-lambda
           :type double-float))
  (:documentation "L1 正则化：penalty = λ · Σ|w|。"))

(defun make-l1-regularizer (&key (lambda 1.0d-4))
  (make-instance 'l1-regularizer :lambda lambda :name "l1"))

(defmethod regularizer-penalty ((reg l1-regularizer) param-list)
  (* (l1-lambda reg)
     (loop for p in param-list
           for tensor = (third p)
           sum (if tensor (vt-item (vt-sum (vt-abs tensor))) 0.0d0))))

(defclass l2-regularizer (regularizer)
  ((lambda :initarg :lambda
           :initform 1.0d-4
           :accessor l2-lambda
           :type double-float))
  (:documentation "L2 正则化：penalty = λ · Σw²。"))

(defun make-l2-regularizer (&key (lambda 1.0d-4))
  (make-instance 'l2-regularizer :lambda lambda :name "l2"))

(defmethod regularizer-penalty ((reg l2-regularizer) param-list)
  (* (l2-lambda reg)
     (loop for p in param-list
           for tensor = (third p)
           sum (if tensor (vt-item (vt-sum (vt-square tensor))) 0.0d0))))

(defclass elastic-regularizer (regularizer)
  ((l1-lambda :initarg :l1-lambda
              :initform 1.0d-4
              :accessor elastic-l1-lambda
              :type double-float)
   (l2-lambda :initarg :l2-lambda
              :initform 1.0d-4
              :accessor elastic-l2-lambda
              :type double-float))
  (:documentation "弹性正则化：penalty = λ1 · Σ|w| + λ2 · Σw²。"))

(defun make-elastic-regularizer (&key (l1-lambda 1.0d-4) (l2-lambda 1.0d-4))
  (make-instance 'elastic-regularizer
                 :l1-lambda l1-lambda :l2-lambda l2-lambda :name "elastic"))

(defmethod regularizer-penalty ((reg elastic-regularizer) param-list)
  (+ (* (elastic-l1-lambda reg)
        (loop for p in param-list
              for tensor = (third p)
              sum (if tensor (vt-item (vt-sum (vt-abs tensor))) 0.0d0)))
     (* (elastic-l2-lambda reg)
        (loop for p in param-list
              for tensor = (third p)
              sum (if tensor (vt-item (vt-sum (vt-square tensor))) 0.0d0)))))

(defclass stop-gradient-node ()
  ((input :initarg :input :reader sg-input))
  (:documentation
   "阻断梯度回流的包装器节点。前向传播直通，反向传播返回 NIL。"))

(defun vt-stop-gradient (tensor)
  "对外暴露的 API：把一个张量包裹成断梯度节点。"
  (make-instance 'stop-gradient-node :input tensor))

(defmethod forward ((node stop-gradient-node) input)
  (declare (ignore input))
  (sg-input node))

(defmethod backward ((node stop-gradient-node) grad)
  "拦截传进来的梯度 grad，直接丢弃，不向 input 传递任何东西。"
  (declare (ignore grad))
  nil)
