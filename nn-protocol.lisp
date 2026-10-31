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

;;; ------------------------------------------------------------------
;;; dtype 契约工具（clnn 侧「类型域」的唯一事实来源）
;;; ------------------------------------------------------------------
;;; 背景：clvt 自 v0.4.x 起把逻辑 dtype 扩到 9 种，其中两族是本框架需要
;;; 明确表态的：
;;;   · :bool（CONVENTIONS §17）—— 比较/逻辑/谓词类函数的缺省返回类型。
;;;     它的物理存储是 1 字节 0/1，但参与算术时 vt-+ / vt-* 会改派逻辑内核
;;;     （True+True=True、True*True=True），vt-- 直接报错。因此 :bool 只能
;;;     作为「掩码/条件」出现在 API 边界，绝不能流进 autodiff 链。
;;;   · :complex128 / :complex64（CONVENTIONS §16）—— 本框架不支持复数参数。
;;;     静默放行的后果是梯度范数变成复数、`(> n clip)` 抛类型错、损失值变复数，
;;;     错误点离成因很远。
;;; 下面这组谓词集中定义类型域，避免各层文件各自 member 一张表；全部在
;;; nn-protocol.lisp 里定义，保证 embedding/loss/dense/model 加载时已可用。

(defparameter +integer-dtypes+
  '(:int8 :int16 :int32 :int64 :uint8 :uint16)
  "clvt 中的整数 dtype 全集（索引 / 类别标签的合法域）。")

(defparameter +compute-dtype+ :float64
  "**clnn 唯一的计算域 dtype**（契约文档 §2.0；三域划分见 §2.1：计算 / 索引 / 掩码）。

   为什么只留一个：clnn 是纯手写求导（§1.5），没有 autograd 统一管理精度传播，
   多精度只能靠人肉在每层的缓存、缓冲、累加器上逐一追踪 —— 而 clnn 里这类缓冲
   有几十处（状态、梯度累加器、running 统计量、掩码、零 dX …），漏掉一处就是
   静默的混合精度。v2.1 查出的缺口 G-11 就是这么攒出来的。

   性能上也站不住。实测（4000×4000，20 次，SBCL 2.6.10）：同 dtype 的 f64 逐元素
   加法 852 ms、f32 584 ms，而 **f32 张量与 f64 张量相加要 7165 ms —— 慢 8.4 倍**，
   因为每个算子都要先按提升表定结果 dtype、再分配 f64 临时数组、再逐元素转换。
   clvt 的 AVX-512 GEMM 微内核（src/gemm-avx512.lisp）也只发射 float64 指令，
   f32 走的是通用 sb-simd 路径。所以「混着跑」在 clnn 里既难维护又慢。

   要换精度就改这一个常量 —— 但前提是先把 §2.6 的逐层实测重跑一遍。")

(defparameter +float-dtypes+ '(:float32 :float64)
  "clvt 认识的浮点 dtype 全集。**注意：这不等于「clnn 两种都支持」** ——
   它只用于「识别出 f32 并把它规约到 +COMPUTE-DTYPE+」这一个目的。
   判断「是否已是计算域」请用 COMPUTE-DTYPE-P。")

(defun integer-dtype-p (dtype)
  "DTYPE 是否为整数（索引 / 标签的合法域）。"
  (member dtype +integer-dtypes+))

(defun float-dtype-p (dtype)
  "DTYPE 是否为浮点（含 f32）。只用于识别，不代表 f32 被接受为计算域。"
  (member dtype +float-dtypes+))

(defun compute-dtype-p (dtype)
  "DTYPE 是否已是 clnn 的计算域（+COMPUTE-DTYPE+）。"
  (eq dtype +compute-dtype+))

(defun bool-dtype-p (dtype)
  "DTYPE 是否为 :bool（掩码 / 条件专用，禁止进入 autodiff 链）。"
  (eq dtype :bool))

(defun complex-dtype-p (dtype)
  "DTYPE 是否为复数（本框架不支持）。"
  (member dtype '(:complex128 :complex64)))

(defun value-float-dtype (dtype)
  "把任意 DTYPE 规约到计算域。**恒返回 +COMPUTE-DTYPE+**。

   DTYPE 参数保留下来只是为了调用点不必改签名，也让「将来若真要支持多精度」
   有一个唯一的收敛点可改 —— 但那时要重跑 §2.6 的逐层验收。

   整数与 :bool 在这里也会被折成 f64 —— 但这不该发生：它们属于索引域/掩码域，
   不该走到「承接浮点计算结果」的位置。真正把关的是 ASSERT-INDEX-DTYPE
   与各层的类型校验。"
  (declare (ignore dtype))
  +compute-dtype+)


;;; ------------------------------------------------------------------
;;; 计算域的入口规约（三域模型的唯一闸门）
;;; ------------------------------------------------------------------
;;; 设计意图：**转换只发生在边界，内部永不再转**。
;;;
;;; 之所以不是「每层各写一遍 astype」，是因为层有 20 个、每个都有 forward 与
;;; backward 两个入口，逐处改必然漏。这里用 CLOS 的 :AROUND 方法（见文件末尾）
;;; 拦截 (LAYER) 上的全部调用，规约逻辑只有一份。
;;;
;;; 三种「不规约」的情形，都属另外两个域：
;;;   · 整数（:int8…:int64/:uint8/:uint16）—— 索引域，embedding / gather / 标签
;;;   · :bool —— 掩码域，dropout 掩码 / vt-where 条件 / 注意力 mask
;;;   · 非张量（fixnum / double-float 标量 / list）—— clvt 的弱标量机制会处理，
;;;     转成 0 维张量反而破坏 NEP 50（这正是 clvt §16.31.7 修掉的那个坑）
;;;
;;; 关于警告：只报一次。同一个 f32 输入在训练循环里会被调用上万次，每次都
;;; warn 会把终端刷死；而「报过就够了」——用户要么改数据源，要么就接受这一次
;;; 转换的成本。

(defvar *compute-dtype-warned* nil
  "已经警告过的 dtype 列表。全局只报一次（不分层、不分调用方），避免刷屏。

   为什么键里**不含层名**：一个 20 层的模型如果每层都报一次，就是 20 条 5 行
   的警告 —— 那不是信息，是噪音，用户的下一个动作一定是 `(setf *warnings* nil)`
   而不是去改数据管道。所以第一次遇到 f32 就报，文案里带上**触发的那个层**，
   之后整场会话静默。想知道还有哪些层在收 f32，把 *COMPUTE-DTYPE-WARNED*
   清空再跑一次即可。")

(defun ensure-compute-dtype (x &optional (who "clnn"))
  "把浮点张量规约到 +COMPUTE-DTYPE+；索引/掩码/标量原样返回。

   这是三域模型的**唯一闸门**：所有进入层计算的张量都从这里过一道。
   已经是计算域的对象原样返回（零分配），所以重复调用是无害的 ——
   SEQUENTIAL 调子层、子层再走自己的 :AROUND，一共过两次，代价只是一次
   VT-DTYPE 加一次 EQ。"
  (cond
    ;; 非张量：标量（含整数/双精度/单精度）与 list 由调用方处理，
    ;; 这里不动 —— 把标量包成 0 维张量会破坏 clvt 的弱标量提升。
    ((not (vt-p x)) x)
    (t
     (let ((dt (vt-dtype x)))
       (cond
         ((eq dt +compute-dtype+) x)
         ;; 索引域 / 掩码域：原样透传。它们的 dtype 契约由各层与
         ;; ASSERT-INDEX-DTYPE 单独把关，不该在这里被折成 f64。
         ((or (integer-dtype-p dt) (bool-dtype-p dt)) x)
         ;; 计算域的窄版本：规约一次，并提示一次。
         ((eq dt :float32)
          (unless (member :float32 *compute-dtype-warned* :test #'eq)
            (push :float32 *compute-dtype-warned*)
            (warn "~a: 收到 :float32 张量，已转换为 :float64（clnn 的单一计算域，~%  ~
                   契约文档 §2.0）。本提示每次会话只出现一次。~%  ~
                   若在训练循环里反复传入 f32，请在数据管道里一次性转换后复用，~%  ~
                   否则每步都要付一次整批转换的成本（1600 万元素约 50 ms）。"
                  who))
          (vt-astype x +compute-dtype+))
         ;; :float64 / 其他（含 :bool 已在上支拦截）—— 原样
         (t x))))))

(defun ensure-compute-input (x &optional (who "clnn"))
  "ENSURE-COMPUTE-DTYPE 的多输入版本：list 逐项规约，单张量直接规约。

   SDPA 与 MHA 的 forward/backward 收的是 (Q K V) 三元 list，
   其余层收单个张量。这里统一，避免每个 :AROUND 各判一次。"
  (if (listp x)
      (mapcar (lambda (e) (ensure-compute-dtype e who)) x)
      (ensure-compute-dtype x who)))

(defun assert-index-dtype (tensor what)
  "校验 TENSOR 是整数张量，否则报出「谁在期待索引」的确定性错误。

   为什么必须显式校验（而不是交给 VT-TAKE）：clvt 对 `:bool` 索引**不会报错**，
   而是把 0/1 当成合法下标（`(vt-take w bool-indices :axis 0)` 恒取第 1 行）。
   于是 embedding / gather 在布尔掩码被误传成索引时「静默取错行」，直到
   backward 里的 (coerce … 'fixnum) 才炸，错误点与成因相隔一整步。
   浮点索引在 clvt 侧会报错，但文案是 clvt 的口径；这里统一成 clnn 的口径。"
  (unless (integer-dtype-p (vt-dtype tensor))
    (error "~a: 索引/标签必须是整数 dtype（:int8…:int64/:uint8/:uint16），~
            实际为 ~a。布尔掩码请先用 (first (vt-nonzero mask))（1-D）或
            (vt-extract mask x)（展平语义）转成整数索引，不要直接把比较结果当索引
            使用 —— clvt 的 vt-take 会把布尔值 0/1 当成合法下标，静默取错行。"
           what (vt-dtype tensor)))
  tensor)

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
  (:documentation "使用优化器更新该组件（含其子层）的参数.")

  ;; 手写求导架构里没有 autograd 能替你把「参数 ↔ 梯度」配对：更新必须显式
  ;; 传出 (PARAMS C) 与 (GRADS C) 两张同构表。LAYER 上的真方法见文件末尾
  ;; （必须排在 OPTIMIZER-STEP 之后，否则编译期会报未定义函数）。
  ;;
  ;; **默认方法为什么是空操作**：如果 T 上只有这个空方法而没有 LAYER 上的真方法，
  ;; （公开导出的）`(update! 某层 opt)` 会静默什么都不做 —— 用户以为参数更新了，
  ;; 其实没有。真方法在下面，空方法只留给非层对象。
  (:method ((c t) optimizer)
    (declare (ignore optimizer))
    (values)))

;;; ------------------------------------------------------------------
;;; 梯度/缓存管理协议（必须在层实现之前声明）
;;; ------------------------------------------------------------------
;;; 说明：ZERO-GRAD-CHILDREN / ZERO-GRAD! / TRANSIENT-CACHE-SLOTS /
;;; CLEAR-FORWARD-CACHE! / CLEAR-STEP-CACHES! 属于「协议」的一部分，
;;; 具体方法实现放在 nn-model.lisp。历史上这几个 defgeneric 写在
;;; nn-model.lisp（最后加载的文件）里，而 nn-layers-attention.lisp
;;; 已经先为 ZERO-GRAD-CHILDREN 定义了方法 —— 这会让 SBCL 在加载
;;; nn-model.lisp 时告警 "redefining ... in DEFGENERIC"，并且在
;;; nn-model.lisp 未能加载时留下一个没有默认方法的残缺泛型。
;;; 因此把这些泛型的声明前移到协议层。

(defgeneric zero-grad-children (component)
  (:documentation "返回需要递归清零梯度的直接子组件列表。默认返回 NIL。"))

(defgeneric zero-grad! (component)
  (:documentation "将组件及其所有子层中的梯度张量置为 NIL。"))

(defgeneric transient-cache-slots (component)
  (:documentation "返回可以在 forward 与 backward 之间安全丢弃的缓存 slot。"))

(defgeneric clear-forward-cache! (component)
  (:documentation "只清理 TRANSIENT-CACHE-SLOTS 列出的缓存（可随时调用）。"))

(defgeneric clear-step-caches! (component)
  (:documentation "清除全部前向缓存以回收内存（只能在 forward+backward 之后调用）。"))


(defun make-loss (&rest args)
  "损失函数基类的通用构造器（与 MAKE-OPTIMIZER / MAKE-LAYER 对称）。
   日常请使用具体子类的构造器，如 MAKE-MSE-LOSS / MAKE-CE-LOSS。"
  (apply #'make-instance 'loss args))

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

;;; ------------------------------------------------------------------
;;; 优化步的边界：一次 UPDATE! = 一步
;;; ------------------------------------------------------------------
;;; STEP-COUNT 是「已执行多少个参数更新步」，不是「OPTIMIZER-STEP 被调了
;;; 多少次」。两者在旧实现里混为一谈，于是同一件事有两种结果：
;;;   (update! model opt)          ; 容器 → 内部一次 optimizer-step → t = 1
;;;   (update! layer1 opt)
;;;   (update! layer2 opt)         ; 逐层 → 内部两次 → t = 2
;;; 而 Adam 的偏差校正 1/(1-β^t) 与 Adagrad 的 1/(1+λt) 都直接吃 t：
;;; 逐层写法会把 t 算大，校正失效、有效学习率偏小；梯度裁剪也变成逐层各剪
;;; 一次（每次按该层的范数），与「全模型全局范数」不是一回事。
;;;
;;; 修法是把步号的所有权上移：**UPDATE! 授予步号，OPTIMIZER-STEP 只读**。
;;; 这样两种写法都自洽 —— 一次 UPDATE! 就是一步，逐层调 N 次就是 N 步
;;; （那本来就是 N 次独立更新，只是过去没人这么告诉用户）。
;;;
;;; 确实需要「分片更新但算一步」（显存受限、或多 head 共享一个优化器）时，
;;; 用 WITH-OPTIMIZER-STEP 把多次 UPDATE! 圈进同一个步。

(defvar *optimizer-step* nil
  "UPDATE! / WITH-OPTIMIZER-STEP 绑定的 (OPTIMIZER . STEP-NUMBER)。
   NIL 表示不在任何步作用域内（用户直接调 OPTIMIZER-STEP）。")

(defun optimizer-step-number (opt)
  "本次 OPTIMIZER-STEP 应使用的步号（1 起）。
   在 UPDATE! 作用域内读到授予值，否则自行推进 OPT 的计数（兼容直接调用）。"
  (if (and *optimizer-step* (eq (car *optimizer-step*) opt))
      (cdr *optimizer-step*)
      (incf (optimizer-step-count opt))))

(defmacro with-optimizer-step ((optimizer) &body body)
  "把 BODY 内的所有 UPDATE! 归并为同一个优化步。

   用例（分片更新、共享一个优化器的多个 head）：
     (with-optimizer-step (opt)
       (update! trunk opt)
       (update! head-a opt)
       (update! head-b opt))     ; 三步更新，但只算一个优化步

   注意：梯度裁剪（:GRAD-CLIP）仍然按每次 OPTIMIZER-STEP 传入的梯度表
   计算范数，所以分片更新时每片各自裁剪。需要真正的全局范数，用
   CLIPPED-GRADIENT-UPDATE!（它会先对整个模型缩放一次梯度）。"
  (let ((o (gensym "OPT")))
    `(let* ((,o ,optimizer)
            (*optimizer-step*
              (if (and *optimizer-step* (eq (car *optimizer-step*) ,o))
                  *optimizer-step*          ; 外层已授予，沿用（分片合并）
                  (cons ,o (incf (optimizer-step-count ,o))))))
       ,@body)))

(defgeneric optimizer-zero-grad! (opt)
  (:documentation "清空所有梯度缓存.")
  (:method ((o optimizer)) (values)))

(defparameter +activation-names+
  '(:none :linear :relu :leaky-relu :sigmoid :tanh :gelu :swish :mish
    :softplus :hard-tanh :hard-sigmoid :softmax :log-softmax)
  "本库支持的激活函数名称（:none / :linear 等价）。

   注意：DENSE 层支持其中除 :softmax / :log-softmax 之外的全部取值
   （softmax 属于 ACTIVATION-LAYER 的职责）。")

(defclass initializer ()
  ((name :initarg :name
	 :initform "init"
	 :accessor initializer-name
	 :type string))
  (:documentation "参数初始化器基类."))

(defun make-initializer (&rest args)
  "初始化器基类的通用构造器（与 MAKE-OPTIMIZER / MAKE-LAYER 对称）。
   日常请使用具体子类的构造器，如 MAKE-HE-NORMAL / MAKE-XAVIER-UNIFORM。"
  (apply #'make-instance 'initializer args))

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

(defclass stop-gradient-node (layer)
  ()
  (:documentation
   "阻断梯度回流的层：前向直通（返回 input），反向返回 NIL。
   用法（推荐放入 sequential，让 forward/backward 自动串接）：
     (seq-add! model (make-stop-gradient))
   也可独立使用：
     (forward (make-stop-gradient) x)   ; => x
     (backward (make-stop-gradient) g)  ; => nil"))

(defun make-stop-gradient (&key (name "stop-gradient") (trainable nil))
  (make-instance 'stop-gradient-node
                 :name name :trainable trainable))

(defun vt-stop-gradient (&optional tensor)
  "向后兼容入口：返回一个 STOP-GRADIENT-NODE 层。
   TENSOR 参数被保留仅为兼容旧调用，不再绑定到节点。"
  (declare (ignore tensor))
  (make-stop-gradient))

(defmethod forward ((node stop-gradient-node) input)
  input)

(defmethod backward ((node stop-gradient-node) grad)
  "阻断梯度：返回与 GRAD 同形状的全零张量。
   这样 SEQUENTIAL 等通用容器可以把零梯度继续传给上游层，
   而不会因 NIL 崩溃。数学上等价于『截断』：上游拿到的梯度是零，
   参数不会被这一步更新。

   用 ZEROS-LIKE 而不是 (VT-ZEROS (VT-SHAPE GRAD))：后者按 clvt 的缺省
   造 float64，float32 模型里会把上游的梯度链静默提升成双精度。"
  (if grad
      (vt-zeros-like grad)
      nil))


;;; ------------------------------------------------------------------
;;; UPDATE! 的 LAYER 方法（必须排在 OPTIMIZER-STEP 之后）
;;; ------------------------------------------------------------------
;;; 纯手写求导架构里，「更新」不是框架的自动行为，而是两件事的显式组合：
;;;   grads  ← backward 写进各层的梯度槽
;;;   params ← 优化器按 (参数表, 梯度表) 逐位更新
;;; 没有任何 autograd 记录过「谁依赖谁」，所以这两张表必须由层自己保证同构
;;; （§4.4）。UPDATE! 就是把这条链路收成一句话，作用域可以是整个模型，也可以
;;; 是任意子树（例如只训练某个 head）。
(defmethod update! ((c layer) optimizer)
  ;; WITH-OPTIMIZER-STEP 已在外层授予步号时，这里不再推进 —— 这正是
  ;; 「分片更新算一步」与「逐层更新算 N 步」两条路径的唯一分叉点。
  (with-optimizer-step (optimizer)
    (optimizer-step optimizer (params c) (grads c))))


;;; ------------------------------------------------------------------
;;; 三域模型的闸门：FORWARD / BACKWARD / COMPUTE-LOSS 的 :AROUND
;;; ------------------------------------------------------------------
;;; 为什么用 :AROUND 而不是在每个 defmethod 里加一行：
;;;   层有 20 个、每个都有 forward 与 backward 两个入口（合计 40 处），
;;;   逐处加必然漏一处，而漏掉的那一处就是静默的混合精度。:AROUND 挂在
;;;   (LAYER) 和 (LOSS) 这两个基类上，无论将来新增多少层都自动被覆盖 ——
;;;   这是「扩展者自查清单」第 1 条（新层必须继承 layer）的一个实际回报。
;;;
;;; 为什么 BACKWARD 也要规约：上游传下来的 grad 若为 f32（例如用户手工构造
;;; 的损失梯度），每层的 dX 计算都会带着它跑一遍提升表。入口折一次，
;;; 内部就永远是 f64。
;;;
;;; COMPUTE-LOSS 的 TARGET 也要过闸门。target 有两种身份：
;;;   · CE 的整数类别标签 —— 索引域，必须原样透传（否则 §10.2 的整数标签
;;;     校验失效，且 f64 标签无法再当索引）；
;;;   · mse / bce / huber / cosine / kl 的数值回归目标 —— 计算域，若留着 f32
;;;     就会和 f64 的 predicted 混跑（实测 mse 损失 4.8×、梯度 6.6×）。
;;; 两种情况不需要按损失类型分派：ENSURE-COMPUTE-DTYPE 本身就按域分流
;;; （整数与 :bool 原样放行，只折 :float32），一个闸门同时满足两者。

(defun %layer-tag (l)
  "给警告文案用的层标识：优先用户起的名字，否则类名。"
  (let ((n (ignore-errors (layer-name l))))
    (if (and (stringp n) (plusp (length n)))
        n
        (string-downcase (symbol-name (class-name (class-of l)))))))

(defmethod forward :around ((l layer) input)
  (call-next-method l (ensure-compute-input input (%layer-tag l))))

(defmethod backward :around ((l layer) grad-output)
  (call-next-method l (ensure-compute-input grad-output (%layer-tag l))))

(defmethod compute-loss :around ((fn loss) predicted target)
  (call-next-method fn (ensure-compute-dtype predicted "compute-loss")
                    (ensure-compute-dtype target "compute-loss")))

(defmethod compute-loss-gradient :around ((fn loss) predicted target)
  (call-next-method fn (ensure-compute-dtype predicted "compute-loss-gradient")
                    (ensure-compute-dtype target "compute-loss-gradient")))


;;; ------------------------------------------------------------------
;;; 参数入口：初始化器与模型加载
;;; ------------------------------------------------------------------
;;; 参数是计算域的「源头」—— 一旦参数是 f32，整条前向/反向都会带着它跑提升表。
;;; 这里用 :AROUND 包住 INIT-WEIGHT / INIT-BIAS 两个泛型，一处覆盖全部 10 个
;;; 初始化器（he/xavier/orthogonal/zeros/ones/constant/kaiming/truncated…），
;;; 新增初始化器自动被覆盖。
;;;
;;; 注意：**不提供 :dtype 参数**。不是漏了，是刻意的 —— 参数只有一个合法精度，
;;; 给个 :dtype 就等于承诺「可以选」，而选错的代价由使用者承担（混合精度、
;;; 慢 8.4 倍、手工追缓冲）。要换精度请改 +COMPUTE-DTYPE+，那是全局一致的。

(defmethod init-weight :around (init shape &key fan-in fan-out layout)
  (declare (ignorable fan-in fan-out layout))
  (ensure-compute-dtype (call-next-method) "init-weight"))

(defmethod init-bias :around (init shape)
  (ensure-compute-dtype (call-next-method) "init-bias"))

(defun normalize-params! (component)
  "把 COMPONENT（含子树）的参数就地规约到 +COMPUTE-DTYPE+，返回改动个数。

   用途只有一个：**从外部接进来的模型**。序列化存档、手工构造的权重表、
   从别的库转过来的参数，都可能在存档时是 f32。加载后跑一次，
   后续所有层都不必再转换（闸门只会在第一次把它们逐层折一遍，纯浪费）。

   **为什么不碰梯度**：契约 M4 规定梯度槽是覆盖写 —— 下一次 backward 会
   整槽重建，规约旧值只是给马上要被丢弃的对象做无用的 astype。"
  (let ((n 0))
    (dolist (pg (params component))
      (let ((v (third pg)) (setter (fourth pg)))
        (when (and (vt-p v)
                   (float-dtype-p (vt-dtype v))
                   (not (compute-dtype-p (vt-dtype v))))
          (when setter
            (funcall setter (vt-astype v +compute-dtype+))
            (incf n)))))
    n))
