(in-package #:nn)

(defclass sequential (layer)
  ((layers :initform '()
	   :initarg :layers
	   :accessor seq-layers
           :type list)
   (layer-names :initform '()
		:initarg :layer-names
		:accessor seq-layer-names))
  (:documentation "顺序模型."))

(defun make-sequential (&key (name "sequential"))
  (make-instance 'sequential
		 :name name :trainable t))

(defun seq-add! (model layer-or-layers)
  "向 Sequential 模型添加层."
  (let ((layers (if (listp layer-or-layers)
                    layer-or-layers
                    (list layer-or-layers))))
    ;; 用 append 而非 push+nreverse，避免 nreverse 破坏 cons 单元导致
    ;; 多次 add! 后层顺序错乱（原始 bug）
    (setf (seq-layers model)
          (append (seq-layers model) layers))
    (dolist (l layers)
      (setf (seq-layer-names model)
            (append (seq-layer-names model)
                    (list (or (layer-name l) "")))))
    model))

(defun seq-insert! (model index layer)
  "在指定位置插入层."
  (let* ((layers (seq-layers model))
         (new-layers
           (append (subseq layers 0 index)
                   (list layer)
                   (subseq layers index))))
    (setf (seq-layers model) new-layers))
  model)

(defmethod forward ((m sequential) input)
  (let ((current input))
    (dolist (layer (seq-layers m))
      (setf current (forward layer current)))
    current))

(defmethod backward ((m sequential) grad-output)
  (let ((current grad-output)
        (reversed (reverse (seq-layers m))))
    (dolist (layer reversed)
      (setf current (backward layer current)))
    current))

(defmethod params ((m sequential))
  (let ((all '()))
    (dolist (layer (seq-layers m))
      (setf all (nconc all (params layer))))
    all))

(defmethod grads ((m sequential))
  (let ((all '()))
    (dolist (layer (seq-layers m))
      (setf all (nconc all (grads layer))))
    all))

(defmethod set-training! ((m sequential) mode)
  (call-next-method)
  (dolist (layer (seq-layers m))
    (set-training! layer mode)))


(defun model-forward (model input)
  "通用前向传播."
  (forward model input))

(defun model-backward (model grad-output)
  "通用反向传播."
  (backward model grad-output))

(defun model-update! (model optimizer)
  "使用优化器更新模型参数."
  (let ((param-list (params model))
        (grad-list (grads model)))
    (optimizer-step optimizer param-list grad-list)))

(defun collect-all-layers (component)
  "递归收集所有子层."
  (cond
    ((typep component 'sequential)
     (mapcan #'collect-all-layers
             (seq-layers component)))
    ((typep component 'residual)
     (cons component
           (collect-all-layers
            (residual-block component))))
    ((typep component 'transformer-block)
     (cons component
           (mapcan
            #'collect-all-layers
            (list (tb-mha component)
                  (tb-ffn1 component)
                  (tb-ffn2 component)
                  (tb-ln1 component)
                  (tb-ln2 component)))))
    (t (list component))))

(defun set-model-training! (model mode)
  "设置整个模型的训练/推理模式."
  (set-training! model mode)
  (dolist (layer (collect-all-layers model))
    (set-training! layer mode)))


(defun param-count (model &key trainable-only)
  "统计模型参数量."
  (let ((total 0))
    (labels
	((count-in (obj)
           (cond
             ((typep obj 'sequential)
              (dolist (l (seq-layers obj))
                (count-in l)))
             (t
              (when (or (not trainable-only)
                        (layer-trainable-p obj))
                (dolist (p (params obj))
                  ;; 协议修复: 取第三个元素
                  (let ((tensor (third p)))
                    (when tensor
                      (incf total
                            (reduce #'* (vt-shape tensor)))))))))))
      (count-in model))
    total))

(defun compute-grad-norm (model)
  "计算所有梯度的 L2 范数."
  (let ((all-grads (grads model))
        (sq-sum 0.0d0))
    (dolist (g all-grads)
      (let ((tensor (cdr g)))
        (when tensor
          (incf sq-sum
               (vt-item (vt-sum (vt-square tensor)))))))
    (sqrt sq-sum)))


(defun scale-all-grads! (component factor)
  "递归将所有梯度张量乘以 factor (原地修改层内部状态)."
  (labels ((scale-slots (obj)
	     (let ((class (class-of obj)))
	       (dolist (slot (c2mop:class-slots class))
		 (let ((name (c2mop:slot-definition-name slot)))
		   (when (and (slot-boundp obj name)
			      (grad-slot-p name))
		     (let ((val (slot-value obj name)))
		       (when (vt-p val)
			 (setf (slot-value obj name)
			       (vt-scale val factor))))))))))
    (typecase component
      (sequential (dolist (l (seq-layers component))
                    (scale-all-grads! l factor)))
      (residual (scale-all-grads! (residual-block component) factor))
      (transformer-block
       (dolist (sub (list (tb-mha component)
                          (tb-ffn1 component)
                          (tb-ffn2 component)
                          (tb-ln1 component)
                          (tb-ln2 component)))
         (when sub (scale-all-grads! sub factor))))
      (layer (scale-slots component))
      (t nil))))

(defun clipped-gradient-update! (model optimizer max-norm)
  "梯度裁剪后更新参数."
  (let* ((grad-norm (compute-grad-norm model))
         (clip-coeff
           (if (> grad-norm max-norm)
               (/ max-norm grad-norm)
               1.0d0)))
    (unless (= clip-coeff 1.0d0)
      (dolist (layer (collect-all-layers model))
        (scale-all-grads! layer clip-coeff)))
    (model-update! model optimizer)))

(defun tensor-ensure-2d (x)
  "确保张量为 2D: (batch, dim)."
  (if (= (length (vt-shape x)) 1)
      (vt-reshape x
                  (list 1 (first (vt-shape x))))
      x))

(defun tensor-unsqueeze (x &optional (dim 0))
  "在指定维度插入大小为 1 的维度."
  (let ((shape (vt-shape x)))
    (vt-reshape
     x
     (append (subseq shape 0 dim)
             (list 1)
             (subseq shape dim)))))

(defun tensor-one-hot (indices num-classes)
  "将整数索引转换为 one-hot 编码."
  (let* ((batch (reduce #'* (vt-shape indices)))
         (result-data
           (make-array
            (list batch num-classes)
            :element-type 'double-float
            :initial-element 0.0d0)))
    (dotimes (i batch)
      (let ((idx (coerce (vt-ref indices i)
                         'fixnum)))
        (setf (aref result-data i idx) 1.0d0)))
    (vt-reshape
     (vt-from-array result-data)
     (list batch num-classes))))

(defun tensor-masked-fill (x mask value)
  "将 mask 为真的位置填充为 value."
  (vt-map
   (lambda (xi mi)
     (if (> mi 0.0d0) value xi))
   x mask))

(defun tensor-where (condition x y)
  "三元选择."
  (vt-map
   (lambda (c xi yi)
     (if (> c 0.0d0) xi yi))
   condition x y))


(defun model->plist (model)
  "将模型序列化为 plist (用于保存)."
  `(:type ,(class-name (class-of model))
    :name ,(layer-name model)
    :layers ,(mapcar
              (lambda (l)
                `(:type ,(class-name (class-of l))
                  :name ,(layer-name l)
                  :params ,(mapcar
                            (lambda (p)
                              (cons (second p)
                                    (vt-to-list
                                     (third p))))
                            (params l))))
              (collect-all-layers model))))

(defun plist->model (plist)
  "从 plist 反序列化模型 (简化)."
  ;; 完整实现需要根据 :type 动态构造
  plist
  )

(defun save-model (model filepath)
  "保存模型到文件."
  (with-open-file
      (out filepath :direction :output
                    :if-exists :supersede)
    (print (model->plist model) out)))

(defun load-model (filepath)
  "从文件加载模型 (简化)."
  (with-open-file (in filepath)
    (plist->model (read in))))


(defclass neural-network-compat (sequential)
  ((lr :initarg :lr
       :initform 0.001d0
       :accessor nn-compat-lr)
   (grad-clip :initarg :grad-clip
	      :initform 1.0d0
	      :accessor nn-compat-grad-clip))
  (:documentation "兼容层."))

(defun find-slot-value (obj name)
  "在对象中查找名为 NAME 的权重 slot."
  (let ((class (class-of obj)))
    (dolist (slot (c2mop:class-slots class))
      (when (string-equal
             (symbol-name
              (c2mop:slot-definition-name slot))
             name)
        (return-from find-slot-value
          (if (slot-boundp
               obj
               (c2mop:slot-definition-name slot))
              (slot-value
               obj
               (c2mop:slot-definition-name slot))
              nil))))
    nil))

(defun slot-exists-p-by-name (obj name)
  "检查对象是否存在指定名称的 slot."
  (let ((class (class-of obj)))
    (dolist (slot (c2mop:class-slots class))
      (when (string-equal
             (symbol-name
              (c2mop:slot-definition-name slot))
             name)
        (return-from slot-exists-p-by-name t)))
    nil))

(defun set-slot-value-by-name (obj name value)
  "按名字设置 slot 值."
  (let ((class (class-of obj)))
    (dolist (slot (c2mop:class-slots class))
      (when (string-equal
             (symbol-name
              (c2mop:slot-definition-name slot))
             name)
        (setf (slot-value
               obj
               (c2mop:slot-definition-name slot))
              value)
        (return-from set-slot-value-by-name)))))

(defun copy-network (source)
  "深拷贝网络，保留所有可学习参数、激活类型和运行统计量。"
  (let ((new-model
          (etypecase source
            (sequential
             (make-sequential
              :name (concatenate 'string (layer-name source) "-copy")))
            (layer
             (let ((class (class-of source)))
               (cond
                 ;; Dense 层
                 ((eq class (find-class 'dense))
                  (make-dense (dense-out-dim source)
                              :in-dim (dense-in-dim source)
                              :activation (dense-activation source)
                              :use-bias (dense-use-bias-p source)
                              :leaky-alpha (dense-leaky-alpha source)
                              :name (concatenate 'string (layer-name source) "-copy")
                              :trainable (layer-trainable-p source)))
                 ;; Conv2d 层
                 ((eq class (find-class 'conv2d))
                  (make-conv2d (conv-out-channels source)
                               (conv-kernel-size source)
                               :in-channels (conv-in-channels source)
                               :stride (conv-stride source)
                               :padding (conv-padding source)
                               :use-bias (conv-use-bias-p source)
                               :name (concatenate 'string (layer-name source) "-copy")
                               :trainable (layer-trainable-p source)))
                 ;; BatchNorm 层
                 ((eq class (find-class 'batch-norm))
                  (make-batch-norm (bn-num-features source)
                                   :eps (bn-eps source)
                                   :momentum (bn-momentum source)
                                   :affine (bn-affine-p source)
                                   :name (concatenate 'string (layer-name source) "-copy")
                                   :trainable (layer-trainable-p source)))
                 ;; LayerNorm 层
                 ((eq class (find-class 'layer-norm))
                  (make-layer-norm (ln-normalized-shape source)
                                   :eps (ln-eps source)
                                   :affine (ln-affine-p source)
                                   :name (concatenate 'string (layer-name source) "-copy")
                                   :trainable (layer-trainable-p source)))
                 ;; 其他层（激活层、dropout、池化层等）直接复制，保留原有构造函数
                 (t
                  (make-instance class
                                 :name (concatenate 'string (layer-name source) "-copy")
                                 :trainable (layer-trainable-p source)))))))))
    ;; 递归复制子层（Sequential）
    (when (typep source 'sequential)
      (dolist (orig (seq-layers source))
        (seq-add! new-model (copy-network orig))))
    ;; 非容器层：复制权重、偏置、运行统计量
    (unless (or (typep source 'sequential)
                (typep source 'residual)
                (typep source 'transformer-block))
      (dolist (pname '("weights" "bias" "weight"
                       "gamma" "beta"
                       "w-q" "w-k" "w-v" "w-o"
                       "b-q" "b-k" "b-v" "b-o"
                       "weight-ih" "weight-hh"
                       "bias-ih" "bias-hh"
                       "wih" "whh" "bih"
                       ;; embedding 的 weight
                       "weight"))
        (when (slot-exists-p-by-name source pname)
          (let ((src-val (find-slot-value source pname)))
            (when (and src-val (vt-p src-val))
              (set-slot-value-by-name new-model pname (vt-copy src-val)))))))
    ;; BatchNorm 额外复制 running-mean / running-var
    (when (typep source 'batch-norm)
      (dolist (stat '("running-mean" "running-var"))
        (when (slot-exists-p-by-name source stat)
          (let ((src-val (find-slot-value source stat)))
            (when (and src-val (vt-p src-val))
              (set-slot-value-by-name new-model stat (vt-copy src-val)))))))
    new-model))

(defun tensor-top-k (x k &key (axis -1))
  "返回 top-k 值和索引."
  (let* ((shape (vt-shape x))
         (rank (length shape))
         (actual-axis
           (if (< axis 0) (+ rank axis) axis))
         (axis-size (nth actual-axis shape))
         (effective-k (min k axis-size))
         (out-shape
           (let ((s (copy-list shape)))
             (setf (nth actual-axis s)
                   effective-k)
             s))
         (tail-size
           (if (= actual-axis (1- rank))
               1
               (reduce #'*
                       (subseq shape
                               (1+ actual-axis)))))
         (num-slices
           (if (= actual-axis 0)
               1
               (reduce #'*
                       (subseq shape 0 actual-axis))))
         (slice-size (* axis-size tail-size))
         (out-slice-size (* effective-k tail-size))
         (flat-x (vt-flatten x))
         (data (vt-data flat-x))
         (total-out (reduce #'* out-shape))
         (result-vals
           (make-array total-out
                       :element-type 'double-float))
         (result-idxs
           (make-array total-out
                       :element-type 'fixnum)))
    (dotimes (s num-slices)
      (let ((slice-start (* s slice-size))
            (dst-start (* s out-slice-size)))
        (let ((block-reps '()))
          (dotimes (i axis-size)
            (let ((offset
                    (+ slice-start
                       (* i tail-size))))
              (push (cons (aref data offset) i)
                    block-reps)))
          (setf block-reps
                (sort block-reps #'> :key #'car))
          (dotimes (i effective-k)
            (let* ((src-axis-idx
                     (cdr (nth i block-reps)))
                   (src-offset
                     (+ slice-start
                        (* src-axis-idx tail-size)))
                   (dst-offset
                     (+ dst-start (* i tail-size))))
              (dotimes (j tail-size)
                (setf (aref result-vals
                            (+ dst-offset j))
                      (aref data
                            (+ src-offset j)))
                (setf (aref result-idxs
                            (+ dst-offset j))
                      src-axis-idx)))))))
    (values
     (vt-reshape
      (vt-from-sequence
       (coerce result-vals 'list))
      out-shape)
     (vt-reshape
      (vt-from-sequence
       (coerce result-idxs 'list))
      out-shape))))

;; ---- zero-grad! 重写：基于 grad-slots 泛型函数 + zero-grad-children 递归，
;; ---- 不再依赖 slot 名字白名单启发式，多层同类网络（如多个 dense/conv/lstm）
;; ---- 每层都被自己的 grad-slots 方法正确清零，不会遗漏也不会误清。

(defgeneric zero-grad-children (component)
  (:documentation "返回需要递归清零的直接子组件列表。默认返回 nil。")
  (:method ((c t)) '())
  (:method ((m sequential)) (coerce (seq-layers m) 'list))
  (:method ((l residual))  (list (residual-block l)))
  (:method ((l transformer-block))
    (list (tb-mha l) (tb-ffn1 l) (tb-ffn2 l)
          (tb-ln1 l) (tb-ln2 l) (tb-drop1 l) (tb-drop2 l)))
  ;; ffn1/ffn2 are dense layers directly (not sequential), no further recursion needed
  (:method ((l neural-network-compat)) (coerce (seq-layers l) 'list)))

(defgeneric zero-grad! (component)
  (:documentation "将组件及其所有子层中的梯度张量清零（置为 NIL）。
实现原则：
  1. 每个叶子层通过 grad-slots 方法返回自己的梯度 accessor 列表（每个
     accessor 是一个 reader 函数，其 setf 可写），通用方法统一调用 setf
     将梯度 vt 置为 nil。
  2. 容器层通过 zero-grad-children 返回需要递归的子组件列表。
  3. 不做任何 slot 名字匹配/字符串白名单扫描，用户自定义层只需要
     实现 grad-slots 方法即可被正确清零。")
  (:method ((component null)) nil)
  (:method ((component t))
    (when (and (typep component 'layer) (layer-trainable-p component))
      (dolist (slot-name (grad-slots component))
        (when (and (slot-boundp component slot-name)
                   (vt-p (slot-value component slot-name)))
          (setf (slot-value component slot-name) nil))))
    (dolist (child (zero-grad-children component))
      (when child (zero-grad! child)))))



(defun clear-all-gradients! (model)
  "zero-grad! 的别名."
  (zero-grad! model))

(defgeneric clear-forward-cache! (component)
  (:documentation "清空前向传播产生的中间缓存, 仅保留参数和梯度.")
  (:method ((c t)) nil)
  (:method ((component layer))
    (let ((class (class-of component)))
      (dolist (slot (c2mop:class-slots class))
        (let ((name (c2mop:slot-definition-name slot)))
          (when (and (slot-boundp component name)
                     (let ((sname (symbol-name name)))
		       ;; 仅仅清理明确带有 cache 字样的 slot
                       ;; 绝对不能清理 BATCH-SIZE 等反向传播依赖的整型状态！
                       (search "cache" sname)))
            (setf (slot-value component name) nil))))))
  (:method ((component sequential))
    (dolist (layer (seq-layers component))
      (clear-forward-cache! layer)))
  (:method ((component residual))
    (clear-forward-cache! (residual-block component)))
  (:method ((component transformer-block))
    (clear-forward-cache! (tb-mha component))
    (clear-forward-cache! (tb-ffn1 component))
    (clear-forward-cache! (tb-ffn2 component))
    (clear-forward-cache! (tb-ln1 component))
    (clear-forward-cache! (tb-ln2 component))))

(defun build-model (model &optional dummy-input)
  "强制初始化模型中所有延迟参数.

   对于 LSTM/GRU/MHA 等构造时已知维度的层, 直接初始化.
   对于 Dense/Conv 等依赖输入维度的层, 必须提供 DUMMY-INPUT
   执行一次虚拟前向传播来推断维度.

   执行完毕后会自动清理产生的中间缓存, 模型处于干净可用状态.

   示例:
     (build-model my-lstm) ; 纯 RNN 不需要 dummy-input
     (build-model my-cnn (vt-zeros (list 1 3 28 28))) ; CNN 需要"
  ;; 1. 显式触发那些不需要输入就能构建的层
  (dolist (l (collect-all-layers model))
    (typecase l
      (lstm (ensure-lstm-params l))
      (gru (ensure-gru-params l))
      (rnn-cell (ensure-rnn-cell-params l))
      (multi-head-attention (ensure-mha-params l))
      ;; Embedding 只需知道词表大小即可初始化
      (embedding
       (unless (emb-weight l)
         ;; 传入一个假的索引 0 触发初始化
         (forward l (vt-zeros (list 1)))))
      (t nil)))

  ;; 2. 如果提供了 dummy-input，跑一次前向传播打通剩余层
  (when dummy-input
    (forward model dummy-input))

  ;; 3. 核心步骤：清理 dry-run 留下的所有垃圾缓存
  (clear-forward-cache! model)

  model)
