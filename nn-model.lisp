(in-package #:nn)

(defclass sequential (layer)
  ((layers :initform '() :accessor seq-layers
           :type list)
   (layer-names :initform '()
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
    (dolist (l layers)
      (push l (seq-layers model))
      (push (or (layer-name l) "")
            (seq-layer-names model))))
  (setf (seq-layers model)
        (nreverse (seq-layers model)))
  (setf (seq-layer-names model)
        (nreverse (seq-layer-names model)))
  model)

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
    (labels ((count-in (obj)
               (cond
                 ((typep obj 'sequential)
                  (dolist (l (seq-layers obj))
                    (count-in l)))
                 (t
                  (when (or (not trainable-only)
                            (layer-trainable-p obj))
                    (dolist (p (params obj))
                      ;; 协议修复: 取第二个元素
                      (let ((tensor (second p)))
                        (when tensor
                          (incf total
                                (reduce #'*
                                       (vt-shape tensor)))))))))))
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
                (vt-mean (vt-square tensor))))))
    (sqrt sq-sum)))


(defun scale-all-grads! (component factor)
  "递归将所有梯度张量乘以 factor (原地修改层内部状态)."
  (let ((class (class-of component)))
    (dolist (slot (c2mop:class-slots class))
      (let ((name (c2mop:slot-definition-name slot)))
        (when (and (slot-boundp component name)
                   (let ((sname (symbol-name name)))
                     (or (and (> (length sname) 0)
                              (char= (char sname 0) #\d))
                         (search "grad" sname))))
          (let ((val (slot-value component name)))
            (when (vt-p val)
              (setf (slot-value component name)
                    (vt-scale val factor)))))))))

(defun clipped-gradient-update! (model optimizer max-norm)
  "梯度裁剪后更新参数."
  (let* ((grad-norm (compute-grad-norm model))
         (clip-coeff
           (if (> grad-norm max-norm)
               (/ max-norm grad-norm)
               1.0d0)))
    (when (< clip-coeff 1.0d0)
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
      (let ((idx (coerce (vt-ref indices (list i))
                         'fixnum)))
        (setf (aref result-data i idx) 1.0d0)))
    (vt-reshape
      (vt-from-2d-array result-data)
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
                              ;; 协议修复: 取第二个元素
                              (cons (first p)
                                    (vt-data->list
                                      (second p))))
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
  ((lr :initarg :lr :initform 0.001d0
    :accessor nn-compat-lr)
   (grad-clip :initarg :grad-clip :initform 1.0d0
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
  "深拷贝网络."
  (let ((new-model
          (etypecase source
            (sequential
              (make-sequential
                :name (concatenate
                        'string
                        (layer-name source) "-copy")))
            (layer
              (make-instance
                (class-of source)
                :name (concatenate
                        'string
                        (layer-name source) "-copy")
                :trainable
                  (layer-trainable-p source))))))
    (when (typep source 'sequential)
      (dolist (orig-layer (seq-layers source))
        (let ((new-layer (copy-network orig-layer)))
          (seq-add! new-model new-layer))))
    (unless (or (typep source 'sequential)
                (typep source 'residual)
                (typep source 'transformer-block))
      (dolist (pname '("weights" "bias" "weight"
                       "gamma" "beta"
                       "w-q" "w-k" "w-v" "w-o"
                       "b-q" "b-k" "b-v" "b-o"
                       "weight-ih" "weight-hh"
                       "bias-ih" "bias-hh"
                       "wih" "whh" "bih"))
        (when (slot-exists-p-by-name source pname)
          (let ((src-val
                  (find-slot-value source pname)))
            (when (and src-val (vt-p src-val))
              (set-slot-value-by-name
                new-model pname
                (vt-copy src-val)))))))
    (when (typep source 'batch-norm)
      (dolist (stat-name '("running-mean" "running-var"))
        (when (slot-exists-p-by-name
                source stat-name)
          (let ((src-val
                  (find-slot-value
                    source stat-name)))
            (when (and src-val (vt-p src-val))
              (set-slot-value-by-name
                new-model stat-name
                (vt-copy src-val)))))))
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

(defgeneric zero-grad! (component)
  (:documentation "递归清零梯度.")
  (:method ((component null)) nil)
  (:method ((component layer))
    (let ((class (class-of component)))
      (dolist (slot (c2mop:class-slots class))
        (let ((name (c2mop:slot-definition-name slot)))
          (when (and (slot-boundp component name)
                     (let ((sname (symbol-name name)))
                       (or (and (> (length sname) 0)
                                (char= (char sname 0)
                                       #\d))
                           (search "grad" sname))))
            (let ((val (slot-value component name)))
              (when (vt-p val)
                (setf (slot-value component name)
                      nil))))))))
  (:method ((component sequential))
    (dolist (layer (seq-layers component))
      (zero-grad! layer)))
  (:method ((component residual))
    (zero-grad! (residual-block component)))
  (:method ((component transformer-block))
    (zero-grad! (tb-mha component))
    (zero-grad! (tb-ffn1 component))
    (zero-grad! (tb-ffn2 component))
    (zero-grad! (tb-ln1 component))
    (zero-grad! (tb-ln2 component))))

(defun clear-all-gradients! (model)
  "zero-grad! 的别名."
  (zero-grad! model))
