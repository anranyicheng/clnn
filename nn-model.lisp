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
  "递归将所有梯度张量乘以 factor（原地修改层内部状态）。
基于 grad-slots 泛型函数定位梯度 slot，避免按名字字符串/白名单的脆弱启发式。"
  (labels ((scale-slots (obj)
             (dolist (slot-name (grad-slots obj))
               (when (and (slot-boundp obj slot-name)
                          (vt-p (slot-value obj slot-name)))
                 (setf (slot-value obj slot-name)
                       (vt-scale (slot-value obj slot-name) factor))))))
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


(defun %serialize-vt (tensor)
  "将张量序列化为 plist（含 shape/dtype/values），可无损还原。"
  (when tensor
    (list :shape (vt-shape tensor)
          :dtype (vt-dtype tensor)
          :values (vt-to-list tensor))))

(defun %restore-vt (spec)
  "从 %serialize-vt 产生的 plist 还原张量。"
  (destructuring-bind (&key shape dtype values) spec
    (vt-reshape (vt-from-sequence values :dtype dtype) shape)))

(defun layer->plist (l)
  "将单个层序列化为 plist（目前支持 dense / activation-layer）。"
  (let ((cls (class-name (class-of l))))
    (ecase cls
      (dense
       (list :type 'dense
             :name (layer-name l)
             :in-dim (dense-in-dim l)
             :out-dim (dense-out-dim l)
             :activation (dense-activation l)
             :use-bias (dense-use-bias-p l)
             :weights (%serialize-vt (dense-weights l))
             :bias (%serialize-vt (dense-bias l))))
      (activation-layer
       (list :type 'activation-layer
             :name (layer-name l)
             :kind (activation-kind l)
             :leaky-alpha (act-leaky-alpha l)))
      (otherwise (error "model->plist: 不支持的层类型 ~a" cls)))))

(defun model->plist (model)
  "将模型序列化为 plist（用于保存）。目前支持 sequential 容器与 dense/activation-layer 层。"
  `(:type ,(class-name (class-of model))
    :name ,(layer-name model)
    :layers ,(mapcar #'layer->plist (collect-all-layers model))))

(defun plist->model (plist)
  "从 plist 反序列化模型（目前支持 sequential 容器与 dense/activation-layer 层）。"
  (let ((model (make-sequential :name (getf plist :name))))
    (dolist (l-plist (getf plist :layers))
      (seq-add! model
                (let ((lt (getf l-plist :type)))
                  (ecase lt
                    (dense
                     (let ((d (make-dense (getf l-plist :out-dim)
                                          :in-dim (getf l-plist :in-dim)
                                          :activation (getf l-plist :activation)
                                          :use-bias (getf l-plist :use-bias)
                                          :name (getf l-plist :name))))
                       (when (getf l-plist :weights)
                         (setf (dense-weights d)
                               (%restore-vt (getf l-plist :weights))))
                       (when (getf l-plist :bias)
                         (setf (dense-bias d)
                               (%restore-vt (getf l-plist :bias))))
                       d))
                    (activation-layer
                     (make-activation-layer (getf l-plist :kind)
                                            :leaky-alpha (getf l-plist :leaky-alpha)
                                            :name (getf l-plist :name)))
                    (otherwise (error "plist->model: 不支持的层类型 ~a" lt))))))
    model))

(defun save-model (model filepath)
  "保存模型到文件."
  (with-open-file
      (out filepath :direction :output
                    :if-exists :supersede)
    (print (model->plist model) out)))

(defun load-model (filepath)
  "从文件加载模型."
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

;;; ------------------------------------------------------------------
;;; copy-network：基于 CLOS 泛型分发的深拷贝
;;; ------------------------------------------------------------------
;;; 每种层通过自己的 copy-network 方法决定复制哪些字段；
;;; 容器层递归子层；未识别的层落到 layer 兜底分支。
;;;
;;; 使用 CLOS 分发（不是 typecase），新增层只需加一个方法，
;;; 不需要修改 copy-network 本身。

(defun copy-layer-name (source)
  "给源层生成副本名（原名 + \"-copy\"）。"
  (concatenate 'string (layer-name source) "-copy"))

(defgeneric copy-network (source)
  (:documentation "深拷贝网络/层。

  返回一个与 SOURCE 结构、配置、可学习参数、运行统计量一致，
  但底层张量完全独立的新对象。

  每种层通过自己的 copy-network 方法决定复制哪些字段；
  用户自定义层若不实现自己的方法，将落到 layer 兜底分支
  （只保留 name/trainable，不复制任何张量）。

  容器层（sequential / residual / transformer-block）递归子层。")
  ;; ---- 兜底：未知层只复制 name/trainable ----
  (:method ((source layer))
    (make-instance (class-of source)
                   :name (copy-layer-name source)
                   :trainable (layer-trainable-p source)))
  ;; ---- 非层对象：直接返回自身 ----
  (:method ((source t)) source))

;;; ---------- 容器层 ----------

(defmethod copy-network ((source sequential))
  (let ((copy (make-sequential :name (copy-layer-name source))))
    (dolist (layer (seq-layers source))
      (seq-add! copy (copy-network layer)))
    copy))

(defmethod copy-network ((source residual))
  (make-residual (copy-network (residual-block source))
                 :name (copy-layer-name source)))

(defmethod copy-network ((source transformer-block))
  (let ((copy (make-transformer-block
               (tb-embed-dim source)
               (tb-num-heads source)
               :ffn-dim (tb-ffn-dim source)
               :dropout-rate (tb-dropout-rate source)
               :eps (tb-eps source)
               :name (copy-layer-name source)
               :trainable (layer-trainable-p source))))
    ;; 用深拷贝覆盖构造时自动创建的子层
    (setf (tb-mha  copy) (copy-network (tb-mha  source))
          (tb-ffn1 copy) (copy-network (tb-ffn1 source))
          (tb-ffn2 copy) (copy-network (tb-ffn2 source))
          (tb-ln1  copy) (copy-network (tb-ln1  source))
          (tb-ln2  copy) (copy-network (tb-ln2  source)))
    ;; drop1/drop2 无状态，构造时已按同一 dropout-rate 创建，无需替换
    copy))

;;; ---------- 全连接 / 卷积 / 池化 ----------

(defmethod copy-network ((source dense))
  (let ((copy (make-dense (dense-out-dim source)
                          :in-dim (dense-in-dim source)
                          :activation (dense-activation source)
                          :use-bias (dense-use-bias-p source)
                          :leaky-alpha (dense-leaky-alpha source)
                          :name (copy-layer-name source)
                          :trainable (layer-trainable-p source))))
    (when (dense-weights source)
      (setf (dense-weights copy) (vt-copy (dense-weights source))))
    (when (dense-bias source)
      (setf (dense-bias copy) (vt-copy (dense-bias source))))
    copy))

(defmethod copy-network ((source conv2d))
  (let ((copy (make-conv2d (conv-out-channels source)
                           (conv-kernel-size source)
                           :in-channels (conv-in-channels source)
                           :stride (conv-stride source)
                           :padding (conv-padding source)
                           :use-bias (conv-use-bias-p source)
                           :weight-init (conv-weight-init source)
                           :name (copy-layer-name source)
                           :trainable (layer-trainable-p source))))
    (when (conv-weights source)
      (setf (conv-weights copy) (vt-copy (conv-weights source))))
    (when (conv-bias source)
      (setf (conv-bias copy) (vt-copy (conv-bias source))))
    copy))

(defmethod copy-network ((source max-pool2d))
  (make-max-pool2d (pool-kernel-size source)
                   :stride (pool-stride source)
                   :padding (pool-padding source)
                   :name (copy-layer-name source)
                   :trainable (layer-trainable-p source)))

(defmethod copy-network ((source avg-pool2d))
  (make-avg-pool2d (apool-kernel-size source)
                   :stride (apool-stride source)
                   :padding (apool-padding source)
                   :name (copy-layer-name source)
                   :trainable (layer-trainable-p source)))

(defmethod copy-network ((source global-avg-pool2d))
  (make-global-avg-pool2d :name (copy-layer-name source)
                          :trainable (layer-trainable-p source)))

;;; ---------- 归一化 / Dropout ----------

(defmethod copy-network ((source batch-norm))
  (let ((copy (make-batch-norm (bn-num-features source)
                               :eps (bn-eps source)
                               :momentum (bn-momentum source)
                               :affine (bn-affine-p source)
                               :name (copy-layer-name source)
                               :trainable (layer-trainable-p source))))
    ;; 可学习参数
    (when (bn-gamma source)
      (setf (bn-gamma copy) (vt-copy (bn-gamma source))))
    (when (bn-beta source)
      (setf (bn-beta copy) (vt-copy (bn-beta source))))
    ;; 运行时统计量
    (when (bn-running-mean source)
      (setf (bn-running-mean copy) (vt-copy (bn-running-mean source))))
    (when (bn-running-var source)
      (setf (bn-running-var copy) (vt-copy (bn-running-var source))))
    copy))

(defmethod copy-network ((source layer-norm))
  (let ((copy (make-layer-norm (ln-normalized-shape source)
                               :eps (ln-eps source)
                               :affine (ln-affine-p source)
                               :name (copy-layer-name source)
                               :trainable (layer-trainable-p source))))
    (when (ln-gamma source)
      (setf (ln-gamma copy) (vt-copy (ln-gamma source))))
    (when (ln-beta source)
      (setf (ln-beta copy) (vt-copy (ln-beta source))))
    copy))

(defmethod copy-network ((source dropout))
  (make-dropout (dropout-p source)
                :name (copy-layer-name source)
                :inverted (dropout-inverted-p source)))

;;; ---------- 激活 / Flatten ----------

(defmethod copy-network ((source activation-layer))
  (make-activation-layer (activation-kind source)
                         :leaky-alpha (act-leaky-alpha source)
                         :name (copy-layer-name source)))

(defmethod copy-network ((source flatten))
  (make-flatten :start-dim (flatten-start-dim source)
                :name (copy-layer-name source)))

;;; ---------- Embedding ----------

(defmethod copy-network ((source embedding))
  (let ((copy (make-embedding (emb-num-embeddings source)
                              (emb-embedding-dim source)
                              :max-norm (emb-max-norm source)
                              :scale-grad-by-freq
                              (emb-scale-grad-by-freq source)
                              :name (copy-layer-name source)
                              :trainable (layer-trainable-p source))))
    (when (emb-weight source)
      (setf (emb-weight copy) (vt-copy (emb-weight source))))
    copy))

;;; ---------- 循环层 ----------

(defmethod copy-network ((source rnn-cell))
  (let ((copy (make-rnn-cell (rnn-input-size source)
                             (rnn-hidden-size source)
                             :activation (rnn-activation source)
                             :name (copy-layer-name source)
                             :trainable (layer-trainable-p source))))
    (when (rnn-wih source) (setf (rnn-wih copy) (vt-copy (rnn-wih source))))
    (when (rnn-whh source) (setf (rnn-whh copy) (vt-copy (rnn-whh source))))
    (when (rnn-bih source) (setf (rnn-bih copy) (vt-copy (rnn-bih source))))
    copy))

(defmethod copy-network ((source lstm))
  (let ((copy (make-lstm (lstm-input-size source)
                         (lstm-hidden-size source)
                         :name (copy-layer-name source)
                         :trainable (layer-trainable-p source)
                         :h-0 (and (lstm-h-0 source)
                                   (vt-copy (lstm-h-0 source)))
                         :c-0 (and (lstm-c-0 source)
                                   (vt-copy (lstm-c-0 source))))))
    (when (lstm-weight-ih source)
      (setf (lstm-weight-ih copy) (vt-copy (lstm-weight-ih source))))
    (when (lstm-weight-hh source)
      (setf (lstm-weight-hh copy) (vt-copy (lstm-weight-hh source))))
    (when (lstm-bias-ih source)
      (setf (lstm-bias-ih copy) (vt-copy (lstm-bias-ih source))))
    (when (lstm-bias-hh source)
      (setf (lstm-bias-hh copy) (vt-copy (lstm-bias-hh source))))
    copy))

(defmethod copy-network ((source gru))
  (let ((copy (make-gru (gru-input-size source)
                        (gru-hidden-size source)
                        :name (copy-layer-name source)
                        :trainable (layer-trainable-p source))))
    (when (gru-weight-ih source)
      (setf (gru-weight-ih copy) (vt-copy (gru-weight-ih source))))
    (when (gru-weight-hh source)
      (setf (gru-weight-hh copy) (vt-copy (gru-weight-hh source))))
    (when (gru-bias-ih source)
      (setf (gru-bias-ih copy) (vt-copy (gru-bias-ih source))))
    (when (gru-bias-hh source)
      (setf (gru-bias-hh copy) (vt-copy (gru-bias-hh source))))
    copy))

;;; ---------- 注意力 ----------

(defmethod copy-network ((source multi-head-attention))
  (let ((copy (make-multi-head-attention
               (mha-embed-dim source)
               (mha-num-heads source)
               :use-bias (mha-use-bias-p source)
               :dropout-rate (mha-dropout-rate source)
               :name (copy-layer-name source)
               :trainable (layer-trainable-p source))))
    (when (mha-wq source) (setf (mha-wq copy) (vt-copy (mha-wq source))))
    (when (mha-wk source) (setf (mha-wk copy) (vt-copy (mha-wk source))))
    (when (mha-wv source) (setf (mha-wv copy) (vt-copy (mha-wv source))))
    (when (mha-wo source) (setf (mha-wo copy) (vt-copy (mha-wo source))))
    (when (mha-bq source) (setf (mha-bq copy) (vt-copy (mha-bq source))))
    (when (mha-bk source) (setf (mha-bk copy) (vt-copy (mha-bk source))))
    (when (mha-bv source) (setf (mha-bv copy) (vt-copy (mha-bv source))))
    (when (mha-bo source) (setf (mha-bo copy) (vt-copy (mha-bo source))))
    copy))

;;; ---------- 兼容层 ----------

(defmethod copy-network ((source neural-network-compat))
  (let ((copy (make-instance 'neural-network-compat
                             :name (copy-layer-name source)
                             :trainable (layer-trainable-p source)
                             :lr (nn-compat-lr source)
                             :grad-clip (nn-compat-grad-clip source))))
    (dolist (layer (seq-layers source))
      (seq-add! copy (copy-network layer)))
    copy))

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
         (data-off (vt-offset flat-x))
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
              (push (cons (aref data (+ data-off offset)) i)
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
                            (+ data-off src-offset j)))
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
  (:documentation "清空前向传播产生的中间缓存。
只清理 cache-slots 显式列出的 slot，仅保留参数、梯度和反向传播依赖的
整型/配置状态（batch-size / norm-size / state 等）。
实现与 zero-grad! 风格一致：
  - 叶子层通过 cache-slots 返回自己的缓存 slot 名列表；
  - 容器层（sequential / residual / transformer-block）通过本方法递归子层。
用户自定义新层只需实现 cache-slots 方法即可被正确清理。")
  (:method ((component null)) nil)
  (:method ((component t)) nil)
  (:method ((component layer))
    (dolist (slot-name (cache-slots component))
      (when (slot-boundp component slot-name)
        (setf (slot-value component slot-name) nil))))
  (:method ((component sequential))
    (dolist (layer (seq-layers component))
      (clear-forward-cache! layer)))
  (:method ((component residual))
    (when (residual-block component)
      (clear-forward-cache! (residual-block component))))
  (:method ((component transformer-block))
    (dolist (sub (list (tb-mha component)
                       (tb-ffn1 component)
                       (tb-ffn2 component)
                       (tb-ln1 component)
                       (tb-ln2 component)
                       (tb-drop1 component)
                       (tb-drop2 component)))
      (when sub (clear-forward-cache! sub)))))

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
