(in-package #:nn)

;; ===================================================================
;; 1. 可微张量定义
;; ===================================================================
(defclass diff-tensor ()
  ((data :accessor dt-data :initarg :data :type vt)
   (grad :accessor dt-grad :initform nil :type (or null vt))
   (grad-fn :accessor dt-grad-fn :initform nil :type (or null function))
   (children :accessor dt-children :initform '() :type list))
  (:documentation "支持自动微分的张量包装器."))

(defun requires-grad-p (x) (typep x 'diff-tensor))

(defun ensure-diff (x &optional (requires-grad nil))
  "将普通 vt 包装为 diff-tensor，或原样返回."
  (declare (ignorable requires-grad))
  (if (typep x 'diff-tensor) x (make-instance 'diff-tensor :data x)))

;; ===================================================================
;; 2. 独立的 Autograd 算子接口 (以 ag- 为前缀，避免污染底层 vt- 接口)
;; ===================================================================

;; --- 加法 ---
(defgeneric ag-+ (a b) (:documentation "Autograd 加法."))

(defmethod ag-+ ((a diff-tensor) (b diff-tensor))
  (let* ((out-data (vt-+ (dt-data a) (dt-data b)))
         (out (make-instance
	       'diff-tensor
	       :data out-data
	       :children (list a b)
               :grad-fn (lambda (g)
                          (when (requires-grad-p a)
                            (setf (dt-grad a)
				  (if (dt-grad a)
				      (vt-+ (dt-grad a) g)
				      g)))
                          (when (requires-grad-p b)
                            (setf (dt-grad b)
				  (if (dt-grad b)
				      (vt-+ (dt-grad b) g)
				      g)))))))
    out))

(defmethod ag-+ ((a vt) (b vt)) (ensure-diff (vt-+ a b)))
(defmethod ag-+ ((a diff-tensor) (b vt)) (ag-+ a (ensure-diff b)))
(defmethod ag-+ ((a vt) (b diff-tensor)) (ag-+ (ensure-diff a) b))

;; --- 矩阵乘法 ---
(defgeneric ag-matmul (a b) (:documentation "Autograd 矩阵乘法."))

(defmethod ag-matmul ((a diff-tensor) (b diff-tensor))
  (let* ((out-data (vt-matmul (dt-data a) (dt-data b)))
         (out (make-instance
	       'diff-tensor
	       :data out-data
	       :children (list a b)
               :grad-fn
	       (lambda (g)
                 (when (requires-grad-p a)
                   (let ((grad-a (vt-matmul g (vt-transpose (dt-data b)))))
                     (setf (dt-grad a)
			   (if (dt-grad a)
			       (vt-+ (dt-grad a) grad-a)
			       grad-a))))
                 (when (requires-grad-p b)
                   (let ((grad-b (vt-matmul (vt-transpose (dt-data a)) g)))
                     (setf (dt-grad b)
			   (if (dt-grad b)
			       (vt-+ (dt-grad b) grad-b)
			       grad-b))))))))
    out))

(defmethod ag-matmul ((a vt) (b vt)) (ensure-diff (vt-matmul a b)))
(defmethod ag-matmul ((a diff-tensor) (b vt)) (ag-matmul a (ensure-diff b)))
(defmethod ag-matmul ((a vt) (b diff-tensor)) (ag-matmul (ensure-diff a) b))

;; --- 减法 ---
(defgeneric ag-- (a b) (:documentation "Autograd 减法."))

(defmethod ag-- ((a diff-tensor) (b diff-tensor))
  (let* ((out-data (vt-- (dt-data a) (dt-data b)))
         (out (make-instance
	       'diff-tensor :data out-data :children (list a b)
               :grad-fn (lambda (g)
                          (when (requires-grad-p a)
                            (setf (dt-grad a)
				  (if (dt-grad a)
				      (vt-+ (dt-grad a) g)
				      g)))
                          (when (requires-grad-p b)
                            (let ((grad-b (vt-scale g -1.0d0)))
                              (setf (dt-grad b)
				    (if (dt-grad b)
					(vt-+ (dt-grad b) grad-b)
					grad-b))))))))
    out))

(defmethod ag-- ((a vt) (b vt)) (ensure-diff (vt-- a b)))
(defmethod ag-- ((a diff-tensor) (b vt)) (ag-- a (ensure-diff b)))
(defmethod ag-- ((a vt) (b diff-tensor)) (ag-- (ensure-diff a) b))

;; --- 乘法 ---
(defgeneric ag-* (a b) (:documentation "Autograd 逐元素乘法."))

(defmethod ag-* ((a diff-tensor) (b diff-tensor))
  (let* ((out-data (vt-* (dt-data a) (dt-data b)))
         (out (make-instance
	       'diff-tensor
	       :data out-data
	       :children (list a b)
               :grad-fn (lambda (g)
                          (when (requires-grad-p a)
                            (let ((grad-a (vt-* g (dt-data b))))
                              (setf (dt-grad a)
				    (if (dt-grad a)
					(vt-+ (dt-grad a) grad-a)
					grad-a))))
                          (when (requires-grad-p b)
                            (let ((grad-b (vt-* g (dt-data a))))
                              (setf (dt-grad b)
				    (if (dt-grad b)
					(vt-+ (dt-grad b) grad-b)
					grad-b))))))))
    out))

(defmethod ag-* ((a vt) (b vt)) (ensure-diff (vt-* a b)))
(defmethod ag-* ((a diff-tensor) (b vt)) (ag-* a (ensure-diff b)))
(defmethod ag-* ((a vt) (b diff-tensor)) (ag-* (ensure-diff a) b))

;; ===================================================================
;; 3. 反向传播引擎
;; ===================================================================
(defun build-topo (node &optional (visited nil) (topo nil))
  (if (member node visited)
      (values visited topo)
      (progn (push node visited)
             (dolist (child (dt-children node))
               (multiple-value-bind (v t-list)
		   (build-topo child visited topo)
                 (setf visited v)
                 (setf topo t-list)))
             (push node topo)
             (values visited topo))))

(defmethod diff-backward ((node diff-tensor) &optional (grad nil))
  (let ((init-grad (or grad (vt-ones (vt-shape (dt-data node)))))) ;; vt-ones 生成张量，符合规范
    (setf (dt-grad node)
	  (if (dt-grad node)
	      (vt-+ (dt-grad node) init-grad)
	      init-grad))
    (multiple-value-bind (visited topo) (build-topo node)
      (declare (ignore visited))
      (dolist (n (reverse topo))
        (when (dt-grad-fn n)
          (funcall (dt-grad-fn n) (dt-grad n)))))))

;; ===================================================================
;; 4. 架构桥接
;; ===================================================================
(defclass autograd-layer (layer)
  ((inputs :initarg :inputs :accessor ag-inputs)
   (output :initarg :output :accessor ag-output)
   (forward-fn :initarg :forward-fn :reader ag-forward-fn))
  (:documentation "将 Autograd 表达式包装成符合 nn 协议的层."))

(defun make-autograd-layer (forward-fn &key (name "autograd"))
  (make-instance 'autograd-layer
		 :forward-fn forward-fn :name name :trainable t))

(defmethod forward ((l autograd-layer) input-list)
  (let* ((diff-inputs (mapcar #'ensure-diff input-list))
         (diff-output (funcall (ag-forward-fn l) diff-inputs)))
    (setf (ag-inputs l) diff-inputs)
    (setf (ag-output l) diff-output)
    (dt-data diff-output)))

(defmethod backward ((l autograd-layer) grad-output)
  (diff-backward (ag-output l) grad-output)
  (mapcar #'dt-grad (ag-inputs l)))

(defclass diff-param (diff-tensor)
  ((owner :initarg :owner :reader dp-owner)
   (name :initarg :name :reader dp-name)
   (setter :initarg :setter :reader dp-setter))
  (:documentation "带有所有权信息的可微参数."))

(defun make-diff-param (data owner name setter)
  (make-instance 'diff-param :data data :owner owner
			     :name name :setter setter))

(defmethod (setf dt-data) :after (new-val (p diff-param))
  (funcall (dp-setter p) new-val)
  (setf (dt-grad p) nil))
