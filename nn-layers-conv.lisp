(in-package #:nn)

(defun im2col (input kh kw sh sw ph pw)
  "将 (batch, c, h, w) 转为 (batch*oh*ow, c*kh*kw) 矩阵。"
  (let* ((shape (vt-shape input))
         (batch (first shape))
         (channels (second shape))
         (in-h (third shape))
         (in-w (fourth shape))
         (oh (1+ (floor (- (+ in-h (* 2 ph)) kh) sh)))
         (ow (1+ (floor (- (+ in-w (* 2 pw)) kw) sw)))
         (col-rows (* batch oh ow))
         (col-cols (* channels kh kw))
         (result (vt-zeros (list col-rows col-cols))))
    (dotimes (b batch)
      (dotimes (i oh)
        (dotimes (j ow)
          (let ((row-idx (+ (* b oh ow) (* i ow) j)))
            (dotimes (c channels)
              (dotimes (ki kh)
                (dotimes (kj kw)
                  (let* ((ih (+ (* i sh) ki (- ph)))
                         (iw (+ (* j sw) kj (- pw)))
                         (val (if (and (>= ih 0) (< ih in-h)
                                       (>= iw 0) (< iw in-w))
                                  (vt-ref input b c ih iw)
                                  0.0d0))
                         (col-idx (+ (* c kh kw) (* ki kw) kj)))
                    (setf (vt-ref result row-idx col-idx) val)))))))))
    result))

(defun col2im (col kh kw sh sw ph pw in-shape)
  "col2im 逆操作。"
  (let* ((batch (first in-shape))
         (channels (second in-shape))
         (in-h (third in-shape))
         (in-w (fourth in-shape))
         (oh (1+ (floor (- (+ in-h (* 2 ph)) kh) sh)))
         (ow (1+ (floor (- (+ in-w (* 2 pw)) kw) sw)))
         (result (vt-zeros in-shape)))
    (dotimes (b batch)
      (dotimes (i oh)
        (dotimes (j ow)
          (let ((row-idx (+ (* b oh ow) (* i ow) j)))
            (dotimes (c channels)
              (dotimes (ki kh)
                (dotimes (kj kw)
                  (let* ((ih (+ (* i sh) ki (- ph)))
                         (iw (+ (* j sw) kj (- pw)))
                         (col-idx (+ (* c kh kw) (* ki kw) kj))
                         (col-val (vt-ref col row-idx col-idx)))
                    (when (and (>= ih 0) (< ih in-h)
                               (>= iw 0) (< iw in-w))
                      (incf (vt-ref result b c ih iw) col-val))))))))))
    result))

(defclass conv2d (layer)
  ((in-channels :initarg :in-channels
		:initform nil
		:reader conv-in-channels)
   (out-channels :initarg :out-channels
		 :reader conv-out-channels)
   (kernel-size :initarg :kernel-size
		:initform nil
		:reader conv-kernel-size)
   (stride :initarg :stride
	   :initform '(1 1)
	   :reader conv-stride)
   (padding :initarg :padding
	    :initform '(0 0)
	    :reader conv-padding)
   (use-bias :initarg :use-bias
	     :initform t
	     :reader conv-use-bias-p)
   (weight-init :initarg :weight-init
		:initform nil
		:accessor conv-weight-init)
   (weights :initarg :weights
	    :initform nil
	    :accessor conv-weights
	    :type (or null vt))
   (bias :initarg :bias
	 :initform nil
	 :accessor conv-bias
	 :type (or null vt))
   (dw :initarg :dw
       :initform nil
       :accessor conv-dw
       :type (or null vt))
   (db :initarg :db
       :initform nil
       :accessor conv-db
       :type (or null vt))
   (input-cache :initarg :input-cache
		:initform nil
		:accessor conv-input-cache)
   (col-cache :initarg :col-cache
	      :initform nil
	      :accessor conv-col-cache)
   (output-shape-cache :initarg :output-shape-cache
		       :initform nil
		       :accessor conv-output-shape-cache))
  (:documentation "2D 卷积层 (基于 im2col)."))

(defun make-conv2d (out-channels kernel-size
                    &key in-channels stride padding
                      use-bias weight-init
                      (name "conv2d") (trainable t))
  (let ((ks (if (listp kernel-size) kernel-size
		(list kernel-size kernel-size)))
        (st (if stride (if (listp stride) stride
                           (list stride stride))
		'(1 1)))
        (pd (if padding (if (listp padding) padding
                            (list padding padding))
		'(0 0))))
    (make-instance 'conv2d
		   :out-channels out-channels
		   :kernel-size ks :stride st :padding pd
		   :use-bias use-bias :in-channels in-channels
		   :weight-init weight-init
		   :name name :trainable trainable)))

(defun compute-conv-output-shape
    (in-h in-w kh kw sh sw ph pw)
  (list (1+ (floor (- (+ in-h (* 2 ph)) kh) sh))
        (1+ (floor (- (+ in-w (* 2 pw)) kw) sw))))

(defmethod forward ((l conv2d) input)
  (let* ((shape (vt-shape input))
         (batch (first shape))
         (in-c (or (conv-in-channels l) (second shape)))
         (in-h (third shape))
         (in-w (fourth shape))
         (out-c (conv-out-channels l))
         (kh (first (conv-kernel-size l)))
         (kw (second (conv-kernel-size l)))
         (sh (first (conv-stride l)))
         (sw (second (conv-stride l)))
         (ph (first (conv-padding l)))
         (pw (second (conv-padding l)))
         (oh-ow (compute-conv-output-shape
                 in-h in-w kh kw sh sw ph pw))
         (oh (first oh-ow))
         (ow (second oh-ow))
         (w (or (conv-weights l)
                (let* ((fan-in (* in-c kh kw))
                       (fan-out (* out-c kh kw))
                       (init (or (conv-weight-init l)
                                 (make-he-normal))))
                  (setf (slot-value l 'in-channels) in-c)
                  (setf (conv-weights l)
                        (vt-reshape
                         (init-weight
                          init
                          (list out-c (* in-c kh kw))
                          :fan-in fan-in :fan-out fan-out)
                         (list out-c in-c kh kw))))))
         (b (when (conv-use-bias-p l)
              (or (conv-bias l)
                  (setf (conv-bias l)
                        (vt-zeros (list out-c))))))
         (col (im2col input kh kw sh sw ph pw))
         (w-mat (vt-reshape w
                            (list out-c (* in-c kh kw))))
         (out-mat (vt-matmul col
                             (vt-transpose w-mat)))
         (out-reshaped (vt-reshape out-mat
                                   (list batch out-c oh ow)))
         (out (if b
                  (let ((b-view (vt-reshape b
                                            (list 1 out-c 1 1))))
                    (vt-+ out-reshaped b-view))
                  out-reshaped)))
    (setf (conv-input-cache l) input)
    (setf (conv-col-cache l) col)
    (setf (conv-output-shape-cache l)
          (list batch out-c oh ow))
    out))

(defmethod backward ((l conv2d) grad-output)
  (let* ((w (conv-weights l))
         (out-c (conv-out-channels l))
         (in-c (conv-in-channels l))
         (kh (first (conv-kernel-size l)))
         (kw (second (conv-kernel-size l)))
         (sh (first (conv-stride l)))
         (sw (second (conv-stride l)))
         (ph (first (conv-padding l)))
         (pw (second (conv-padding l)))
         (input (conv-input-cache l))
         (in-shape (vt-shape input))
         (col (conv-col-cache l))
         (out-shape (conv-output-shape-cache l))
         (batch (first out-shape))
         (oh (third out-shape))
         (ow (fourth out-shape))
         (go-2d (vt-reshape grad-output
                            (list (* batch oh ow) out-c)))
         (w-mat (vt-reshape w
                            (list out-c (* in-c kh kw))))
         (dw-mat (vt-matmul (vt-transpose go-2d) col))
         (dw (vt-reshape dw-mat
                         (list out-c in-c kh kw)))
         (d-col (vt-matmul go-2d w-mat))
         (d-input (col2im d-col kh kw sh sw ph pw in-shape)))
    (setf (conv-dw l) dw)
    (when (conv-use-bias-p l)
      (setf (conv-db l) (vt-sum go-2d :axis 0)))
    d-input))


(defmethod params ((l conv2d))
  (let ((r '()))
    (when (conv-weights l)
      (push (list l "weights" (conv-weights l)
                  #'(lambda (v) (setf (conv-weights l) v)))
            r))
    (when (and (conv-use-bias-p l) (conv-bias l))
      (push (list l "bias" (conv-bias l)
                  #'(lambda (v) (setf (conv-bias l) v)))
            r))
    (nreverse r)))

(defmethod grads ((l conv2d))
  (let ((r '()))
    (when (conv-dw l)
      (push (cons "weights" (conv-dw l)) r))
    (when (and (conv-use-bias-p l) (conv-db l))
      (push (cons "bias" (conv-db l)) r))
    (nreverse r)))


(defclass max-pool2d (layer)
  ((kernel-size :initarg :kernel-size
		:initform nil
		:reader pool-kernel-size)
   (stride :initarg :stride
	   :initform nil
	   :reader pool-stride)
   (padding :initarg :padding
	    :initform '(0 0)
	    :reader pool-padding)
   (cache :initarg :cache
	  :accessor pool-cache))
  (:documentation "2D 最大池化."))

(defun make-max-pool2d (kernel-size &key stride padding
				      (name "max-pool2d") (trainable nil))
  (let ((ks (if (listp kernel-size) kernel-size
		(list kernel-size kernel-size))))
    (make-instance 'max-pool2d
		   :kernel-size ks
		   :stride (or stride ks)
		   :padding (if padding
				(if (listp padding) padding
				    (list padding padding))
				'(0 0))
		   :name name :trainable trainable)))

(defmethod forward ((l max-pool2d) input)
  (let* ((shape (vt-shape input))
         (batch (first shape))
         (channels (second shape))
         (in-h (third shape))
         (in-w (fourth shape))
         (kh (first (pool-kernel-size l)))
         (kw (second (pool-kernel-size l)))
         (sh (first (pool-stride l)))
         (sw (second (pool-stride l)))
         (oh (1+ (floor (- in-h kh) sh)))
         (ow (1+ (floor (- in-w kw) sw)))
         (result-data
           (make-array (list batch channels oh ow)
                       :element-type 'double-float
                       :initial-element 0.0d0))
         (mask-data
           (make-array (list batch channels oh ow)
                       :element-type 'fixnum
                       :initial-element 0)))
    (dotimes (b batch)
      (dotimes (c channels)
        (dotimes (i oh)
          (dotimes (j ow)
            (let ((max-val most-negative-double-float)
                  (max-idx 0))
              (dotimes (ki kh)
                (dotimes (kj kw)
                  (let* ((ih (+ (* i sh) ki))
                         (iw (+ (* j sw) kj))
                         (val (coerce
                               (vt-ref input b c ih iw)
                               'double-float)))
                    (when (> val max-val)
                      (setf max-val val
                            max-idx (+ (* ki kw) kj))))))
              (setf (aref result-data b c i j) max-val)
              (setf (aref mask-data b c i j) max-idx))))))
    (setf (pool-cache l) (cons input mask-data))
    (let* ((total (reduce #'*
			  (list batch channels oh ow)))
           (flat (make-array total
                             :element-type 'double-float)))
      (dotimes (i total)
        (setf (aref flat i) (row-major-aref result-data i)))
      (vt-reshape (vt-from-sequence (coerce flat 'list))
                  (list batch channels oh ow)))))


(defmethod backward ((l max-pool2d) grad-output)
  (let* ((cached (pool-cache l))
         (input (first cached))
         (mask (second cached))
         (shape (vt-shape input))
         (batch (first shape))
         (channels (second shape))
         (in-h (third shape))
         (in-w (fourth shape))
         (go-shape (vt-shape grad-output))
         (oh (third go-shape))
         (ow (fourth go-shape))
         (kw (second (pool-kernel-size l)))
         (sh (first (pool-stride l)))
         (sw (second (pool-stride l)))
         (result-data
           (make-array (reduce #'* shape)
                       :element-type 'double-float
                       :initial-element 0.0d0)))
    (dotimes (b batch)
      (dotimes (c channels)
        (dotimes (i oh)
          (dotimes (j ow)
            (let* ((grad-val
                     (coerce
                      (vt-ref grad-output b c i j)
                      'double-float))
                   (idx (aref mask b c i j))
                   (ki (floor idx kw))
                   (kj (rem idx kw))
                   (ih (+ (* i sh) ki))
                   (iw (+ (* j sw) kj))
                   (flat-idx
                     (+ (* (+ (* (+ (* b channels) c)
                                 in-h)
			      ih)
			   in-w)
			iw)))
              (incf (aref result-data flat-idx)
                    grad-val))))))
    (let* ((total (reduce #'* shape))
           (flat (make-array total
                             :element-type 'double-float)))
      (dotimes (i total)
        (setf (aref flat i)
              (row-major-aref result-data i)))
      (vt-reshape (vt-from-sequence (coerce flat 'list))
                  shape))))


(defclass avg-pool2d (layer)
  ((kernel-size :initarg :kernel-size
		:initform nil
		:reader apool-kernel-size)
   (stride :initarg :stride
	   :initform nil
	   :reader apool-stride)
   (padding :initarg :padding
	    :initform '(0 0)
	    :reader apool-padding)
   (input-cache :initarg :input-cache
		:initform nil
		:accessor apool-input-cache))
  (:documentation "2D 平均池化."))

(defun make-avg-pool2d
    (kernel-size &key stride padding
		   (name "avg-pool2d") (trainable nil))
  (let ((ks (if (listp kernel-size) kernel-size
		(list kernel-size kernel-size))))
    (make-instance 'avg-pool2d
		   :kernel-size ks
		   :stride (or stride ks)
		   :padding (if padding
				(if (listp padding) padding
				    (list padding padding))
				'(0 0))
		   :name name :trainable trainable)))

(defmethod forward ((l avg-pool2d) input)
  (setf (apool-input-cache l) input)
  (let* ((shape (vt-shape input))
         (batch (first shape))
         (channels (second shape))
         (in-h (third shape))
         (in-w (fourth shape))
         (kh (first (apool-kernel-size l)))
         (kw (second (apool-kernel-size l)))
         (sh (first (apool-stride l)))
         (sw (second (apool-stride l)))
         (oh (1+ (floor (- in-h kh) sh)))
         (ow (1+ (floor (- in-w kw) sw)))
         (pool-size (coerce (* kh kw) 'double-float))
         (result-data
           (make-array (list batch channels oh ow)
                       :element-type 'double-float
                       :initial-element 0.0d0)))
    (dotimes (b batch)
      (dotimes (c channels)
        (dotimes (i oh)
          (dotimes (j ow)
            (let ((sum 0.0d0))
              (dotimes (ki kh)
                (dotimes (kj kw)
                  (incf sum
                        (coerce
                         (vt-ref input
                                 b
				 c
                                 (+ (* i sh) ki)
                                 (+ (* j sw) kj))
                         'double-float))))
              (setf (aref result-data b c i j)
                    (/ sum pool-size)))))))
    (let* ((total (reduce #'*
			  (list batch channels oh ow)))
           (flat (make-array total
                             :element-type 'double-float)))
      (dotimes (i total)
        (setf (aref flat i) (row-major-aref result-data i)))
      (vt-reshape (vt-from-sequence (coerce flat 'list))
                  (list batch channels oh ow)))))

(defmethod backward ((l avg-pool2d) grad-output)
  (let* ((input (apool-input-cache l))
         (shape (vt-shape input))
         (batch (first shape))
         (channels (second shape))
         (in-h (third shape))
         (in-w (fourth shape))
         (kh (first (apool-kernel-size l)))
         (kw (second (apool-kernel-size l)))
         (sh (first (apool-stride l)))
         (sw (second (apool-stride l)))
         (go-shape (vt-shape grad-output))
         (oh (third go-shape))
         (ow (fourth go-shape))
         (pool-size (coerce (* kh kw) 'double-float))
         (result-data
           (make-array (reduce #'* shape)
                       :element-type 'double-float
                       :initial-element 0.0d0)))
    (dotimes (b batch)
      (dotimes (c channels)
        (dotimes (i oh)
          (dotimes (j ow)
            (let ((grad (/ (coerce (vt-ref grad-output
                                           b c i j)
                                   'double-float)
                           pool-size)))
              (dotimes (ki kh)
                (dotimes (kj kw)
                  (let* ((ih (+ (* i sh) ki))
                         (iw (+ (* j sw) kj))
                         (flat-idx
                           (+ (* (+ (* (+ (* b channels) c)
                                       in-h) ih)
                                 in-w)
                              iw)))
                    (incf (aref result-data flat-idx)
                          grad)))))))))
    (let* ((total (reduce #'* shape))
           (flat (make-array total
                             :element-type 'double-float)))
      (dotimes (i total)
        (setf (aref flat i) (row-major-aref result-data i)))
      (vt-reshape (vt-from-sequence (coerce flat 'list))
                  shape))))

(defclass global-avg-pool2d (layer)
  ((cache-shape :accessor gap-cache-shape :type list))
  (:documentation "全局平均池化: (b,c,h,w) → (b,c)."))

(defun make-global-avg-pool2d
    (&key (name "global-avg-pool2d") (trainable nil))
  (make-instance 'global-avg-pool2d
		 :name name :trainable trainable))

(defmethod forward ((l global-avg-pool2d) input)
  (let* ((shape (vt-shape input))
         (batch (first shape))
         (channels (second shape))
         (h (third shape))
         (w (fourth shape)))
    (setf (gap-cache-shape l) shape)
    (let ((flat (vt-reshape input
                            (list batch channels (* h w)))))
      (vt-mean flat :axis -1))))

(defmethod backward ((l global-avg-pool2d) grad-output)
  "d_input = d_output / (h * w), 安全广播."
  (let* ((shape (gap-cache-shape l))
         (h (third shape))
         (w (fourth shape))
         (hw (* h w))
         ;; 显式构造全 1 张量相乘，消除对隐式广播的依赖
         (grad-scaled
           (vt-* (vt-reshape grad-output
                             (list (first shape)
                                   (second shape) 1 1))
                 (vt-const shape (/ 1.0d0 hw)))))
    grad-scaled))
