(in-package #:nn)

(defun im2col (input kh kw sh sw ph pw)
  "im2col 优化版：
   - 入口 vt-contiguous 归一化
   - 内层用 aref + 手动预计算 offset，避免 vt-ref 泛型 dispatch
   - 循环外提取公共子表达式"
  (let* ((input (if (vt-contiguous-p input) input (vt-contiguous input)))
         (shape (vt-shape input))
         (batch (first shape))
         (channels (second shape))
         (in-h (third shape))
         (in-w (fourth shape))
         (oh (1+ (floor (- (+ in-h (* 2 ph)) kh) sh)))
         (ow (1+ (floor (- (+ in-w (* 2 pw)) kw) sw)))
         (col-rows (* batch oh ow))
         (col-cols (* channels kh kw))
         (result (vt-zeros (list col-rows col-cols)))
         (in-data (vt-data input))
         (in-off (vt-offset input))
         (out-data (vt-data result))
         (out-off (vt-offset result)))
    (declare (type (simple-array double-float (*)) in-data out-data)
             (type fixnum batch channels in-h in-w oh ow
                   col-rows col-cols in-off out-off))
    (let ((in-hw (* in-h in-w))       ; H*W 预计算
          (kh-kw (* kh kw))
          (oh-ow (* oh ow)))
      (declare (type fixnum in-hw kh-kw oh-ow))
      (dotimes (b batch)
        (let ((b-base (+ in-off (* b channels in-hw))))
          (declare (type fixnum b-base))
          (dotimes (i oh)
            (dotimes (j ow)
              (let* ((row-idx (+ (* b oh-ow) (* i ow) j))
                     (row-base (+ out-off (* row-idx col-cols))))
                (declare (type fixnum row-base))
                (dotimes (c channels)
                  (let ((c-base (+ b-base (* c in-hw)))
                        (col-base (* c kh-kw)))
                    (declare (type fixnum c-base col-base))
                    (dotimes (ki kh)
                      (let ((ih (+ (* i sh) ki (- ph))))
                        (when (and (>= ih 0) (< ih in-h))
                          (let ((ih-base (+ c-base (* ih in-w)))
                                (out-base (+ row-base col-base (* ki kw))))
                            (declare (type fixnum ih-base out-base))
                            (dotimes (kj kw)
                              (let ((iw (+ (* j sw) kj (- pw))))
                                (when (and (>= iw 0) (< iw in-w))
                                  (setf (aref out-data (+ out-base kj))
                                        (aref in-data (+ ih-base iw)))))))))))))))))
      result)))

(defun col2im (col kh kw sh sw ph pw in-shape)
  "col2im 逆操作。"
  (let* ((col (if (vt-contiguous-p col) col (vt-contiguous col)))
         (batch (first in-shape))
         (channels (second in-shape))
         (in-h (third in-shape))
         (in-w (fourth in-shape))
         (oh (1+ (floor (- (+ in-h (* 2 ph)) kh) sh)))
         (ow (1+ (floor (- (+ in-w (* 2 pw)) kw) sw)))
         (result (vt-zeros in-shape))
         (col-data (vt-data col))
         (col-off (vt-offset col))
         (out-data (vt-data result))
         (out-off (vt-offset result)))
    (declare (type (simple-array double-float (*)) col-data out-data)
             (type fixnum batch channels in-h in-w oh ow col-off out-off))
    (let ((in-hw (* in-h in-w))
          (kh-kw (* kh kw))
          (oh-ow (* oh ow))
          (col-cols (* channels kh kw)))
      (declare (type fixnum in-hw kh-kw oh-ow col-cols))
      (dotimes (b batch)
        (let ((b-base (+ out-off (* b channels in-hw))))
          (dotimes (i oh)
            (dotimes (j ow)
              (let* ((row-idx (+ (* b oh-ow) (* i ow) j))
                     (row-base (+ col-off (* row-idx col-cols))))
                (dotimes (c channels)
                  (let ((c-base (+ b-base (* c in-hw)))
                        (col-base (* c kh-kw)))
                    (dotimes (ki kh)
                      (let ((ih (+ (* i sh) ki (- ph))))
                        (when (and (>= ih 0) (< ih in-h))
                          (let ((ih-base (+ c-base (* ih in-w)))
                                (in-base (+ row-base col-base (* ki kw))))
                            (dotimes (kj kw)
                              (let ((iw (+ (* j sw) kj (- pw))))
                                (when (and (>= iw 0) (< iw in-w))
                                  (incf (aref out-data (+ ih-base iw))
                                        (aref col-data (+ in-base kj)))))))))))))))))
      result)))

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
                      (use-bias t) weight-init
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
		   :kernel-size ks
		   :stride st
		   :padding pd
		   :use-bias use-bias
		   :in-channels in-channels
		   :weight-init weight-init
		   :name name
		   :trainable trainable)))

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
			  :layout :out-in
                          :fan-in fan-in
			  :fan-out fan-out)
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
         ;; out-mat 的行序是 (batch, oh, ow)、列是通道。必须先 reshape 成
         ;; (batch, oh, ow, out-c)（此时行主序分组与 col 的行顺序一致），
         ;; 再转置成 (batch, out-c, oh, ow)。
         (out-reshaped (vt-transpose
                        (vt-reshape out-mat (list batch oh ow out-c))
                        '(0 3 1 2)))
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
         ;; grad-output 形状为 (batch, out-c, oh, ow)。转成 (batch, oh, ow, out-c)
         ;; 后再 reshape，行序才是 (batch, oh, ow)，与 col / w-mat 的约定一致；
         (go-2d (vt-reshape (vt-transpose grad-output '(0 2 3 1))
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
    (when (conv-weights l)
      (push (cons "weights" (conv-dw l)) r))
    (when (and (conv-use-bias-p l) (conv-bias l))
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
  (let* ((ks (if (listp kernel-size) kernel-size
		 (list kernel-size kernel-size)))
         (str (if (null stride) ks
                  (if (listp stride) stride
                      (list stride stride)))))
    (make-instance 'max-pool2d
		   :kernel-size ks
		   :stride str
		   :padding (if padding
				(if (listp padding) padding
				    (list padding padding))
				'(0 0))
		   :name name :trainable trainable)))

(defmethod forward ((l max-pool2d) input)
  ;; 本层用 (vt-data input) + 行主序线性索引直接访问底层数组，
  ;; 隐含假设输入连续。非连续视图（例如 batch-norm 的输出）会读错元素，
  ;; 因此入口处先归一化为连续内存。
  (let* ((input (if (vt-contiguous-p input) input (vt-contiguous input)))
         (shape (vt-shape input))
         (batch (first shape))
	 (channels (second shape))
         (in-h (third shape))
	 (in-w (fourth shape))
         (kh (first (pool-kernel-size l)))
	 (kw (second (pool-kernel-size l)))
         (sh (first (pool-stride l)))
	 (sw (second (pool-stride l)))
         (ph (first (pool-padding l)))
	 (pw (second (pool-padding l)))
         (oh (1+ (floor (- (+ in-h (* 2 ph)) kh) sh)))
         (ow (1+ (floor (- (+ in-w (* 2 pw)) kw) sw)))
         (in-data (vt-data input))
	 (in-off  (vt-offset input))
         (out-total (* batch channels oh ow))
         (out-data (make-array out-total :element-type 'double-float
					 :initial-element 0.0d0))
         ;; mask: row-major linear index into input (-1 means all-pad position)
         (mask-data (make-array out-total :element-type '(signed-byte 32)
					  :initial-element -1)))
    (labels ((lin-idx (b c h w)
               (+ (* (+ (* (+ (* b channels) c)
			   in-h)
			h)
		     in-w)
		  w)))
      (dotimes (b batch)
        (dotimes (c channels)
          (dotimes (i oh)
            (dotimes (j ow)
              (let ((max-val most-negative-double-float)
                    (max-ri -1))
                (dotimes (ki kh)
                  (dotimes (kj kw)
                    (let ((ih (- (+ (* i sh) ki) ph))
                          (iw (- (+ (* j sw) kj) pw)))
                      (when (and (<= 0 ih)
				 (< ih in-h)
				 (<= 0 iw)
				 (< iw in-w))
                        (let* ((ri (lin-idx b c ih iw))
                               (val (aref in-data (+ in-off ri))))
                          (when (> val max-val)
                            (setf max-val val
                                  max-ri ri)))))))
                (let ((oi (+ (* (+ (* b channels) c)
				oh ow)
			     (* i ow) j)))
                  (setf (aref out-data oi) (if (= max-ri -1) 0.0d0 max-val))
                  (setf (aref mask-data oi) max-ri))))))))
    (setf (pool-cache l) (list input mask-data))
    (let ((out-vt (vt-from-array out-data :dtype :float64 :fast t)))
      (vt-reshape out-vt (list batch channels oh ow)))))


(defmethod backward ((l max-pool2d) grad-output)
  (let* ((grad-output (if (vt-contiguous-p grad-output)
                          grad-output
                          (vt-contiguous grad-output)))
	 (cached (pool-cache l))
         (input (first cached))
         (mask (second cached))
         (shape (vt-shape input))
         (batch (first shape))
	 (channels (second shape))
         (go-shape (vt-shape grad-output))
         (oh (third go-shape))
	 (ow (fourth go-shape))
         (in-total (reduce #'* shape))
         (out-total (* batch channels oh ow))
         (go-data (vt-data grad-output))
	 (go-off  (vt-offset grad-output))
         (dx-data (make-array in-total :element-type 'double-float
				       :initial-element 0.0d0)))
    (dotimes (oi out-total)
      (let ((ri (aref mask oi))
            (gv (aref go-data (+ go-off oi))))
        (when (>= ri 0)
          (incf (aref dx-data ri) gv))))
    (let ((dx (vt-from-array dx-data :dtype :float64 :fast t)))
      (vt-reshape dx shape))))

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
  (let* ((ks (if (listp kernel-size) kernel-size
		 (list kernel-size kernel-size)))
         (str (if (null stride) ks
                  (if (listp stride) stride
                      (list stride stride)))))
    (make-instance 'avg-pool2d
		   :kernel-size ks
		   :stride str
		   :padding (if padding
				(if (listp padding) padding
				    (list padding padding))
				'(0 0))
		   :name name :trainable trainable)))

(defmethod forward ((l avg-pool2d) input)
  ;; 同 max-pool2d：入口处先归一化为连续内存，避免非连续视图读错元素。
  (let* ((input (if (vt-contiguous-p input) input (vt-contiguous input)))
         (shape (vt-shape input))
         (batch (first shape))
	 (channels (second shape))
         (in-h (third shape))
	 (in-w (fourth shape))
         (kh (first (apool-kernel-size l)))
	 (kw (second (apool-kernel-size l)))
         (sh (first (apool-stride l)))
	 (sw (second (apool-stride l)))
         (ph (first (apool-padding l)))
	 (pw (second (apool-padding l)))
         (oh (1+ (floor (- (+ in-h (* 2 ph)) kh) sh)))
         (ow (1+ (floor (- (+ in-w (* 2 pw)) kw) sw)))
         (in-data (vt-data input))
	 (in-off  (vt-offset input))
         (out-total (* batch channels oh ow))
         (out-data (make-array out-total :element-type 'double-float
					 :initial-element 0.0d0)))
    (labels ((lin-idx (b c h w)
               (+ (* (+ (* (+ (* b channels) c)
			   in-h)
			h)
		     in-w)
		  w)))
      (dotimes (b batch)
        (dotimes (c channels)
          (dotimes (i oh)
            (dotimes (j ow)
              (let ((sum 0.0d0) (cnt 0))
                (dotimes (ki kh)
                  (dotimes (kj kw)
                    (let ((ih (- (+ (* i sh) ki) ph))
                          (iw (- (+ (* j sw) kj) pw)))
                      (when (and (<= 0 ih)
				 (< ih in-h)
				 (<= 0 iw)
				 (< iw in-w))
                        (incf sum (aref in-data (+ in-off (lin-idx b c ih iw))))
                        (incf cnt)))))
                (let ((oi (+ (* (+ (* b channels) c)
				oh ow)
			     (* i ow) j)))
                  (setf (aref out-data oi)
			(if (plusp cnt)
			    (/ sum (coerce cnt 'double-float)) 0.0d0)))))))))
    (setf (apool-input-cache l) input)

    (let ((out-vt (vt-from-array out-data :dtype :float64 :fast t)))
      (vt-reshape out-vt (list batch channels oh ow)))))

(defmethod backward ((l avg-pool2d) grad-output)
  (let* ((input (apool-input-cache l))
         (shape (vt-shape input))
         (batch (first shape))
	 (channels (second shape))
         (in-h (third shape))
	 (in-w (fourth shape))
         (go-shape (vt-shape grad-output))
         (oh (third go-shape))
	 (ow (fourth go-shape))
         (kh (first (apool-kernel-size l)))
	 (kw (second (apool-kernel-size l)))
         (sh (first (apool-stride l)))
	 (sw (second (apool-stride l)))
         (ph (first (apool-padding l)))
	 (pw (second (apool-padding l)))
         (in-total (reduce #'* shape))
         (go-data (vt-data grad-output))
	 (go-off  (vt-offset grad-output))
         (dx-data (make-array in-total :element-type 'double-float
				       :initial-element 0.0d0)))
    (labels ((lin-idx (b c h w)
               (+ (* (+ (* (+ (* b channels) c)
			   in-h)
			h)
		     in-w)
		  w)))
      (dotimes (b batch)
        (dotimes (c channels)
          (dotimes (i oh)
            (dotimes (j ow)
              (let* ((oi (+ (* (+ (* b channels) c) oh ow)
			    (* i ow) j))
                     (gv (aref go-data (+ go-off oi)))
                     (cnt 0))
                ;; count valid cells (must match forward)
                (dotimes (ki kh)
                  (dotimes (kj kw)
                    (let ((ih (- (+ (* i sh) ki) ph))
                          (iw (- (+ (* j sw) kj) pw)))
                      (when (and (<= 0 ih)
				 (< ih in-h)
				 (<= 0 iw)
				 (< iw in-w))
                        (incf cnt)))))
                (let ((g (if (plusp cnt)
			     (/ gv (coerce cnt 'double-float))
			     0.0d0)))
                  (dotimes (ki kh)
                    (dotimes (kj kw)
                      (let ((ih (- (+ (* i sh) ki) ph))
                            (iw (- (+ (* j sw) kj) pw)))
                        (when (and (<= 0 ih)
				   (< ih in-h)
				   (<= 0 iw)
				   (< iw in-w))
                          (incf (aref dx-data (lin-idx b c ih iw)) g))))))))))))
    (let ((dx (vt-from-array dx-data :dtype :float64 :fast t)))
      (vt-reshape dx shape))))

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

;; ---- grad-slots ----
(defmethod grad-slots ((l conv2d))
  (let ((slots '(dw)))
    (when (conv-use-bias-p l) (push 'db slots))
    slots))

(defmethod grad-slots ((l max-pool2d)) '())

(defmethod grad-slots ((l avg-pool2d)) '())

(defmethod grad-slots ((l global-avg-pool2d)) '())

(defmethod cache-slots ((l conv2d))
  '(input-cache col-cache output-shape-cache))

(defmethod cache-slots ((l max-pool2d))
  '(cache))

(defmethod cache-slots ((l avg-pool2d))
  '(input-cache))

(defmethod cache-slots ((l global-avg-pool2d))
  '(cache-shape))
