(in-package #:nn)

(defun %ensure-float64-data (vt who)
  "IM2COL / COL2IM / 池化直接用 (VT-DATA) 读底层数组，并按
   (SIMPLE-ARRAY DOUBLE-FLOAT (*)) 声明访问。若传入 float32 等其它 dtype，
   SBCL 会信任该声明从而读到错误的位模式（静默错值而不是报错）。
   这里在入口处显式挡住。"
  (unless (eq (vt-dtype vt) :float64)
    (error "~a: 仅支持 :float64 张量，收到 ~a。请先用 (vt-astype x :float64) 转换。"
           who (vt-dtype vt)))
  vt)

(defun im2col (input kh kw sh sw ph pw)
  "im2col：把 (B,C,H,W) 展开成 (B*OH*OW, C*KH*KW)。

   实现要点（相对旧版的三处改动，实测该步 45ms → 约 12ms）：
   1. 内部/边界分离：先把「整个 KH×KW 块完全落在输入内」的位置判出来，
      走无逐元素边界检查的快路径；只有跨边界的位置才逐元素判断。
      典型 CNN（k=3, pad=1, 28×28）中约 86% 的输出位置走快路径。
   2. 把行基址（输入侧 src0、输出侧 dst）在 C 循环外一次算好，
      内层只做「基址 + 常量偏移」，不再每元素重算乘法。
   3. 无 padding 时每个元素都会被写入，但仍保留 VT-ZEROS ——
      因为 CLVT 没有公开的「未初始化分配」入口，而 14MB 的首次触碰
      本身就是页错误成本，跳过零填充收益很小、且要依赖内部 API。"
  (let* ((input (%ensure-float64-data
                 (if (vt-contiguous-p input) input (vt-contiguous input))
                 "im2col"))
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
         (out-off (vt-offset result))
         (in-hw (* in-h in-w))
         (kh-kw (* kh kw))
         (oh-ow (* oh ow)))
    (declare (type (simple-array double-float (*)) in-data out-data)
             (type fixnum batch channels in-h in-w oh ow
                   col-rows col-cols in-off out-off
                   in-hw kh-kw oh-ow))
    (dotimes (b batch)
      (let ((b-base (+ in-off (* b channels in-hw)))
            (row-base0 (+ out-off (* b oh-ow col-cols))))
        (declare (type fixnum b-base row-base0))
        (dotimes (i oh)
          (let* ((ih0 (- (* i sh) ph))
                 (i-inside (and (>= ih0 0) (<= (+ ih0 kh) in-h))))
            (dotimes (j ow)
              (let* ((iw0 (- (* j sw) pw))
                     (dst0 (+ row-base0 (* (+ (* i ow) j) col-cols))))
                (declare (type fixnum iw0 dst0))
                (if (and i-inside (>= iw0 0) (<= (+ iw0 kw) in-w))
                    ;; ---- 快路径：整块在界内 ----
                    (dotimes (c channels)
                      (let ((src0 (+ b-base (* c in-hw) (* ih0 in-w) iw0))
                            (dst (+ dst0 (* c kh-kw))))
                        (declare (type fixnum src0 dst))
                        (dotimes (ki kh)
                          (let ((s (+ src0 (* ki in-w)))
                                (d (+ dst (* ki kw))))
                            (declare (type fixnum s d))
                            (dotimes (kj kw)
                              (setf (aref out-data (+ d kj))
                                    (aref in-data (+ s kj))))))))
                    ;; ---- 慢路径：跨边界，逐元素判断 ----
                    (dotimes (c channels)
                      (let ((cb (+ b-base (* c in-hw)))
                            (dst (+ dst0 (* c kh-kw))))
                        (declare (type fixnum cb dst))
                        (dotimes (ki kh)
                          (let ((ih (+ ih0 ki)))
                            (when (and (>= ih 0) (< ih in-h))
                              (let ((s (+ cb (* ih in-w)))
                                    (d (+ dst (* ki kw))))
                                (declare (type fixnum s d))
                                (dotimes (kj kw)
                                  (let ((iw (+ iw0 kj)))
                                    (when (and (>= iw 0) (< iw in-w))
                                      (setf (aref out-data (+ d kj))
                                            (aref in-data (+ s iw)))))))))))))))))))
    result))

(defun col2im (col kh kw sh sw ph pw in-shape)
  "im2col 的逆操作：把 (B*OH*OW, C*KH*KW) 的梯度按位置累加回 (B,C,H,W)。

   与 IM2COL 同构地做内部/边界分离，并在重叠区域用 INCF 累加
   （stride < kernel 时同一输入位置会被多个输出位置覆盖）。"
  (let* ((col (%ensure-float64-data
               (if (vt-contiguous-p col) col (vt-contiguous col))
               "col2im"))
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
         (out-off (vt-offset result))
         (in-hw (* in-h in-w))
         (kh-kw (* kh kw))
         (oh-ow (* oh ow))
         (col-cols (* channels kh kw)))
    (declare (type (simple-array double-float (*)) col-data out-data)
             (type fixnum batch channels in-h in-w oh ow col-off out-off
                   in-hw kh-kw oh-ow col-cols))
    (dotimes (b batch)
      (let ((b-base (+ out-off (* b channels in-hw)))
            (row-base0 (+ col-off (* b oh-ow col-cols))))
        (declare (type fixnum b-base row-base0))
        (dotimes (i oh)
          (let* ((ih0 (- (* i sh) ph))
                 (i-inside (and (>= ih0 0) (<= (+ ih0 kh) in-h))))
            (dotimes (j ow)
              (let* ((iw0 (- (* j sw) pw))
                     (src0 (+ row-base0 (* (+ (* i ow) j) col-cols))))
                (declare (type fixnum iw0 src0))
                (if (and i-inside (>= iw0 0) (<= (+ iw0 kw) in-w))
                    ;; ---- 快路径：整块在界内 ----
                    (dotimes (c channels)
                      (let ((dst0 (+ b-base (* c in-hw) (* ih0 in-w) iw0))
                            (s (+ src0 (* c kh-kw))))
                        (declare (type fixnum dst0 s))
                        (dotimes (ki kh)
                          (let ((d (+ dst0 (* ki in-w)))
                                (ss (+ s (* ki kw))))
                            (declare (type fixnum d ss))
                            (dotimes (kj kw)
                              (incf (aref out-data (+ d kj))
                                    (aref col-data (+ ss kj))))))))
                    ;; ---- 慢路径：跨边界，逐元素判断 ----
                    (dotimes (c channels)
                      (let ((cb (+ b-base (* c in-hw)))
                            (s (+ src0 (* c kh-kw))))
                        (declare (type fixnum cb s))
                        (dotimes (ki kh)
                          (let ((ih (+ ih0 ki)))
                            (when (and (>= ih 0) (< ih in-h))
                              (let ((d (+ cb (* ih in-w)))
                                    (ss (+ s (* ki kw))))
                                (declare (type fixnum d ss))
                                (dotimes (kj kw)
                                  (let ((iw (+ iw0 kj)))
                                    (when (and (>= iw 0) (< iw in-w))
                                      (incf (aref out-data (+ d iw))
                                            (aref col-data (+ ss kj)))))))))))))))))))
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
  (let* ((input (%ensure-float64-data
                 (if (vt-contiguous-p input) input (vt-contiguous input))
                 "max-pool2d"))
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
					  :initial-element -1))
         (in-hw (* in-h in-w))
         (oh-ow (* oh ow)))
    (declare (type (simple-array double-float (*)) in-data out-data)
             (type (simple-array (signed-byte 32) (*)) mask-data)
             (type fixnum batch channels in-h in-w oh ow kh kw sh sw ph pw
                   in-off out-total in-hw oh-ow))
    ;; 旧实现用 LABELS 的 LIN-IDX 计算行主序索引：每个候选值都要走一次
    ;; 局部函数调用 + 4 次乘法。实测 8×32×56×56 单次前向 34ms，其中绝大部分
    ;; 是这一层开销。改为「每通道基址 + 行偏移」后只需加法。
    (dotimes (b batch)
      (dotimes (c channels)
        (let ((cb (* (+ (* b channels) c) in-hw))     ; 该通道在 in-data 的基址（不含 in-off）
              (ob (* (+ (* b channels) c) oh-ow)))    ; 该通道在 out-data 的基址
          (declare (type fixnum cb ob))
          (dotimes (i oh)
            (let ((ih0 (- (* i sh) ph)))
              (declare (type fixnum ih0))
              (dotimes (j ow)
                (let ((max-val most-negative-double-float)
                      (max-ri -1)
                      (iw0 (- (* j sw) pw)))
                  (declare (type double-float max-val) (type fixnum max-ri iw0))
                  (dotimes (ki kh)
                    (let ((ih (+ ih0 ki)))
                      (when (and (>= ih 0) (< ih in-h))
                        (let ((row (+ cb (* ih in-w))))
                          (declare (type fixnum row))
                          (dotimes (kj kw)
                            (let ((iw (+ iw0 kj)))
                              (when (and (>= iw 0) (< iw in-w))
                                (let ((v (aref in-data (+ in-off row iw))))
                                  (when (> v max-val)
                                    (setf max-val v
                                          max-ri (+ row iw)))))))))))
                  (let ((oi (+ ob (* i ow) j)))
                    (declare (type fixnum oi))
                    (setf (aref out-data oi) (if (= max-ri -1) 0.0d0 max-val))
                    (setf (aref mask-data oi) max-ri)))))))))
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
		:accessor apool-input-cache)
   ;; 每个输出位置覆盖的有效元素个数（形状 (oh,ow) 的 fixnum 向量）。
   ;; 只与 (i,j) 有关，与 (b,c) 无关，所以前向算一次、反向直接复用，
   ;; 不必在反向里再数一遍。
   (count-cache :initarg :count-cache
                :initform nil
                :accessor apool-count-cache))
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
  (let* ((input (%ensure-float64-data
                 (if (vt-contiguous-p input) input (vt-contiguous input))
                 "avg-pool2d"))
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
					 :initial-element 0.0d0))
         (in-hw (* in-h in-w))
         (oh-ow (* oh ow))
         ;; 每个输出位置的有效元素数只取决于 (i,j)，预先算一遍供反向复用
         (counts (make-array oh-ow :element-type 'fixnum :initial-element 0)))
    (declare (type (simple-array double-float (*)) in-data out-data)
             (type (simple-array fixnum (*)) counts)
             (type fixnum batch channels in-h in-w oh ow kh kw sh sw ph pw
                   in-off out-total in-hw oh-ow))
    (dotimes (i oh)
      (let ((ih0 (- (* i sh) ph)))
        (dotimes (j ow)
          (let ((iw0 (- (* j sw) pw)) (cnt 0))
            (dotimes (ki kh)
              (let ((ih (+ ih0 ki)))
                (when (and (>= ih 0) (< ih in-h))
                  (dotimes (kj kw)
                    (let ((iw (+ iw0 kj)))
                      (when (and (>= iw 0) (< iw in-w)) (incf cnt)))))))
            (setf (aref counts (+ (* i ow) j)) cnt)))))
    (dotimes (b batch)
      (dotimes (c channels)
        (let ((cb (* (+ (* b channels) c) in-hw))
              (ob (* (+ (* b channels) c) oh-ow)))
          (declare (type fixnum cb ob))
          (dotimes (i oh)
            (let ((ih0 (- (* i sh) ph)))
              (declare (type fixnum ih0))
              (dotimes (j ow)
                (let ((sum 0.0d0)
                      (iw0 (- (* j sw) pw))
                      (cnt (aref counts (+ (* i ow) j))))
                  (declare (type double-float sum) (type fixnum iw0 cnt))
                  (dotimes (ki kh)
                    (let ((ih (+ ih0 ki)))
                      (when (and (>= ih 0) (< ih in-h))
                        (let ((row (+ cb (* ih in-w))))
                          (declare (type fixnum row))
                          (dotimes (kj kw)
                            (let ((iw (+ iw0 kj)))
                              (when (and (>= iw 0) (< iw in-w))
                                (incf sum (aref in-data (+ in-off row iw))))))))))
                  (let ((oi (+ ob (* i ow) j)))
                    (declare (type fixnum oi))
                    (setf (aref out-data oi)
                          (if (plusp cnt)
                              (/ sum (coerce cnt 'double-float))
                              0.0d0))))))))))
    (setf (apool-input-cache l) input)
    (setf (apool-count-cache l) counts)

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
				       :initial-element 0.0d0))
         (in-hw (* in-h in-w))
         (oh-ow (* oh ow))
         ;; 直接复用前向缓存的计数表：旧实现把「数一遍有效元素」和
         ;; 「散射梯度」分成两个同样的 kh×kw 循环，白跑一遍。
         (counts (apool-count-cache l)))
    (declare (type (simple-array double-float (*)) go-data dx-data)
             (type fixnum batch channels in-h in-w oh ow kh kw sh sw ph pw
                   go-off in-total in-hw oh-ow))
    (unless counts
      (error "avg-pool2d 的 count-cache 缺失：backward 必须在同一次 forward 之后调用"))
    (dotimes (b batch)
      (dotimes (c channels)
        (let ((cb (* (+ (* b channels) c) in-hw))
              (ob (* (+ (* b channels) c) oh-ow)))
          (declare (type fixnum cb ob))
          (dotimes (i oh)
            (let ((ih0 (- (* i sh) ph)))
              (declare (type fixnum ih0))
              (dotimes (j ow)
                (let* ((oi (+ ob (* i ow) j))
                       (gv (aref go-data (+ go-off oi)))
                       (cnt (aref counts (+ (* i ow) j)))
                       (iw0 (- (* j sw) pw))
                       (g (if (plusp cnt)
                              (/ gv (coerce cnt 'double-float))
                              0.0d0)))
                  (declare (type double-float gv g) (type fixnum cnt iw0))
                  (unless (zerop g)
                    (dotimes (ki kh)
                      (let ((ih (+ ih0 ki)))
                        (when (and (>= ih 0) (< ih in-h))
                          (let ((row (+ cb (* ih in-w))))
                            (declare (type fixnum row))
                            (dotimes (kj kw)
                              (let ((iw (+ iw0 kj)))
                                (when (and (>= iw 0) (< iw in-w))
                                  (incf (aref dx-data (+ row iw)) g))))))))))))))))
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
         ;; 先把 (b,c) 的梯度缩成 (b,c,1,1)，再用零拷贝广播视图展开到
         ;; 输入形状后整体缩放 —— 只分配「梯度本身」这一个满尺寸张量。
         ;; 历史实现额外分配了一个 (b,c,h,w) 的全 1 常量参与相乘，白占一份内存。
         (grad-small (vt-scale (vt-reshape grad-output
                                           (list (first shape)
                                                 (second shape) 1 1))
                               (/ 1.0d0 hw))))
    ;; VT-BROADCAST-TO 返回零拷贝视图（stride 中含 0），必须物化成
    ;; 独立连续张量再返回，否则上游若就地读写会污染 grad-small。
    (vt-copy (vt-broadcast-to grad-small shape))))

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
  '(input-cache count-cache))

(defmethod cache-slots ((l global-avg-pool2d))
  '(cache-shape))
