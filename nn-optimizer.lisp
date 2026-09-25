(in-package #:nn)

(defun clip-gradient (grad clip)
  "对【单个】梯度张量做 L2 裁剪。
   注意：优化器的 grad-clip 语义已改为全局范数裁剪（见 GLOBAL-CLIP-COEFF），
   本函数仅作为单张量工具保留，不要用于 optimizer-step。"
  (if (> clip 0.0d0)
      (let ((gnorm (sqrt (vt-item (vt-sum (vt-square grad))))))
        (if (> gnorm clip)
            (vt-scale grad (/ clip gnorm))
            grad))
      grad))

;;; ------------------------------------------------------------------
;;; 全局梯度范数裁剪
;;; ------------------------------------------------------------------
;;; PyTorch 的 clip_grad_norm_ 用的是「所有参数梯度的全局 L2 范数」，
;;; 而不是逐个参数张量各自裁剪。这里提供全局版本，
;;; 各优化器在 optimizer-step 开头调用一次。

(defun global-grad-norm (grad-list)
  "GRAD-LIST 中所有梯度张量的全局 L2 范数。"
  (let ((sq 0.0d0))
    (dolist (g grad-list (sqrt sq))
      (let ((tensor (cdr g)))
        (when tensor
          (incf sq (vt-item (vt-sum (vt-square tensor)))))))))

(defun global-clip-coeff (grad-list clip)
  "全局范数裁剪系数。CLIP <= 0 时返回 1.0（不裁剪）。"
  (if (<= clip 0.0d0)
      1.0d0
      (let ((n (global-grad-norm grad-list)))
        (if (> n clip) (/ clip n) 1.0d0))))

(defun apply-clip (grad coeff)
  "按系数缩放梯度；COEFF = 1.0 时原样返回。"
  (if (= coeff 1.0d0) grad (vt-scale grad coeff)))

(defclass sgd (optimizer)
  ((momentum :initarg :momentum
             :initform 0.0d0
             :accessor sgd-momentum)
   (nesterov :initarg :nesterov
             :initform nil
             :accessor sgd-nesterov-p))
  (:documentation "随机梯度下降 (可选动量和 Nesterov)."))

(defun make-sgd (&key lr momentum nesterov grad-clip weight-decay)
  (make-instance 'sgd
                 :lr (coerce (or lr 1e-3) 'double-float)
                 :momentum (or momentum 0.0d0)
                 :nesterov nesterov
                 :grad-clip (or grad-clip 0.0d0)
                 :weight-decay (or weight-decay 0.0d0)))

(defmethod optimizer-step ((opt sgd) param-list grad-list)
  (let* ((lr (optimizer-lr opt))
        (mu (sgd-momentum opt))
        (wd (optimizer-weight-decay opt))
        (clip (optimizer-grad-clip opt))
        (clip-coeff (global-clip-coeff grad-list clip))
        (registry (optimizer-state-registry opt)))
    (loop for p in param-list
          for (gname . grad) in grad-list
          for owner = (first p)
          for name = (second p)
          for param = (third p)
          for setter = (fourth p)
          when (and param grad (layer-trainable-p owner))
            do (let* ((base-key (list owner name))
                      (g (apply-clip grad clip-coeff))
                      (g-reg (if (> wd 0.0d0)
                                 (vt-+ g (vt-scale param wd))
                                 g))
                      (buf (or (gethash base-key registry)
			       (setf (gethash base-key registry)
                                     (vt-zeros (vt-shape param)))))
                      (new-buf (vt-+ (vt-scale buf mu) g-reg)))
                 (setf (gethash base-key registry) new-buf)
                 (let ((update (if (sgd-nesterov-p opt)
                                   (vt-+ g-reg (vt-scale new-buf mu))
                                   new-buf)))
                   (funcall setter (vt-- param (vt-scale update lr))))))))

(defun make-sgd-momentum (&key lr (momentum 0.9d0) nesterov
                            grad-clip weight-decay)
  "带动量的 SGD。等价于 (make-sgd :momentum MOMENTUM ...)，MOMENTUM 默认 0.9。
   注：导出的 SGD-MOMENTUM 符号已是 SGD 类的动量访问器，
   因此这里不再单独定义 sgd-momentum 类。"
  (make-sgd :lr lr :momentum momentum :nesterov nesterov
            :grad-clip grad-clip :weight-decay weight-decay))

(defclass adam (optimizer)
  ((beta1 :initarg :beta1
          :initform 0.9d0
          :accessor adam-beta1)
   (beta2 :initarg :beta2
          :initform 0.999d0
          :accessor adam-beta2)
   (eps :initarg :eps
        :initform 1.0d-8
        :accessor adam-eps)
   (amsgrad :initarg :amsgrad
            :initform nil
            :accessor adam-amsgrad-p))
  (:documentation "Adam 优化器."))

(defun make-adam (&key lr beta1 beta2 eps amsgrad grad-clip
                    weight-decay)
  (make-instance 'adam
                 :lr (coerce (or lr 1e-3) 'double-float)
                 :beta1 (or beta1 0.9d0)
                 :beta2 (or beta2 0.999d0)
                 :eps (or eps 1.0d-8)
                 :amsgrad amsgrad
                 :grad-clip (or grad-clip 0.0d0)
                 :weight-decay (or weight-decay 0.0d0)))

(defmethod optimizer-step ((opt adam) param-list grad-list)
  (let* ((lr (optimizer-lr opt))
        (b1 (adam-beta1 opt))
        (b2 (adam-beta2 opt))
        (eps (adam-eps opt))
        (wd (optimizer-weight-decay opt))
        (clip (optimizer-grad-clip opt))
        (clip-coeff (global-clip-coeff grad-list clip))
        (tt (incf (optimizer-step-count opt)))
        (registry (optimizer-state-registry opt)))
    (loop for p in param-list
          for (gname . grad) in grad-list
	  for owner = (first p)
          for name = (second p)
          for param = (third p)
          for setter = (fourth p)
          when (and param grad (layer-trainable-p owner))
            do (let* ((base-key (list owner name))
                      (key-m (list base-key 'm))
                      (key-v (list base-key 'v))
                      (key-vmax (list base-key 'vmax))
                      (g (apply-clip grad clip-coeff))
                      (g-reg (if (> wd 0.0d0)
				 (vt-+ g (vt-scale param wd))
				 g))
                      (bc1 (/ 1.0d0 (- 1.0d0 (expt b1 tt))))
                      (bc2 (/ 1.0d0 (- 1.0d0 (expt b2 tt)))))
		 ;; 一阶矩
		 (let ((m (or (gethash key-m registry)
                              (setf (gethash key-m registry)
				    (vt-zeros (vt-shape param))))))
                   (setf (gethash key-m registry)
			 (vt-+ (vt-scale m b1)
                               (vt-scale g-reg
					 (- 1.0d0 b1)))))
		 ;; 二阶矩
		 (let ((v (or (gethash key-v registry)
                              (setf (gethash key-v registry)
				    (vt-zeros (vt-shape param))))))
                   (setf (gethash key-v registry)
			 (vt-+ (vt-scale v b2)
                               (vt-scale (vt-square g-reg)
					 (- 1.0d0 b2)))))
		 ;; 更新参数
		 (let* ((m-hat (vt-scale (gethash key-m registry)
					 bc1))
			(v-hat (vt-scale (gethash key-v registry)
					 bc2))
			(new-p
                          (if (adam-amsgrad-p opt)

			      (let* ((v-max-old (or (gethash key-vmax registry)
						    (setf (gethash key-vmax registry)
							  (vt-zeros (vt-shape param)))))
				     (v-max (vt-map #'max v-max-old v-hat)))
				(setf (gethash key-vmax registry) v-max)
				(vt-- param
				      (vt-scale
				       (vt-/ m-hat
					     (vt-+ (vt-map #'sqrt v-max) eps))
				       lr)))
                              (vt-- param
                                    (vt-scale
                                     (vt-/ m-hat
                                           (vt-+ (vt-map #'sqrt v-hat)
						 eps))
                                     lr)))))
                   (funcall setter new-p))))))

(defclass adamw (adam) ()
  (:documentation "AdamW: 解耦权重衰减."))

(defun make-adamw (&key lr beta1 beta2 eps amsgrad grad-clip
                     weight-decay)
  (make-instance 'adamw
                 :lr (coerce (or lr 1e-3) 'double-float)
                 :beta1 (or beta1 0.9d0)
                 :beta2 (or beta2 0.999d0)
                 :eps (or eps 1.0d-8)
                 :amsgrad amsgrad
                 :grad-clip (or grad-clip 0.0d0)
                 :weight-decay (or weight-decay 0.01d0)))

(defmethod optimizer-step ((opt adamw) param-list grad-list)
  (let* ((lr (optimizer-lr opt))
        (b1 (adam-beta1 opt))
        (b2 (adam-beta2 opt))
        (eps (adam-eps opt))
        (wd (optimizer-weight-decay opt))
        (clip (optimizer-grad-clip opt))
        (clip-coeff (global-clip-coeff grad-list clip))
        (tt (incf (optimizer-step-count opt)))
        (registry (optimizer-state-registry opt)))
    (loop for p in param-list
          for (gname . grad) in grad-list
	  for owner = (first p)
          for name = (second p)
          for param = (third p)
          for setter = (fourth p)
          when (and param grad (layer-trainable-p owner))
            do (let* ((base-key (list owner name))
                      (key-m (list base-key 'm))
                      (key-v (list base-key 'v))
                      (g (apply-clip grad clip-coeff))
                      (bc1 (/ 1.0d0 (- 1.0d0 (expt b1 tt))))
                      (bc2 (/ 1.0d0 (- 1.0d0 (expt b2 tt)))))
		 ;; 一阶矩 (无 wd)
		 (let ((m (or (gethash key-m registry)
                              (setf (gethash key-m registry)
				    (vt-zeros (vt-shape param))))))
                   (setf (gethash key-m registry)
			 (vt-+ (vt-scale m b1)
                               (vt-scale g
					 (- 1.0d0 b1)))))
		 ;; 二阶矩 (无 wd)
		 (let ((v (or (gethash key-v registry)
			      (setf (gethash key-v registry)
                                    (vt-zeros (vt-shape param))))))
                   (setf (gethash key-v registry)
			 (vt-+ (vt-scale v b2)
                               (vt-scale (vt-square g)
					 (- 1.0d0 b2)))))
		 ;; 解耦更新
		 (let* ((m-hat (vt-scale (gethash key-m registry)
					 bc1))
			(v-hat (vt-scale (gethash key-v registry)
					 bc2))
			(denom (vt-+ (vt-map #'sqrt v-hat)
                                     eps))
			(update (vt-+ (vt-/ m-hat denom)
                                      (vt-scale param wd)))
			(new-p (vt-- param
                                     (vt-scale update lr))))
                   (funcall setter new-p))))))

(defclass rmsprop (optimizer)
  ((alpha :initarg :alpha
          :initform 0.99d0
          :accessor rmsprop-alpha)
   (eps :initarg :eps
        :initform 1.0d-8
        :accessor rmsprop-eps)
   (centered :initarg :centered
             :initform nil
             :accessor rmsprop-centered-p)
   (momentum :initarg :momentum
             :initform 0.0d0
             :accessor rmsprop-momentum))
  (:documentation "RMSprop 优化器."))

(defun make-rmsprop (&key lr alpha eps centered momentum grad-clip
                       weight-decay)
  (make-instance 'rmsprop
                 :lr (coerce (or lr 1e-3) 'double-float)
                 :alpha (or alpha 0.99d0)
                 :eps (or eps 1.0d-8)
                 :centered centered
                 :momentum (or momentum 0.0d0)
                 :grad-clip (or grad-clip 0.0d0)
                 :weight-decay (or weight-decay 0.0d0)))

(defmethod optimizer-step ((opt rmsprop) param-list grad-list)
  (let* ((lr (optimizer-lr opt))
        (alpha (rmsprop-alpha opt))
        (eps (rmsprop-eps opt))
        (clip (optimizer-grad-clip opt))
        (clip-coeff (global-clip-coeff grad-list clip))
        (mom (rmsprop-momentum opt))
        (wd (optimizer-weight-decay opt))
        (registry (optimizer-state-registry opt)))
    (loop for p in param-list
          for (gname . grad) in grad-list
	  for owner = (first p)
          for name = (second p)
          for param = (third p)
          for setter = (fourth p)
          when (and param grad (layer-trainable-p owner))
            do (let* ((base-key (list owner name))
                      (key-v (list base-key 'v))
                      (key-mg (list base-key 'mg))
                      (key-buf (list base-key 'buf))
                      (g (apply-clip grad clip-coeff))
                      (g-reg (if (> wd 0.0d0)
                                 (vt-+ g (vt-scale param wd))
                                 g))
                      (sq (vt-square g-reg))
                      (v (or (gethash key-v registry)
                             (setf (gethash key-v registry)
				   (vt-zeros (vt-shape param))))))
		 ;; 更新平方梯度移动平均
		 (setf (gethash key-v registry)
                       (vt-+ (vt-scale v alpha)
                             (vt-scale sq
                                       (- 1.0d0 alpha))))
		 ;; Centered 逻辑
		 (when (rmsprop-centered-p opt)
                   (let ((mg (or (gethash key-mg registry)
				 (setf (gethash key-mg registry)
                                       (vt-zeros (vt-shape param))))))
                     (setf (gethash key-mg registry)
                           (vt-+ (vt-scale mg alpha)
				 (vt-scale g-reg
                                           (- 1.0d0 alpha))))))
		 ;; 计算分母
		 (let* ((v-new (gethash key-v registry))
			(denom
			  (if (rmsprop-centered-p opt)
			      (let* ((mg (gethash key-mg registry))
				     (var (vt-map (lambda (x) (max 0.0d0 x))
						  (vt-- v-new (vt-square mg)))))
				(vt-+ (vt-map #'sqrt var) eps))
			      (vt-+ (vt-map #'sqrt v-new) eps))))		   
                   (if (> mom 0.0d0)
                       ;; 有动量
                       (let* ((buf (gethash key-buf registry
                                            (vt-zeros
                                             (vt-shape param))))
                              (new-buf (vt-+ (vt-scale buf mom)
                                             (vt-/ g-reg denom))))
			 (setf (gethash key-buf registry)
                               new-buf)
			 (funcall setter
                                  (vt-- param
					(vt-scale new-buf lr))))
                       ;; 无动量
                       (funcall setter
				(vt-- param
                                      (vt-scale (vt-/ g-reg denom)
						lr)))))))))

(defclass adagrad (optimizer)
  ((eps :initarg :eps
        :initform 1.0d-8
        :accessor adagrad-eps)
   (lr-decay :initarg :lr-decay
             :initform 0.0d0
             :accessor adagrad-lr-decay))
  (:documentation "Adagrad 优化器."))

(defun make-adagrad (&key lr eps lr-decay grad-clip weight-decay)
  (make-instance 'adagrad
                 :lr (coerce (or lr 1e-2) 'double-float)
                 :eps (or eps 1.0d-8)
                 :lr-decay (or lr-decay 0.0d0)
                 :grad-clip (or grad-clip 0.0d0)
                 :weight-decay (or weight-decay 0.0d0)))

(defmethod optimizer-step ((opt adagrad) param-list grad-list)
  (let* ((lr (optimizer-lr opt))
         (eps (adagrad-eps opt))
         (clip (optimizer-grad-clip opt))
         (clip-coeff (global-clip-coeff grad-list clip))
         (wd (optimizer-weight-decay opt))
         (tt (incf (optimizer-step-count opt)))
         (registry (optimizer-state-registry opt)))
    (loop for p in param-list
          for (gname . grad) in grad-list
	  for owner = (first p)
          for name = (second p)
          for param = (third p)
          for setter = (fourth p)
          when (and param grad (layer-trainable-p owner))
            do (let* ((base-key (list owner name))
                      (key-v (list base-key 'v))
                      (g (apply-clip grad clip-coeff))
                      (g-reg (if (> wd 0.0d0)
                                 (vt-+ g (vt-scale param wd))
                                 g))
                      (eff-lr (/ lr (+ 1.0d0
                                       (* (adagrad-lr-decay opt)
					  tt)))))
		 ;; 累积平方梯度
		 (let ((v (or (gethash key-v registry)
                              (setf (gethash key-v registry)
				    (vt-zeros (vt-shape param))))))
                   (setf (gethash key-v registry)
			 (vt-+ v (vt-square g-reg))))
		 ;; 更新参数
		 (let* ((v (gethash key-v registry))
			(new-p (vt-- param
                                     (vt-scale (vt-/ g-reg
                                                     (vt-+
                                                      (vt-map #'sqrt v)
                                                      eps))
                                               eff-lr))))
                   (funcall setter new-p))))))
