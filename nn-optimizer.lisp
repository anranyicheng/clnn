(in-package #:nn)

(defun clip-gradient (grad clip)
  "计算 L2 Norm 并裁剪梯度."
  (if (> clip 0.0d0)
      (let ((gnorm (sqrt (vt-sum (vt-square grad)))))
        (if (> gnorm clip)
            (vt-scale grad (/ clip gnorm))
            grad))
      grad))


(defclass sgd (optimizer)
  ((momentum :initarg :momentum
	     :initform 0.0d0
	     :accessor sgd-momentum)
   (nesterov :initarg :nesterov
	     :initform nil
	     :accessor sgd-nesterov-p))
  (:documentation
   "随机梯度下降 (可选动量和 Nesterov)."))

(defun make-sgd
    (&key lr momentum nesterov grad-clip weight-decay)
  (make-instance 'sgd
		 :lr (or lr 1e-3)
		 :momentum (or momentum 0.0d0)
		 :nesterov nesterov
		 :grad-clip (or grad-clip 0.0d0)
		 :weight-decay (or weight-decay 0.0d0)))

(defmethod optimizer-step
    ((opt sgd) param-list grad-list)
  (let ((lr (optimizer-lr opt))
        (mu (sgd-momentum opt))
        (wd (optimizer-weight-decay opt))
        (clip (optimizer-grad-clip opt))
        (registry (optimizer-state-registry opt)))
    (loop for p in param-list
          for (gname . grad) in grad-list
          for idx upfrom 0
          for name = (first p)
          for param = (second p)
          for setter = (third p)
          when (and param grad)
            do (let* ((key (format nil "~a-~d" name idx))
                      (g (clip-gradient grad clip))
                      (g-reg (if (> wd 0.0d0)
				 (vt-+ g
                                       (vt-scale param wd))
				 g)))
		 (if (> mu 0.0d0)
                     (let* ((vel (gethash key registry
					  (vt-zeros
                                           (vt-shape param))))
                            (new-vel (vt-+
                                      (vt-scale vel mu)
                                      g-reg)))
                       (setf (gethash key registry) new-vel)
                       (let ((upd
                               (if (sgd-nesterov-p opt)
                                   (vt-+ g-reg
					 (vt-scale new-vel mu))
                                   new-vel)))
			 (funcall setter
                                  (vt-- param
					(vt-scale upd lr)))))
                     (funcall setter
                              (vt-- param
                                    (vt-scale g-reg lr))))))))


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

(defun make-adam
    (&key lr beta1 beta2 eps amsgrad grad-clip weight-decay)
  (make-instance 'adam
		 :lr (or lr 1e-3)
		 :beta1 (or beta1 0.9d0)
		 :beta2 (or beta2 0.999d0)
		 :eps (or eps 1.0d-8)
		 :amsgrad amsgrad
		 :grad-clip (or grad-clip 0.0d0)
		 :weight-decay (or weight-decay 0.0d0)))

(defmethod optimizer-step
    ((opt adam) param-list grad-list)
  (let ((lr (optimizer-lr opt))
        (b1 (adam-beta1 opt))
        (b2 (adam-beta2 opt))
        (eps (adam-eps opt))
        (wd (optimizer-weight-decay opt))
        (clip (optimizer-grad-clip opt))
        (tt (incf (optimizer-step-count opt)))
        (registry (optimizer-state-registry opt)))
    (loop
      for p in param-list
      for (gname . grad) in grad-list
      for idx upfrom 0
      for name = (first p)
      for param = (second p)
      for setter = (third p)
      when (and param grad)
        do (let* ((key-m (format nil "~a-~d-m" name idx))
                  (key-v (format nil "~a-~d-v" name idx))
                  (key-vmax
		    (format nil "~a-~d-vmax" name idx))
                  (g (clip-gradient grad clip))
                  (g-reg (if (> wd 0.0d0)
			     (vt-+ g
                                   (vt-scale param wd))
			     g))
                  (bc1 (/ 1.0d0
                          (- 1.0d0 (expt b1 tt))))
                  (bc2 (/ 1.0d0
                          (- 1.0d0 (expt b2 tt)))))
	     ;; 一阶矩
	     (let ((m (gethash key-m registry
                               (vt-zeros
                                (vt-shape param)))))
               (setf (gethash key-m registry)
		     (vt-+ (vt-scale m b1)
                           (vt-scale g-reg
				     (- 1.0d0 b1)))))
	     ;; 二阶矩
	     (let ((v (gethash key-v registry
                               (vt-zeros
                                (vt-shape param)))))
               (setf (gethash key-v registry)
		     (vt-+ (vt-scale v b2)
                           (vt-scale (vt-square g-reg)
				     (- 1.0d0 b2)))))
	     ;; 更新参数
	     (let* ((m-hat
		      (vt-scale
                       (gethash key-m registry) bc1))
		    (v-hat
		      (vt-scale
                       (gethash key-v registry) bc2))
		    (new-p
                      (if (adam-amsgrad-p opt)
                          (let ((v-max
                                  (gethash key-vmax registry
					   (vt-zeros
					    (vt-shape
					     param)))))
			    (setf (gethash key-vmax registry)
                                  (vt-map #'max
                                          v-max v-hat))
			    (vt-- param
                                  (vt-scale
                                   (vt-/
                                    m-hat
                                    (vt-+
                                     (vt-map
                                      #'sqrt v-max)
                                     eps))
                                   lr)))
                          (vt-- param
                                (vt-scale
                                 (vt-/
                                  m-hat
                                  (vt-+
                                   (vt-map
                                    #'sqrt v-hat)
                                   eps))
                                 lr)))))
               (funcall setter new-p))))))


(defclass adamw (adam) ()
  (:documentation "AdamW: 解耦权重衰减."))

(defun make-adamw
    (&key lr beta1 beta2 eps amsgrad grad-clip weight-decay)
  (make-instance 'adamw
		 :lr (or lr 1e-3)
		 :beta1 (or beta1 0.9d0)
		 :beta2 (or beta2 0.999d0)
		 :eps (or eps 1.0d-8)
		 :amsgrad amsgrad
		 :grad-clip (or grad-clip 0.0d0)
		 :weight-decay (or weight-decay 0.01d0)))

(defmethod optimizer-step
    ((opt adamw) param-list grad-list)
  (let ((lr (optimizer-lr opt))
        (b1 (adam-beta1 opt))
        (b2 (adam-beta2 opt))
        (eps (adam-eps opt))
        (wd (optimizer-weight-decay opt))
        (clip (optimizer-grad-clip opt))
        (tt (incf (optimizer-step-count opt)))
        (registry (optimizer-state-registry opt)))
    (loop for p in param-list
          for (gname . grad) in grad-list
          for idx upfrom 0
          for name = (first p)
          for param = (second p)
          for setter = (third p)
          when (and param grad)
            do (let* ((key-m (format nil "~a-~d-m" name idx))
                      (key-v (format nil "~a-~d-v" name idx))
                      (g (clip-gradient grad clip))
                      (bc1 (/ 1.0d0
                              (- 1.0d0 (expt b1 tt))))
                      (bc2 (/ 1.0d0
                              (- 1.0d0 (expt b2 tt)))))
		 ;; 一阶矩 (无 wd)
		 (let ((m (gethash key-m registry
                                   (vt-zeros
                                    (vt-shape param)))))
                   (setf (gethash key-m registry)
			 (vt-+ (vt-scale m b1)
                               (vt-scale g
					 (- 1.0d0 b1)))))
		 ;; 二阶矩 (无 wd)
		 (let ((v (gethash key-v registry
                                   (vt-zeros
                                    (vt-shape param)))))
                   (setf (gethash key-v registry)
			 (vt-+ (vt-scale v b2)
                               (vt-scale (vt-square g)
					 (- 1.0d0 b2)))))
		 ;; 解耦更新
		 (let* ((m-hat
			  (vt-scale
                           (gethash key-m registry) bc1))
			(v-hat
			  (vt-scale
                           (gethash key-v registry) bc2))
			(denom
                          (vt-+ (vt-map #'sqrt v-hat) eps))
			(update
                          (vt-+ (vt-/ m-hat denom)
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

(defun make-rmsprop
    (&key lr alpha eps centered momentum
       grad-clip weight-decay)
  (make-instance 'rmsprop
		 :lr (or lr 1e-3)
		 :alpha (or alpha 0.99d0)
		 :eps (or eps 1.0d-8)
		 :centered centered
		 :momentum (or momentum 0.0d0)
		 :grad-clip (or grad-clip 0.0d0)
		 :weight-decay (or weight-decay 0.0d0)))

(defmethod optimizer-step
    ((opt rmsprop) param-list grad-list)
  (let ((lr (optimizer-lr opt))
        (alpha (rmsprop-alpha opt))
        (eps (rmsprop-eps opt))
        (clip (optimizer-grad-clip opt))
        (mom (rmsprop-momentum opt))
        (registry (optimizer-state-registry opt)))
    (loop
      for p in param-list
      for (gname . grad) in grad-list
      for idx upfrom 0
      for name = (first p)
      for param = (second p)
      for setter = (third p)
      when (and param grad)
        do (let* ((key-v (format nil "~a-~d-v" name idx))
                  (key-mg (format nil "~a-~d-mg" name idx))
                  (key-buf
		    (format nil "~a-~d-buf" name idx))
                  (g (clip-gradient grad clip))
                  (sq (vt-square g))
                  (v (gethash key-v registry
                              (vt-zeros
                               (vt-shape param)))))
	     ;; 更新平方梯度移动平均
	     (setf (gethash key-v registry)
                   (vt-+ (vt-scale v alpha)
                         (vt-scale sq
                                   (- 1.0d0 alpha))))
	     ;; Centered 逻辑
	     (when (rmsprop-centered-p opt)
               (let ((mg (gethash key-mg registry
                                  (vt-zeros
                                   (vt-shape param)))))
                 (setf (gethash key-mg registry)
                       (vt-+ (vt-scale mg alpha)
			     (vt-scale g
                                       (- 1.0d0 alpha)))))
               ;; 计算分母
               (let* ((v-new
                        (gethash key-v registry))
                      (denom
                        (if (rmsprop-centered-p opt)
			    (let ((mg (gethash key-mg
                                               registry)))
                              (vt-+
                               (vt-map
                                #'sqrt
                                (vt-- v-new
                                      (vt-square mg)))
                               eps))
			    (vt-+ (vt-map #'sqrt v-new)
                                  eps))))
                 (if (> mom 0.0d0)
		     ;; 有动量
		     (let* ((buf
			      (gethash key-buf registry
				       (vt-zeros
					(vt-shape param))))
			    (new-buf
                              (vt-+ (vt-scale buf mom)
				    (vt-/ g denom))))
                       (setf (gethash key-buf registry)
			     new-buf)
                       (funcall setter
                                (vt-- param
                                      (vt-scale new-buf
                                                lr))))
		     ;; 无动量
		     (funcall setter
                              (vt-- param
				    (vt-scale
                                     (vt-/ g denom)
                                     lr))))))))))


(defclass adagrad (optimizer)
  ((eps :initarg :eps
	:initform 1.0d-8
	:accessor adagrad-eps)
   (lr-decay :initarg :lr-decay
	     :initform 0.0d0
	     :accessor adagrad-lr-decay))
  (:documentation "Adagrad 优化器."))

(defun make-adagrad
    (&key lr eps lr-decay grad-clip weight-decay)
  (make-instance 'adagrad
		 :lr (or lr 1e-2)
		 :eps (or eps 1.0d-8)
		 :lr-decay (or lr-decay 0.0d0)
		 :grad-clip (or grad-clip 0.0d0)
		 :weight-decay (or weight-decay 0.0d0)))

(defmethod optimizer-step
    ((opt adagrad) param-list grad-list)
  (let* ((lr (optimizer-lr opt))
         (eps (adagrad-eps opt))
         (clip (optimizer-grad-clip opt))
         (tt (incf (optimizer-step-count opt)))
         (registry (optimizer-state-registry opt)))
    (loop for p in param-list
          for (gname . grad) in grad-list
          for idx upfrom 0
          for name = (first p)
          for param = (second p)
          for setter = (third p)
          when (and param grad)
            do (let* ((key-v (format nil "~a-~d-v" name idx))
                      (g (clip-gradient grad clip))
                      (eff-lr
			(/ lr
                           (+ 1.0d0
                              (* (adagrad-lr-decay opt)
				 tt)))))
		 ;; 累积平方梯度
		 (let ((v (gethash key-v registry
                                   (vt-zeros
                                    (vt-shape param)))))
                   (setf (gethash key-v registry)
			 (vt-+ v (vt-square g))))
		 ;; 更新参数
		 (let* ((v (gethash key-v registry))
			(new-p
                          (vt-- param
				(vt-scale
                                 (vt-/ g
                                       (vt-+
                                        (vt-map #'sqrt v)
                                        eps))
                                 eff-lr))))
                   (funcall setter new-p))))))
