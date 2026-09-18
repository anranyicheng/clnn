(in-package #:nn)

(defclass lr-scheduler ()
  ((optimizer :initarg :optimizer
	      :initform nil
	      :reader scheduler-optimizer)
   (name :initarg :name
	 :initform "scheduler"
	 :accessor scheduler-name
	 :type string)
   (last-lr :initarg :last-lr
	    :initform 0.0d0
	    :accessor scheduler-last-lr
	    :type double-float)
   (step-count :initform 0
	       :initarg :step-count
	       :accessor scheduler-step-count))
  (:documentation "学习率调度器基类."))

(defun make-lr-scheduler (optimizer &key (name "scheduler"))
  (let ((s (make-instance 'lr-scheduler
			  :optimizer optimizer :name name)))
    (setf (scheduler-last-lr s) (optimizer-lr optimizer))
    s))

(defgeneric scheduler-step! (scheduler &optional metric)
  (:documentation "更新调度器状态."))

(defgeneric scheduler-get-lr (scheduler)
  (:documentation "获取当前学习率.")
  (:method ((s lr-scheduler))
    (optimizer-lr (scheduler-optimizer s))))

(defclass step-lr (lr-scheduler)
  ((step-size :initarg :step-size
	      :initform 10
	      :reader step-lr-step-size)
   (gamma :initarg :gamma
	  :initform 0.1d0
	  :reader step-lr-gamma))
  (:documentation "每 step_size 个 epoch 将学习率乘以 gamma."))

(defun make-step-lr (optimizer step-size &key gamma)
  (let ((s (make-instance 'step-lr
			  :optimizer optimizer
			  :step-size step-size
			  :gamma (or gamma 0.1d0))))
    (setf (scheduler-last-lr s) (optimizer-lr optimizer))
    s))

(defmethod scheduler-step! ((s step-lr) &optional metric)
  (declare (ignore metric))
  (incf (scheduler-step-count s))
  (when (zerop (mod (scheduler-step-count s)
                    (step-lr-step-size s)))
    (let* ((opt (scheduler-optimizer s))
           (new-lr (* (optimizer-lr opt) (step-lr-gamma s))))
      (setf (optimizer-lr opt) new-lr)
      (setf (scheduler-last-lr s) new-lr))))


(defclass exponential-lr (lr-scheduler)
  ((gamma :initarg :gamma
	  :initform 0.95d0
	  :reader exp-lr-gamma))
  (:documentation "每个 epoch 将学习率乘以 gamma."))

(defun make-exponential-lr (optimizer &key gamma)
  (let ((s (make-instance 'exponential-lr
			  :optimizer optimizer
			  :gamma (or gamma 0.95d0))))
    (setf (scheduler-last-lr s) (optimizer-lr optimizer))
    s))

(defmethod scheduler-step! ((s exponential-lr)
                            &optional metric)
  (declare (ignore metric))
  (incf (scheduler-step-count s))
  (let* ((opt (scheduler-optimizer s))
         (new-lr (* (optimizer-lr opt) (exp-lr-gamma s))))
    (setf (optimizer-lr opt) new-lr)
    (setf (scheduler-last-lr s) new-lr)))


(defclass cosine-annealing-lr (lr-scheduler)
  ((t-max :initarg :t-max
	  :initform 100
	  :reader cos-lr-t-max)
   (eta-min :initarg :eta-min
	    :initform 0.0d0
	    :reader cos-lr-eta-min)
   (base-lr :initarg :base-lr
	    :initform 0.0d0
	    :accessor cos-lr-base-lr))
  (:documentation "余弦退火调度."))

(defun make-cosine-annealing-lr (optimizer t-max
                                 &key eta-min)
  (let ((s (make-instance 'cosine-annealing-lr
			  :optimizer optimizer
			  :t-max t-max
			  :eta-min (or eta-min 0.0d0))))
    (setf (cos-lr-base-lr s) (optimizer-lr optimizer))
    (setf (scheduler-last-lr s) (optimizer-lr optimizer))
    s))

(defmethod scheduler-step! ((s cosine-annealing-lr)
                            &optional metric)
  (declare (ignore metric))
  (incf (scheduler-step-count s))
  (let* ((tt (scheduler-step-count s))
         (t-max (cos-lr-t-max s))
         (base (cos-lr-base-lr s))
         (eta-min (cos-lr-eta-min s))
         (new-lr (+ eta-min
                    (* 0.5d0 (- base eta-min)
                       (+ 1.0d0
                          (cos (/ (* pi tt) t-max)))))))
    (setf (optimizer-lr (scheduler-optimizer s)) new-lr)
    (setf (scheduler-last-lr s) new-lr)))


(defclass reduce-on-plateau (lr-scheduler)
  ((mode :initarg :mode
	 :initform :min
	 :reader rop-mode)
   (factor :initarg :factor
	   :initform 0.1d0
	   :reader rop-factor)
   (patience :initarg :patience
	     :initform 10
	     :reader rop-patience)
   (threshold :initarg :threshold
	      :initform 1.0d-4
	      :reader rop-threshold)
   (min-lr :initarg :min-lr
	   :initform 1.0d-6
	   :reader rop-min-lr)
   (best-metric :initform nil
		:initarg :best-metric
		:accessor rop-best-metric)
   (num-bad-epochs :initform 0
		   :initarg :num-bad-epochs
		   :accessor rop-num-bad-epochs))
  (:documentation "当指标不再改善时降低学习率."))

(defun make-reduce-on-plateau (optimizer
                               &key mode factor
                                 patience threshold min-lr)
  (let ((s (make-instance 'reduce-on-plateau
			  :optimizer optimizer
			  :mode (or mode :min)
			  :factor (or factor 0.1d0)
			  :patience (or patience 10)
			  :threshold (or threshold 1.0d-4)
			  :min-lr (or min-lr 1.0d-6))))
    (setf (scheduler-last-lr s) (optimizer-lr optimizer))
    s))

(defmethod scheduler-step! ((s reduce-on-plateau)
                            &optional metric)
  (unless metric (return-from scheduler-step!))
  (let* ((opt (scheduler-optimizer s))
         (current-lr (optimizer-lr opt))
         (mode (rop-mode s))
         (best (rop-best-metric s)))
    (cond
      ;; 首次记录
      ((null best)
       (setf (rop-best-metric s) metric))
      ;; min 模式改善
      ((and (eq mode :min)
            (< metric (- best (rop-threshold s))))
       (setf (rop-best-metric s) metric)
       (setf (rop-num-bad-epochs s) 0))
      ;; max 模式改善
      ((and (eq mode :max)
            (> metric (+ best (rop-threshold s))))
       (setf (rop-best-metric s) metric)
       (setf (rop-num-bad-epochs s) 0))
      ;; 未改善
      (t
       (incf (rop-num-bad-epochs s))
       (when (>= (rop-num-bad-epochs s)
                 (rop-patience s))
         (let ((new-lr (max (* current-lr (rop-factor s))
                            (rop-min-lr s))))
           (setf (optimizer-lr opt) new-lr)
           (setf (scheduler-last-lr s) new-lr)
           (setf (rop-num-bad-epochs s) 0)))))))


(defclass warmup-cosine-lr (lr-scheduler)
  ((warmup-steps :initarg :warmup-steps
		 :initform nil
		 :reader warmup-warmup-steps)
   (total-steps :initarg :total-steps
		:initform nil
		:reader warmup-total-steps)
   (min-lr :initarg :min-lr
	   :initform 0.0d0
	   :reader warmup-min-lr)
   (base-lr :initarg :base-lr
	    :initform 0.0d0
	    :accessor warmup-base-lr))
  (:documentation "Warmup + Cosine Decay 调度."))

(defun make-warmup-cosine-lr (optimizer warmup-steps
                              total-steps &key min-lr)
  (let ((s (make-instance 'warmup-cosine-lr
			  :optimizer optimizer
			  :warmup-steps warmup-steps
			  :total-steps total-steps
			  :min-lr (or min-lr 0.0d0))))
    (setf (warmup-base-lr s) (optimizer-lr optimizer))
    (setf (scheduler-last-lr s) (optimizer-lr optimizer))
    s))

(defmethod scheduler-step! ((s warmup-cosine-lr) &optional metric)
  (declare (ignore metric))
  (incf (scheduler-step-count s))
  (let* ((step (scheduler-step-count s))
         (warmup (or (warmup-warmup-steps s) 0))   ; 保证有整数
         (total (warmup-total-steps s))
         (base (warmup-base-lr s))
         (min-lr (warmup-min-lr s))
         (new-lr
           (cond
             ;; 阶段1: warmup（仅在 warmup > 0 且 step < warmup 时）
             ((and (> warmup 0) (< step warmup))
              (* base (/ step warmup 1.0d0)))
             ;; 阶段2: Cosine decay（需要 total > warmup）
             ((and (> total warmup) (<= step total))
              (+ min-lr
                 (* 0.5d0 (- base min-lr)
                    (+ 1.0d0
                       (cos (/ (* pi (- step warmup))
                               (- total warmup)))))))
             ;; 阶段3: 保底（训练已结束或无效阶段）
             (t min-lr))))
    (setf (optimizer-lr (scheduler-optimizer s)) new-lr)
    (setf (scheduler-last-lr s) new-lr)))


(defclass one-cycle-lr (lr-scheduler)
  ((max-lr :initarg :max-lr
	   :initform 0.0d0
	   :reader one-cycle-max-lr)
   (total-steps :initarg :total-steps
		:initform 0
		:reader one-cycle-total-steps)
   (pct-start :initarg :pct-start
	      :initform 0.3d0
	      :reader one-cycle-pct-start)
   (div-factor :initarg :div-factor
	       :initform 25.0d0
	       :reader one-cycle-div-factor)
   (final-div-factor :initarg :final-div-factor
		     :initform 1.0d4
		     :reader one-cycle-final-div-factor)
   (base-lr :initarg :base-lr
	    :initform 0.0d0
	    :accessor one-cycle-base-lr))
  (:documentation "1Cycle 策略."))

(defun make-one-cycle-lr (optimizer max-lr total-steps
                          &key pct-start
                            div-factor
                            final-div-factor)
  (let ((s (make-instance 'one-cycle-lr
			  :optimizer optimizer
			  :max-lr max-lr
			  :total-steps total-steps
			  :pct-start (or pct-start 0.3d0)
			  :div-factor (or div-factor 25.0d0)
			  :final-div-factor
			  (or final-div-factor 1.0d4))))
    (setf (one-cycle-base-lr s)
          (/ max-lr (one-cycle-div-factor s)))
    (setf (scheduler-last-lr s) (optimizer-lr optimizer))
    (setf (optimizer-lr optimizer) (one-cycle-base-lr s))
    s))

(defmethod scheduler-step! ((s one-cycle-lr)
                            &optional metric)
  (declare (ignore metric))
  (incf (scheduler-step-count s))
  (let* ((step (scheduler-step-count s))
         (total (one-cycle-total-steps s))
         (max-lr (one-cycle-max-lr s))
         (base (one-cycle-base-lr s))
         (final-div (one-cycle-final-div-factor s))
         (pct-start (one-cycle-pct-start s))
         (start-step (floor (* pct-start total)))
         (final-lr (/ max-lr final-div))
         (new-lr
           (cond
             ((and (> start-step 0) (< step start-step))
              (+ base
                 (* (- max-lr base)
                    (/ step start-step 1.0d0))))
             ;; 下降阶段 (total=start-step 时直接跳到底)
             ((and (> total start-step) (<= step total))
              (let* ((progress
                       (/ (- step start-step)
                          (- total start-step)
			  1.0d0))
                     (cos-val (cos (* pi progress))))
                (+ final-lr
                   (* (- max-lr final-lr)
                      (* 0.5d0 (+ 1.0d0 cos-val))))))
             ;; 超出总步数或退化情况
             (t final-lr))))
    (setf (optimizer-lr (scheduler-optimizer s)) new-lr)
    (setf (scheduler-last-lr s) new-lr)))
