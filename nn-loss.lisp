(in-package #:nn)

(defclass mse-loss (loss) ()
  (:documentation "均方误差: L = mean((pred - target)^2)"))

(defun make-mse-loss (&key reduction (name "mse"))
  (make-instance 'mse-loss
		 :reduction (or reduction :mean) :name name))

(defmethod compute-loss ((l mse-loss) predicted target)
  (let ((sq-diff (vt-square (vt-- predicted target))))
    (ecase (loss-reduction l)
      (:mean (vt-mean sq-diff))
      (:sum (vt-sum sq-diff))
      (:none sq-diff))))

(defmethod compute-loss-gradient ((l mse-loss) predicted target)
  "dL/dpred = 2 * (pred - target) / N"
  (let* ((diff (vt-- predicted target))
         (n (reduce #'* (vt-shape predicted))))
    (ecase (loss-reduction l)
      (:mean (vt-scale diff (/ 2.0d0 n)))
      (:sum (vt-scale diff 2.0d0))
      (:none (vt-scale diff 2.0d0)))))


(defclass bce-loss (loss)
  ((eps :initarg :eps
	:initform 1.0d-7
	:accessor bce-eps))
  (:documentation "二元交叉熵."))

(defun make-bce-loss (&key reduction eps (name "bce"))
  (make-instance 'bce-loss
		 :reduction (or reduction :mean)
		 :eps (or eps 1.0d-7) :name name))

(defmethod compute-loss ((l bce-loss) predicted target)
  (let* ((eps (bce-eps l))
         (p (vt-clip predicted eps (- 1.0d0 eps)))
         (loss (vt-+ (vt-* (vt-- 0.0d0 target) (vt-log p))
                     (vt-* (vt-- 1.0d0 target)
                           (vt-log (vt-- 1.0d0 p))))))
    (ecase (loss-reduction l)
      (:mean (vt-mean loss))
      (:sum (vt-sum loss))
      (:none loss))))

(defmethod compute-loss-gradient ((l bce-loss) predicted target)
  (let* ((eps (bce-eps l))
         (p (vt-clip predicted eps (- 1.0d0 eps)))
         (n (reduce #'* (vt-shape predicted)))
         (grad (vt-/ (vt-- p target)
                     (vt-* p (vt-- 1.0d0 p)))))
    (ecase (loss-reduction l)
      (:mean (vt-scale grad (/ 1.0d0 n)))
      (:sum grad)
      (:none grad))))

(defclass ce-loss (loss)
  ((eps :initarg :eps
	:initform 1.0d-7
	:accessor ce-eps)
   (label-smoothing :initarg :label-smoothing
		    :initform 0.0d0
		    :accessor ce-label-smoothing))
  (:documentation "多分类CE. PREDICTED: log-probs. TARGET: 整数"))

(defun make-ce-loss (&key reduction eps label-smoothing (name "ce"))
  (make-instance 'ce-loss
		 :reduction (or reduction :mean)
		 :eps (or eps 1.0d-7)
		 :label-smoothing (or label-smoothing 0.0d0)
		 :name name))

(defmethod compute-loss ((l ce-loss) predicted target)
  (let* ((eps (ce-eps l))
         (smoothing (ce-label-smoothing l))
         (shape (vt-shape predicted))
         (batch (first shape))
         (n-classes (second shape))
         ;; 数值稳定性裁剪
         (pred-clipped (vt-clip predicted eps 1.0d0))
         ;; 构造 one-hot 目标 (batch, n-classes)
         (eye (vt-eye n-classes :value 1.0d0 :type 'double-float))
         (target-flat (if (= (length (vt-shape target)) 1)
                          target
                          (vt-flatten target)))   ; 确保是一维
         (one-hot (vt-reshape (vt-take eye target-flat :axis 0)
                              (list batch n-classes)))
         ;; 标签平滑处理
         (target-smoothed
           (if (> smoothing 0.0d0)
               (vt-+ (vt-scale one-hot (- 1.0d0 smoothing))
                     (vt-scale (vt-ones (list batch n-classes))
                               (/ smoothing n-classes 1.0d0)))
               one-hot))
         ;; 逐样本损失 = - sum(target_smoothed * log_pred, axis=-1)
         (per-sample (vt-scale
                      (vt-sum (vt-* target-smoothed pred-clipped)
                              :axis -1)
                      -1.0d0)))
    (ecase (loss-reduction l)
      (:mean (vt-mean per-sample))
      (:sum  (vt-sum per-sample))
      (:none per-sample))))

(defmethod compute-loss-gradient ((l ce-loss) predicted target)
  (let* ((smoothing (ce-label-smoothing l))
         (shape (vt-shape predicted))
         (batch (first shape))
         (n-classes (second shape))
         ;; 计算 softmax 概率（predicted 已经是 log-probs，故 exp 即可）
         (probs (vt-exp predicted))
         ;; 构造 one-hot
         (eye (vt-eye n-classes :value 1.0d0 :type 'double-float))
         (target-flat (if (= (length (vt-shape target)) 1)
                          target
                          (vt-flatten target)))
         (one-hot (vt-reshape (vt-take eye target-flat :axis 0)
                              (list batch n-classes)))
         ;; 标签平滑后的目标
         (target-smoothed
           (if (> smoothing 0.0d0)
               (vt-+ (vt-scale one-hot (- 1.0d0 smoothing))
                     (vt-scale (vt-ones (list batch n-classes))
                               (/ smoothing n-classes 1.0d0)))
               one-hot))
         ;; 梯度 = probs - target_smoothed
         (grad (vt-- probs target-smoothed)))
    ;; 根据 reduction 缩放梯度
    (ecase (loss-reduction l)
      (:mean (vt-scale grad (/ 1.0d0 batch)))
      (:sum grad)
      (:none grad))))

(defclass huber-loss (loss)
  ((delta :initarg :delta
	  :initform 1.0d0
	  :accessor huber-delta))
  (:documentation "Huber 损失."))

(defun make-huber-loss (&key delta reduction (name "huber"))
  (make-instance 'huber-loss
		 :delta (or delta 1.0d0)
		 :reduction (or reduction :mean) :name name))

(defmethod compute-loss ((l huber-loss) predicted target)
  (let* ((diff (vt-- predicted target))
         (abs-diff (vt-abs diff))
         (delta (huber-delta l))
         (loss (vt-map
                (lambda (ad)
                  (if (<= ad delta)
                      (* 0.5d0 ad ad)
                      (* delta (- ad (* 0.5d0 delta)))))
                abs-diff)))
    (ecase (loss-reduction l)
      (:mean (vt-mean loss))
      (:sum (vt-sum loss))
      (:none loss))))

(defmethod compute-loss-gradient ((l huber-loss) predicted target)
  (let* ((diff (vt-- predicted target))
         (delta (huber-delta l))
         (n (reduce #'* (vt-shape predicted)))
         (grad (vt-map
                (lambda (d)
                  (if (<= (abs d) delta)
                      d
                      (* delta (signum d))))
                diff)))
    (ecase (loss-reduction l)
      (:mean (vt-scale grad (/ 1.0d0 n)))
      (:sum grad)
      (:none grad))))


(defclass kl-divergence-loss (loss)
  ((eps :initarg :eps
	:initform 1.0d-7
	:accessor kl-eps))
  (:documentation "KL散度: KL(P||Q). 输入为 log Q."))

(defun make-kl-divergence-loss (&key reduction eps
                                  (name "kl-div"))
  (make-instance 'kl-divergence-loss
		 :reduction (or reduction :mean)
		 :eps (or eps 1.0d-7) :name name))

(defmethod compute-loss ((l kl-divergence-loss) predicted target)
  (let* ((eps (kl-eps l))
         (q (vt-clip (vt-exp predicted) eps (- 1.0d0 eps)))
         (kl (vt-- (vt-* target
                         (vt-log (vt-clip target eps
                                          (- 1.0d0 eps))))
                   (vt-* target (vt-log q)))))
    (ecase (loss-reduction l)
      (:mean (vt-mean (vt-sum-axis kl -1)))
      (:sum (vt-sum (vt-sum-axis kl -1)))
      (:none (vt-sum-axis kl -1)))))

(defmethod compute-loss-gradient
    ((l kl-divergence-loss) predicted target)
  (let* ((grad (vt-scale target -1.0d0))
         (batch (first (vt-shape predicted))))
    (ecase (loss-reduction l)
      (:mean (vt-scale grad (/ 1.0d0  batch)))
      (:sum grad)
      (:none grad))))

(defclass cosine-similarity-loss (loss)
  ()
  (:documentation "余弦相似度: 1 - cos(x, y)"))

(defun make-cosine-similarity-loss (&key reduction
                                      (name "cosine"))
  (make-instance 'cosine-similarity-loss
		 :reduction (or reduction :mean) :name name))

(defmethod compute-loss
    ((l cosine-similarity-loss) predicted target)
  (let* ((dot (vt-sum-axis (vt-* predicted target) -1))
         (norm-p (vt-map #'sqrt
			 (vt-+ (vt-sum-axis
				(vt-square predicted) -1)
                               1.0d-8)))
         (norm-t (vt-map #'sqrt
			 (vt-+ (vt-sum-axis
				(vt-square target) -1)
                               1.0d-8)))
         (cos-sim (vt-mean
                   (vt-/ dot (vt-* norm-p norm-t)))))
    (vt-- 1.0d0 cos-sim)))

(defmethod compute-loss-gradient
    ((l cosine-similarity-loss) predicted target)
  (let* ((batch (first (vt-shape predicted)))
         (safe-norm-p
           (vt-map #'sqrt
                   (vt-+ (vt-sum-axis
                          (vt-square predicted) -1)
                         1.0d-8)))
         (safe-norm-t
           (vt-map #'sqrt
                   (vt-+ (vt-sum-axis
                          (vt-square target) -1)
                         1.0d-8)))
         (dot (vt-sum-axis (vt-* predicted target) -1))
         (cos-sim (vt-/ dot (vt-* safe-norm-p safe-norm-t)))
         ;; 转为列向量以便广播 (batch, 1)
         (cos-vec (vt-reshape cos-sim (list batch 1)))
         (norm-p-vec (vt-reshape safe-norm-p
                                 (list batch 1)))
         (grad (vt-- (vt-* (vt-scale predicted cos-vec)
                           (vt-scale norm-p-vec -1.0d0))
                     (vt-/ target norm-p-vec))))
    (ecase (loss-reduction l)
      (:mean (vt-scale grad (/ 1.0d0 batch)))
      (:sum grad)
      (:none grad))))

(defclass smooth-l1-loss (loss)
  ((beta :initarg :beta
	 :initform 1.0d0
	 :accessor smooth-l1-beta))
  (:documentation "Smooth L1 Loss."))

(defun make-smooth-l1-loss (&key beta reduction
                              (name "smooth-l1"))
  (make-instance 'smooth-l1-loss
		 :beta (or beta 1.0d0)
		 :reduction (or reduction :mean) :name name))

(defmethod compute-loss
    ((l smooth-l1-loss) predicted target)
  (let* ((diff (vt-- predicted target))
         (beta (smooth-l1-beta l))
         (abs-d (vt-abs diff))
         (loss (vt-map
                (lambda (ad)
                  (if (< ad beta)
                      (* 0.5d0 (/ (* ad ad) beta))
                      (- ad (* 0.5d0 beta))))
                abs-d)))
    (ecase (loss-reduction l)
      (:mean (vt-mean loss))
      (:sum (vt-sum loss))
      (:none loss))))

(defmethod compute-loss-gradient
    ((l smooth-l1-loss) predicted target)
  (let* ((diff (vt-- predicted target))
         (beta (smooth-l1-beta l))
         (n (reduce #'* (vt-shape predicted)))
         (grad (vt-map
                (lambda (d)
                  (if (< (abs d) beta)
                      (/ d beta)
                      (signum d)))
                diff)))
    (ecase (loss-reduction l)
      (:mean (vt-scale grad (/ 1.0d0 n)))
      (:sum grad)
      (:none grad))))

(defclass cross-entropy-loss (layer)
  ((probs-cache :accessor ce-probs-cache 
                :initform nil 
                :initarg :probs-cache)
   (targets-cache :accessor ce-targets-cache 
                  :initform nil 
                  :initarg :targets-cache)
   (batch-size :accessor ce-batch-size 
               :initform nil 
               :initarg :batch-size))
  (:documentation "结合 Softmax 的交叉熵损失层，输入为 未归一化得分 和 整数标签"))
(defun make-cross-entropy-loss ()
  "创建一个结合了 Softmax 的交叉熵损失层实例."
  (make-instance 'cross-entropy-loss))

(defmethod forward ((l cross-entropy-loss) inputs)
  (let* ((logits (first inputs))
         (targets (second inputs)) ;; 现在必须是 vt 张量！
         (shape (vt-shape logits))
         (batch-size (first shape))
         (inv-batch (/ 1.0d0 batch-size))
         (max-val (vt-amax logits :axis 1 :keepdims t))
         (shifted (vt-- logits max-val))
         (exp-s (vt-exp shifted))
         (sum-exp (vt-sum exp-s :axis 1 :keepdims t))
         (log-sum-exp (vt-log sum-exp))
         (probs (vt-/ exp-s sum-exp))
         (losses (vt-zeros (list batch-size 1))))
    
    ;; 全部基于 vt-ref 访问
    (loop for i fixnum from 0 below batch-size do
      (let* ;; 假设 targets 是 1D 张量，用 vt-ref 取出，转为 fixnum 作为列索引
          ((target-idx (the fixnum (truncate (vt-ref targets i)))) 
           (target-logit (the double-float (vt-ref shifted i target-idx)))
           (lse-val (the double-float (vt-ref log-sum-exp i 0)))
           (sample-loss (the double-float (- lse-val target-logit))))
        (setf (vt-ref losses i 0) sample-loss)))
    
    (psetf (ce-probs-cache l) probs
           (ce-targets-cache l) targets
           (ce-batch-size l) batch-size)
    
    (vt-scale losses inv-batch)))

(defmethod backward ((l cross-entropy-loss) upstream-grad)
  (declare (ignore upstream-grad))
  (let* ((probs (ce-probs-cache l))
         (targets (ce-targets-cache l))
         (batch-size (ce-batch-size l))
         (inv-batch (/ 1.0d0 batch-size))
         (grad (vt-copy probs)))
    
    ;; 全部基于 vt-ref 访问
    (loop for i fixnum from 0 below batch-size do
      (let ((target-idx (the fixnum (truncate (vt-ref targets i)))))
        (decf (vt-ref grad i target-idx) 1.0d0)))
    
    (vt-scale grad inv-batch)))

