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
         (loss (vt-scale
                (vt-+ (vt-* target (vt-log p))
                      (vt-* (vt-- 1.0d0 target)
                            (vt-log (vt-- 1.0d0 p))))
                -1.0d0)))
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
  (:documentation "多分类CE (NLL+LogSoftmax). PREDICTED: 未归一化logits; TARGET: 整数类别 (一维整数张量)."))

(defun make-ce-loss (&key reduction eps label-smoothing (name "ce"))
  (make-instance 'ce-loss
		 :reduction (or reduction :mean)
		 :eps (or eps 1.0d-7)
		 :label-smoothing (or label-smoothing 0.0d0)
		 :name name))

(defmethod compute-loss ((l ce-loss) predicted target)
  "predicted=logits (B,C); target=整数标签 (B,). 内部做log_softmax后取 -log p_{y_i}."
  (let* ((eps (ce-eps l))
         (smoothing (ce-label-smoothing l))
         (shape (vt-shape predicted))
         (batch (first shape))
         (n-classes (second shape))
         (max-val (vt-amax predicted :axis 1 :keepdims t))
         (shifted (vt-- predicted max-val))
         (exp-s (vt-exp shifted))
         (sum-exp (vt-sum exp-s :axis 1 :keepdims t))
         (log-sum-exp (vt-log (vt-+ sum-exp eps)))
         (log-probs (vt-- shifted log-sum-exp))
         (eye (vt-eye n-classes :value 1.0d0 :dtype :float64))
         (target-flat (if (= (length (vt-shape target)) 1)
                          target
                          (vt-flatten target)))
         (one-hot (vt-reshape (vt-take eye target-flat :axis 0)
                              (list batch n-classes)))
         (target-smoothed
           (if (> smoothing 0.0d0)
               (vt-+ (vt-scale one-hot (- 1.0d0 smoothing))
                     (vt-scale (vt-ones (list batch n-classes))
                               (/ smoothing n-classes 1.0d0)))
               one-hot))
         (per-sample (vt-scale
                      (vt-sum (vt-* target-smoothed log-probs) :axis -1)
                      -1.0d0)))
    (ecase (loss-reduction l)
      (:mean (vt-mean per-sample))
      (:sum  (vt-sum per-sample))
      (:none per-sample))))

(defmethod compute-loss-gradient ((l ce-loss) predicted target)
  "dL/dlogits = softmax(logits) - target_smoothed  (:mean 时除以 batch)."
  (let* ((eps (ce-eps l))
         (smoothing (ce-label-smoothing l))
         (shape (vt-shape predicted))
         (batch (first shape))
         (n-classes (second shape))
         (max-val (vt-amax predicted :axis 1 :keepdims t))
         (shifted (vt-- predicted max-val))
         (exp-s (vt-exp shifted))
         (sum-exp (vt-sum exp-s :axis 1 :keepdims t))
         (probs (vt-/ exp-s (vt-+ sum-exp eps)))
         (eye (vt-eye n-classes :value 1.0d0 :dtype :float64))
         (target-flat (if (= (length (vt-shape target)) 1)
                          target
                          (vt-flatten target)))
         (one-hot (vt-reshape (vt-take eye target-flat :axis 0)
                              (list batch n-classes)))
         (target-smoothed
           (if (> smoothing 0.0d0)
               (vt-+ (vt-scale one-hot (- 1.0d0 smoothing))
                     (vt-scale (vt-ones (list batch n-classes))
                               (/ smoothing n-classes 1.0d0)))
               one-hot))
         (grad (vt-- probs target-smoothed)))
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
  "L = reduction(1 - cos(p,t)), cos(p,t) = <p,t> / (|p| * |t| + eps).
   eps 只加在分母除法上做数值稳定（与 PyTorch cosine_similarity 一致），
   不加在 sqrt 里——否则梯度公式会变复杂、小模长下误差大。"
  (let* ((eps 1.0d-8)
         (dot (vt-sum-axis (vt-* predicted target) -1))
         (sq-p (vt-sum-axis (vt-square predicted) -1))
         (sq-t (vt-sum-axis (vt-square target) -1))
         (norm-p (vt-map #'sqrt sq-p))
         (norm-t (vt-map #'sqrt sq-t))
         ;; denom = |p| * |t| + eps，eps 做除法稳定（防除零）
         (denom (vt-+ (vt-* norm-p norm-t) eps))
         (cos-vec (vt-/ dot denom))
         (n (reduce #'* (vt-shape cos-vec))))
    (ecase (loss-reduction l)
      (:mean (vt-- 1.0d0 (vt-mean cos-vec)))
      (:sum  (vt-- (coerce n 'double-float) (vt-sum cos-vec)))
      (:none (vt-- 1.0d0 cos-vec)))))

(defmethod compute-loss-gradient
    ((l cosine-similarity-loss) predicted target)
  "梯度公式与前向严格一致：cos = dot / D, D = |p|*|t| + eps.
   dcos/dp_i = (t_i * D - dot * dD/dp_i) / D^2
   dD/dp_i = (p_i / |p|) * |t|    （因为 d|p|/dp_i = p_i/|p|，|t| 与 p 无关）
   dL/dp_i = -dcos/dp_i   （L = 1 - cos）
   mean reduction 再除以 batch。
   向量形状 (batch,D); norm 量 (batch,) 需 reshape 成 (batch,1) 广播."
  (let* ((shape (vt-shape predicted))
         (batch (first shape))
         (extra-dims (- (length shape) 1))
         (eps 1.0d-8)
         (dot (vt-sum-axis (vt-* predicted target) -1))
         (sq-p (vt-sum-axis (vt-square predicted) -1))
         (sq-t (vt-sum-axis (vt-square target) -1))
         (norm-p (vt-map #'sqrt sq-p))
         (norm-t (vt-map #'sqrt sq-t))
         (denom (vt-+ (vt-* norm-p norm-t) eps))
         (cos-vec (vt-/ dot denom))
         (b1 (append (list batch) (make-list extra-dims :initial-element 1)))
         (cos-r (vt-reshape cos-vec b1))
         (norm-p-r (vt-reshape norm-p b1))
         (norm-t-r (vt-reshape norm-t b1))
         (denom-r (vt-reshape denom b1))
         ;; term1: t_i / D
         (term1 (vt-/ target denom-r))
         ;; term2: (dot / D) * (p_i / |p|) * (|t| / D)
         ;;       = cos(p,t) * (p_i / |p|) * (|t| / D)
         (term2 (vt-* cos-r
                      (vt-/ predicted norm-p-r)
                      (vt-/ norm-t-r denom-r)))
         (dcos (vt-- term1 term2))
         (grad (vt-scale dcos -1.0d0)))
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



