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
         (n (coerce (reduce #'* (vt-shape predicted))
                   'double-float)))
    (ecase (loss-reduction l)
      (:mean (vt-scale diff (/ 2.0d0 n)))
      (:sum (vt-scale diff 2.0d0))
      (:none (vt-scale diff 2.0d0)))))


(defclass bce-loss (loss)
  ((eps :initarg :eps :initform 1.0d-7 :accessor bce-eps))
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
         (n (coerce (reduce #'* (vt-shape predicted))
                   'double-float))
         (grad (vt-/ (vt-- p target)
                     (vt-* p (vt-- 1.0d0 p)))))
    (ecase (loss-reduction l)
      (:mean (vt-scale grad (/ 1.0d0 n)))
      (:sum grad)
      (:none grad))))

(defclass ce-loss (loss)
  ((eps :initarg :eps :initform 1.0d-7 :accessor ce-eps)
   (label-smoothing :initarg :label-smoothing
    :initform 0.0d0 :accessor ce-label-smoothing))
  (:documentation "多分类CE. PREDICTED: log-probs. TARGET: 整数"))

(defun make-ce-loss (&key reduction eps label-smoothing (name "ce"))
  (make-instance 'ce-loss
    :reduction (or reduction :mean)
    :eps (or eps 1.0d-7)
    :label-smoothing (or label-smoothing 0.0d0)
    :name name))

(defmethod compute-loss ((l ce-loss) predicted target)
  (let* ((batch (first (vt-shape predicted)))
         (smoothing (ce-label-smoothing l))
         (nll 0.0d0))
    (dotimes (i batch)
      (let ((label (coerce (vt-ref target (list i)) 'fixnum)))
        (incf nll (- (coerce (vt-ref predicted (list i label))
                             'double-float)))))
    (let ((smooth-loss
            (if (> smoothing 0.0d0)
                (* smoothing
                   (coerce (vt-mean predicted) 'double-float))
                0.0d0)))
      (let ((total (+ (* (- 1.0d0 smoothing) (/ nll batch))
                      smooth-loss)))
        (ecase (loss-reduction l)
          (:mean total)
          (:sum (* total batch))
          (:none
           (let ((per-sample (make-array batch
                                        :element-type
                                        'double-float)))
             (dotimes (i batch)
               (let* ((label (coerce (vt-ref target (list i))
                                     'fixnum))
                      (s-loss (- (coerce
                                  (vt-ref predicted
                                          (list i label))
                                  'double-float))))
                 (when (> smoothing 0.0d0)
                   (incf s-loss
                         (* smoothing
                            (coerce
                             (vt-mean
                              (vt-slice predicted i :all))
                             'double-float))))
                 (setf (aref per-sample i) s-loss)))
             (vt-reshape
              (vt-from-sequence per-sample)
              (list batch)))))))))

(defmethod compute-loss-gradient ((l ce-loss) predicted target)
  (let* ((batch (first (vt-shape predicted)))
         (n-classes (second (vt-shape predicted)))
         (smoothing (ce-label-smoothing l))
         (probs (vt-exp predicted))
         (grad (vt-copy probs)))
    (dotimes (i batch)
      (let ((label (coerce (vt-ref target (list i)) 'fixnum)))
        (let ((old-val (coerce (vt-ref grad (list i label))
                               'double-float)))
          (setf (vt-ref grad (list i label))
                (- old-val 1.0d0)))))
    (when (> smoothing 0.0d0)
      (let ((uniform (/ smoothing
                       (coerce n-classes 'double-float))))
        (setf grad (vt-- grad uniform))
        (dotimes (i batch)
          (let ((label (coerce (vt-ref target (list i))
                               'fixnum)))
            (setf (vt-ref grad (list i label))
                  (+ (vt-ref grad (list i label))
                     smoothing))))))
    (ecase (loss-reduction l)
      (:mean (vt-scale grad (/ 1.0d0 (coerce batch
                                            'double-float))))
      (:sum grad)
      (:none grad))))

(defclass huber-loss (loss)
  ((delta :initarg :delta :initform 1.0d0
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
         (n (coerce (reduce #'* (vt-shape predicted))
                   'double-float))
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
  ((eps :initarg :eps :initform 1.0d-7 :accessor kl-eps))
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
      (:mean (vt-scale grad (/ 1.0d0 (coerce batch
                                            'double-float))))
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
    (- 1.0d0 cos-sim)))

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
      (:mean (vt-scale grad (/ 1.0d0 (coerce batch
                                            'double-float))))
      (:sum grad)
      (:none grad))))

(defclass smooth-l1-loss (loss)
  ((beta :initarg :beta :initform 1.0d0
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
         (n (coerce (reduce #'* (vt-shape predicted))
                   'double-float))
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
