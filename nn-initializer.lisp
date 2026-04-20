(in-package #:nn)

(defclass he-normal (initializer) ()
  (:documentation "He Normal 初始化: N(0, sqrt(2/fan_in))"))

(defun make-he-normal ()
  (make-instance 'he-normal :name "he-normal"))

(defmethod init-weight ((init he-normal) shape
                        &key fan-in fan-out)
  (declare (ignore fan-out init))
  (let* ((fi (or fan-in (reduce #'* (butlast shape))))
         (std (sqrt (/ 2.0d0 fi))))
    (vt-scale (vt-random-normal shape) std)))


(defclass he-uniform (initializer) ())

(defun make-he-uniform ()
  (make-instance 'he-uniform :name "he-uniform"))

(defmethod init-weight ((init he-uniform) shape
                        &key fan-in fan-out)
  (declare (ignore fan-out init))
  (let* ((fi (or fan-in (reduce #'* (butlast shape))))
         (bound (sqrt (/ 6.0d0 fi))))
    (vt-map (lambda (x)
              (declare (ignore x))
              (- (* 2.0d0 (random 1.0d0) bound) bound))
            (vt-zeros shape))))

(defclass xavier-normal (initializer) ())

(defun make-xavier-normal ()
  (make-instance 'xavier-normal :name "xavier-normal"))

(defmethod init-weight ((init xavier-normal) shape
                        &key fan-in fan-out)
  (declare (ignore init))
  (let* ((fi (or fan-in (reduce #'* (butlast shape))))
         (fo (or fan-out (first (last shape))))
         (std (sqrt (/ 2.0d0 (+ fi fo)))))
    (vt-scale (vt-random-normal shape) std)))

(defclass xavier-uniform (initializer) ())

(defun make-xavier-uniform ()
  (make-instance 'xavier-uniform :name "xavier-uniform"))

(defmethod init-weight ((init xavier-uniform) shape
                        &key fan-in fan-out)
  (declare (ignore init))
  (let* ((fi (or fan-in (reduce #'* (butlast shape))))
         (fo (or fan-out (first (last shape))))
         (bound (sqrt (/ 6.0d0 (+ fi fo)))))
    (vt-map (lambda (x)
              (declare (ignore x))
              (- (* 2.0d0 (random 1.0d0) bound) bound))
            (vt-zeros shape))))

(defclass orthogonal-init (initializer)
  ((gain :initarg :gain :initform 1.0d0 :reader orth-gain))
  (:documentation "正交初始化."))

(defun make-orthogonal-init (&key gain)
  (make-instance 'orthogonal-init :gain (or gain 1.0d0)))

(defmethod init-weight ((init orthogonal-init) shape
                        &key fan-in fan-out)
  "正交初始化: 在 rows x cols 上直接 Gram-Schmidt."
  (declare (ignore fan-in fan-out))
  (let* ((rows (first shape))
         (cols (second shape))
         (num-orth (min rows cols))
         (gain (orth-gain init))
         (flat-size (* rows cols))
         (raw-data (make-array flat-size
                               :element-type 'double-float)))
    ;; 填充随机正态数据
    (dotimes (i flat-size)
      (setf (aref raw-data i) (clvt::get-random-normal)))
    (let ((q (make-array (list rows cols)
                         :element-type 'double-float)))
      ;; 将 raw-data 写入 q
      (dotimes (i flat-size)
        (setf (row-major-aref q i) (aref raw-data i)))
      ;; Gram-Schmidt: 对前 num-orth 列正交化
      (dotimes (j num-orth)
        (let ((v (make-array rows
                             :element-type 'double-float)))
          ;; 提取第 j 列
          (dotimes (i rows)
            (setf (aref v i) (aref q i j)))
          ;; 减去在之前所有列上的投影
          (dotimes (k j)
            (let ((dot-prod 0.0d0))
              (dotimes (i rows)
                (incf dot-prod
                      (* (aref v i) (aref q i k))))
              (dotimes (i rows)
                (decf (aref v i)
                      (* dot-prod (aref q i k))))))
          ;; 归一化
          (let ((norm 0.0d0))
            (dotimes (i rows)
              (incf norm (* (aref v i) (aref v i))))
            (setf norm (sqrt norm))
            (when (> norm 1.0d-10)
              (dotimes (i rows)
                (setf (aref v i)
                      (/ (aref v i) norm)))))
          ;; 写回第 j 列
          (dotimes (i rows)
            (setf (aref q i j) (aref v i)))))
      ;; 直接复用底层数组转换为 vt，避免 list/coerce 开销
      (vt-scale
       (vt-reshape
        (vt-from-sequence raw-data) shape)
       gain))))

(defclass zeros-init (initializer) ())

(defun make-zeros-init ()
  (make-instance 'zeros-init :name "zeros"))

(defmethod init-weight ((init zeros-init) shape
                        &key &allow-other-keys)
  (declare (ignore init))
  (vt-zeros shape))

(defclass ones-init (initializer) ())

(defun make-ones-init ()
  (make-instance 'ones-init :name "ones"))

(defmethod init-weight ((init ones-init) shape
                        &key &allow-other-keys)
  (declare (ignore init))
  (vt-ones shape))

(defclass constant-init (initializer)
  ((value :initarg :value :initform 0.0d0 :reader const-value)))

(defun make-constant-init (value)
  (make-instance 'constant-init :value value))

(defmethod init-weight ((init constant-init) shape
                        &key &allow-other-keys)
  (vt-const shape (const-value init)))

(defclass kaiming-normal (initializer)
  ((mode :initarg :mode :initform :fan-in
    :accessor kaiming-mode)
   (nonlinearity :initarg :nonlinearity :initform :relu
    :accessor kaiming-nonlinearity))
  (:documentation "Kaiming Normal 初始化."))

(defun make-kaiming-normal (&key mode nonlinearity)
  (make-instance 'kaiming-normal
    :mode (or mode :fan-in)
    :nonlinearity (or nonlinearity :relu)))

(defmethod init-weight ((init kaiming-normal) shape
                        &key fan-in fan-out)
  (let* ((fan (ecase (kaiming-mode init)
                 ((:fan-in fan-in)
                  (or fan-in (reduce #'* (butlast shape))))
                 ((:fan-out fan-out)
                  (or fan-out (first (last shape))))
                 ((:fan-avg fan-avg)
                  (/ (+ (or fan-in
                            (reduce #'* (butlast shape)))
                        (or fan-out (first (last shape))))
                     2.0d0))))
         (gain (ecase (kaiming-nonlinearity init)
                 ((:relu relu) (sqrt 2.0d0))
                 ((:tanh tanh) (sqrt (/ 5.0d0 3.0d0)))
                 ((:leaky-relu leaky-relu) (sqrt 2.0d0))
                 ((:sigmoid sigmoid) 1.0d0)
                 ((:linear linear) 1.0d0)))
         (std (/ gain (sqrt fan))))
    (vt-scale (vt-random-normal shape) std)))


(defclass truncated-normal-init (initializer)
  ((std :initarg :std :initform 0.02d0 :reader trunc-std)
   (mean :initarg :mean :initform 0.0d0 :reader trunc-mean)))

(defun make-truncated-normal-init (&key std mean)
  (make-instance 'truncated-normal-init
    :std (or std 0.02d0)
    :mean (or mean 0.0d0)))

(defmethod init-weight ((init truncated-normal-init)
                        shape &key fan-in fan-out)
  (declare (ignore fan-in fan-out))
  (let* ((s (trunc-std init))
         (m (trunc-mean init))
         (clipped (vt-clip (vt-random-normal shape)
                           -2.0d0 2.0d0))
         (scaled (vt-scale clipped s)))
    (vt-+ scaled m)))
