(in-package #:nn)

(defun %%fan-in (shape fan-in)
  "fan-in 计算: 1D 视为 bias 向量 (不应使用, fan-in=1);
   2D (out,in): fan-in = in;
   ND (卷积 out,in/groups,kH,kW): fan-in = in * kH * kW."
  (or fan-in
      (let ((rank (length shape)))
        (cond ((= rank 1) 1)            ; 1D bias 退化情况
              ((= rank 2) (second shape))
              (t (reduce #'* (rest shape)))))))

(defun %%fan-out (shape fan-out)
  (or fan-out
      (let ((rank (length shape)))
        (cond ((= rank 1) (first shape))
              ((= rank 2) (first shape))
              (t (* (first shape)
                    (if (nth 2 shape) (nth 2 shape) 1)
                    (if (nth 3 shape) (nth 3 shape) 1)))))))

(defclass he-normal (initializer) ()
  (:documentation "He Normal 初始化: N(0, sqrt(2/fan_in))"))

(defun make-he-normal ()
  (make-instance 'he-normal :name "he-normal"))

(defmethod init-weight ((init he-normal) shape
                        &key fan-in fan-out)
  (declare (ignore fan-out init))
  (let* ((fi (%%fan-in shape fan-in))
         (std (sqrt (/ 2.0d0 fi))))
    (vt-scale (vt-random-normal shape) std)))


(defclass he-uniform (initializer) ())

(defun make-he-uniform ()
  (make-instance 'he-uniform :name "he-uniform"))

(defmethod init-weight ((init he-uniform) shape
                        &key fan-in fan-out)
  (declare (ignore fan-out init))
  (let* ((fi (%%fan-in shape fan-in))
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
  (let* ((fi (%%fan-in shape fan-in))
         (fo (%%fan-out shape fan-out))
         (std (sqrt (/ 2.0d0 (+ fi fo)))))
    (vt-scale (vt-random-normal shape) std)))

(defclass xavier-uniform (initializer) ())

(defun make-xavier-uniform ()
  (make-instance 'xavier-uniform :name "xavier-uniform"))

(defmethod init-weight ((init xavier-uniform) shape
                        &key fan-in fan-out)
  (declare (ignore init))
  (let* ((fi (%%fan-in shape fan-in))
         (fo (%%fan-out shape fan-out))
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

(defmethod init-weight ((init orthogonal-init) shape &key fan-in fan-out)
  "生成一个行数为 ROWS、列数为 COLS 的近似正交矩阵，并乘以增益 GAIN。
   修复: 使用正态分布生成初始权重，并优化 Gram-Schmidt 的底层内存访问性能。"
  (declare (ignore fan-in fan-out))
  (let* ((rows (first shape))
         (cols (second shape))
         ;; 仅正交化前 MIN(ROWS, COLS) 列，以获得尽可能多的正交方向
         (num-orth (min rows cols))
         (gain (orth-gain init))
         ;; 1. 直接使用正态分布生成张量 (数学上正确的初始化方式)
         (q-vt (vt-random-normal (list rows cols)))
         ;; 2. 提取底层一维连续数组，用于极致优化的正交化计算
         (q-data (vt-data q-vt)))

    (declare (type (simple-array double-float (*)) q-data)
             (type fixnum rows cols num-orth))

    ;; Gram–Schmidt 正交化
    (dotimes (j num-orth)
      (let ((v (make-array rows :element-type 'double-float
				:initial-element 0.0d0)))
        ;; 提取当前列 (使用一维索引计算: i * cols + j，消除二维 aref 开销)
        (dotimes (i rows)
          (setf (aref v i) (aref q-data (+ (* i cols) j))))

        ;; 减去在前 j 列上的投影
        (dotimes (k j)
          (let ((dot-prod 0.0d0))
            ;; 计算点积
            (dotimes (i rows)
              (incf dot-prod
		    (* (aref v i) (aref q-data (+ (* i cols) k)))))
            ;; 减去投影
            (dotimes (i rows)
              (decf (aref v i)
		    (* dot-prod (aref q-data (+ (* i cols) k)))))))

        ;; 归一化（防止零除）
        (let ((norm-sq 0.0d0))
          (dotimes (i rows)
            (incf norm-sq (* (aref v i) (aref v i))))
          (let ((norm (sqrt norm-sq)))
            (when (> norm 1.0d-12)
              (let ((inv-norm (/ 1.0d0 norm)))
                ;; 写回正交化后的列
                (dotimes (i rows)
                  (setf (aref q-data (+ (* i cols) j))
			(* (aref v i) inv-norm)))))))))

    ;; 原地应用增益并返回 (由于 q-vt 数据已被修改，直接 scale 即可)
    (vt-scale q-vt gain)))

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
  (let* ((fi (%%fan-in shape fan-in))
         (fo (%%fan-out shape fan-out))
         (fan (ecase (kaiming-mode init)
                 ((:fan-in fan-in) fi)
                 ((:fan-out fan-out) fo)
                 ((:fan-avg fan-avg) (/ (+ fi fo) 2.0d0))))
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
