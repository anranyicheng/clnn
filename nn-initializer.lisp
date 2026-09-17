(in-package #:nn)

(defun %%fan-in (shape fan-in &key (layout :in-out))
  "计算 fan-in。

   FAN-IN 非 nil 时直接返回 (调用方已显式指定)。

   否则按 shape 推断:
     1D          -> 1                (bias 向量)
     2D          -> 按 LAYOUT 解释:
                       :in-out  -> shape = (in,  out)   [PyTorch Linear]
                       :out-in  -> shape = (out, in)    [PyTorch Conv2d 展平]
     ND (rank>=3)-> 视为卷积 (out, in, k1, k2, ...), fan-in = in * k1 * k2 * ..."
  (or fan-in
      (let ((rank (length shape)))
        (cond
          ((= rank 1) 1)
          ((= rank 2)
           (ecase layout
             (:in-out (first shape))
             (:out-in (second shape))))
          (t (reduce #'* (rest shape)))))))

(defun %%fan-out (shape fan-out &key (layout :in-out))
  "计算 fan-out。语义与 %%fan-in 对称:
     1D          -> 向量长度
     2D          -> LAYOUT 解释的另一维
     ND          -> out * k1 * k2 * ..."
  (or fan-out
      (let ((rank (length shape)))
        (cond
          ((= rank 1) (first shape))
          ((= rank 2)
           (ecase layout
             (:in-out (second shape))
             (:out-in (first shape))))
          (t (let ((k-prod 1))
               (loop for i from 2 below rank
                     do (setf k-prod (* k-prod (nth i shape))))
               (* (first shape) k-prod)))))))

(defclass he-normal (initializer) ()
  (:documentation "He Normal 初始化: N(0, sqrt(2/fan_in))"))

(defun make-he-normal ()
  (make-instance 'he-normal :name "he-normal"))

(defmethod init-weight ((init he-normal) shape
                        &key fan-in fan-out (layout :in-out))
  (declare (ignore fan-out init))
  (let* ((fi (%%fan-in shape fan-in :layout layout))
         (std (sqrt (/ 2.0d0 fi))))
    (vt-scale (vt-random-normal shape) std)))

(defclass he-uniform (initializer) ())

(defun make-he-uniform ()
  (make-instance 'he-uniform :name "he-uniform"))

(defmethod init-weight ((init he-uniform) shape
                        &key fan-in fan-out (layout :in-out))
  (declare (ignore fan-out init))
  (let* ((fi (%%fan-in shape fan-in :layout layout))
         (bound (sqrt (/ 6.0d0 fi))))
    (vt-random-uniform shape :low (- bound) :high bound)))

(defclass xavier-normal (initializer) ())

(defun make-xavier-normal ()
  (make-instance 'xavier-normal :name "xavier-normal"))

(defmethod init-weight ((init xavier-normal) shape
                        &key fan-in fan-out (layout :in-out))
  (declare (ignore init))
  (let* ((fi (%%fan-in  shape fan-in  :layout layout))
         (fo (%%fan-out shape fan-out :layout layout))
         (std (sqrt (/ 2.0d0 (+ fi fo)))))
    (vt-scale (vt-random-normal shape) std)))

(defclass xavier-uniform (initializer) ())

(defun make-xavier-uniform ()
  (make-instance 'xavier-uniform :name "xavier-uniform"))

(defmethod init-weight ((init xavier-uniform) shape
                        &key fan-in fan-out (layout :in-out))
  (declare (ignore init))
  (let* ((fi (%%fan-in  shape fan-in  :layout layout))
         (fo (%%fan-out shape fan-out :layout layout))
         (bound (sqrt (/ 6.0d0 (+ fi fo)))))
    (vt-random-uniform shape :low (- bound) :high bound)))

(defclass orthogonal-init (initializer)
  ((gain :initarg :gain :initform 1.0d0 :reader orth-gain))
  (:documentation "正交初始化."))

(defun make-orthogonal-init (&key gain)
  (make-instance 'orthogonal-init :gain (or gain 1.0d0)))

(defmethod init-weight ((init orthogonal-init) shape &key fan-in fan-out
						       (layout :in-out))
  "生成一个半正交矩阵并乘以增益 GAIN。
   维度约定（与 PyTorch torch.nn.init.orthogonal_ 一致）：
     - rows >= cols : 保证列的 orthonormality（||Q^T Q - I|| ≈ 0）
     - rows <  cols : 保证行的 orthonormality（||Q Q^T - I|| ≈ 0）
   实现策略：
     - rows >= cols 时直接在 (rows, cols) 上对列做 Gram–Schmidt；
     - rows <  cols 时先转置为 (cols, rows)，对列做 Gram–Schmidt，
       再转置回 (rows, cols)——此时原本的正交列变成了正交的行。
   这样任意 (rows, cols) 都能得到正确的半正交矩阵。"
  (declare (ignore fan-in fan-out layout))
  (let* ((rows (first shape))
         (cols (second shape))
         ;; 当需要行正交时，把矩阵转置后再处理
         (transposed-p (< rows cols))
         ;; 待正交化矩阵的形状 (m-rows, m-cols)，对它的 m-cols 列做 GS
         (m-rows (if transposed-p cols rows))
         (m-cols (if transposed-p rows cols))
         (gain (orth-gain init))
         ;; 1. 正态分布生成初始矩阵（此时它就是待 GS 的矩阵）
         (q-vt (vt-random-normal (list m-rows m-cols)))
         ;; 2. 抽取底层连续一维数组，避免二维 aref 开销
         (q-data (vt-data q-vt)))

    (declare (type (simple-array double-float (*)) q-data)
             (type fixnum rows cols m-rows m-cols))

    ;; Gram–Schmidt：对 m-rows × m-cols 矩阵的全部 m-cols 列正交化
    ;; （m-cols = min(rows, cols)，保证能全部正交化）
    (dotimes (j m-cols)
      (let ((v (make-array m-rows :element-type 'double-float
                                    :initial-element 0.0d0)))
        ;; 提取当前列 (一维索引: i * m-cols + j)
        (dotimes (i m-rows)
          (setf (aref v i) (aref q-data (+ (* i m-cols) j))))

        ;; 减去在前 j 列上的投影
        (dotimes (k j)
          (let ((dot-prod 0.0d0))
            (dotimes (i m-rows)
              (incf dot-prod
                    (* (aref v i) (aref q-data (+ (* i m-cols) k)))))
            (dotimes (i m-rows)
              (decf (aref v i)
                    (* dot-prod (aref q-data (+ (* i m-cols) k)))))))

        ;; 归一化（防止零除）
        (let ((norm-sq 0.0d0))
          (dotimes (i m-rows)
            (incf norm-sq (* (aref v i) (aref v i))))
          (let ((norm (sqrt norm-sq)))
            (when (> norm 1.0d-12)
              (let ((inv-norm (/ 1.0d0 norm)))
                (dotimes (i m-rows)
                  (setf (aref q-data (+ (* i m-cols) j))
                        (* (aref v i) inv-norm)))))))))

    ;; 3. 需要行正交时转置回来，再乘以增益
    (vt-scale (if transposed-p
                  (vt-transpose q-vt)
                  q-vt)
              gain)))

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
  ((mode :initarg :mode
         :initform :fan-in
         :accessor kaiming-mode)
   (nonlinearity :initarg :nonlinearity
                 :initform :relu
                 :accessor kaiming-nonlinearity)
   (negative-slope :initarg :negative-slope
                   :initform 0.01d0
                   :accessor kaiming-negative-slope
                   :type double-float))
  (:documentation "Kaiming Normal 初始化 (He et al. 2015).
     mode            : :fan-in / :fan-out / :fan-avg
     nonlinearity    : :relu / :leaky-relu / :tanh / :sigmoid / :linear
     negative-slope  : LeakyReLU 的负半轴斜率 a (默认 0.01),
                       仅当 nonlinearity = :leaky-relu 时参与 gain 计算。
     gain 公式 (He 2015):
       :relu        gain = sqrt(2)
       :leaky-relu  gain = sqrt(2 / (1 + a^2))    ; a = negative-slope
       :tanh        gain = sqrt(5/3)
       :sigmoid     gain = 1
       :linear      gain = 1
     std = gain / sqrt(fan)"))


(defun make-kaiming-normal (&key mode nonlinearity negative-slope)
  (make-instance 'kaiming-normal
    :mode (or mode :fan-in)
    :nonlinearity (or nonlinearity :relu)
    :negative-slope (or negative-slope 0.01d0)))

(defmethod init-weight ((init kaiming-normal) shape
                        &key fan-in fan-out (layout :in-out))
  (let* ((fi (%%fan-in  shape fan-in  :layout layout))
         (fo (%%fan-out shape fan-out :layout layout))
         (fan (ecase (kaiming-mode init)
                ((:fan-in  fan-in)  fi)
                ((:fan-out fan-out) fo)
                ((:fan-avg fan-avg) (/ (+ fi fo) 2.0d0))))
         (gain (ecase (kaiming-nonlinearity init)
                 ((:relu relu)         (sqrt 2.0d0))
                 ((:tanh tanh)         (sqrt (/ 5.0d0 3.0d0)))
                 ((:leaky-relu leaky-relu)
                  (let* ((a (kaiming-negative-slope init))
                         (a2 (* a a)))
                    (sqrt (/ 2.0d0 (+ 1.0d0 a2)))))
                 ((:sigmoid sigmoid)   1.0d0)
                 ((:linear linear)     1.0d0)))
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
                        shape &key fan-in fan-out (layout :in-out))
  (declare (ignore fan-in fan-out layout))
  (let* ((s (trunc-std init))
         (m (trunc-mean init))
         (clipped (vt-clip (vt-random-normal shape)
                           -2.0d0 2.0d0))
         (scaled (vt-scale clipped s)))
    (vt-+ scaled m)))
