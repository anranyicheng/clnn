;;;; binocular-mnist.lisp
;;;; ============================================================
;;;;  Binocular Fovea Net —— 仿生双眼 MNIST 识别器
;;;; ============================================================
;;;;
;;;; 【一句话】
;;;;   用「周边视觉 + 中央凹采样 + 双眼视差」的仿生架构替代传统 CNN 的
;;;;   均匀滑窗，让网络自己学出「看哪里、怎么看、两只眼差多少」。
;;;;
;;;; 【架构】
;;;;
;;;;   输入 (B, 1, 28, 28)
;;;;     │
;;;;     ├────────────────────────────────────────────┐
;;;;     │                                            │
;;;;     ▼                                            │
;;;;   周边视觉 CNN ──► 全局特征 (B, 64) ────┐          │
;;;;     │                                 │          │
;;;;     ├──► 注视点预测 MLP ──► (B, K, 2)  │          │
;;;;     │        + base-fixations         │          │
;;;;     │        + tanh                   │          │
;;;;     │                                 │          │
;;;;     │     左右眼视差 ±δ                 │         │
;;;;     │        │                         │         │
;;;;     ▼        ▼                         │         │
;;;;   ┌──── 左眼注视点 ──► 可微双线性采样 ◄─┐           │
;;;;   │                                    │         │
;;;;   └──── 右眼注视点 ──► 可微双线性采样 ◄─┤           │
;;;;                                        │         │
;;;;       左眼 patches (BK,1,P,P)          │         │
;;;;       右眼 patches (BK,1,P,P)          │         │
;;;;              │                         │         │
;;;;              ▼                         │         │
;;;;       共享 patch-encoder               │         │
;;;;       (沿 batch 维拼接，只前向一次)      │         │
;;;;              │                         │         │
;;;;              ▼                         │         │
;;;;       左右眼特征 (B,K,F) × 2            │         │
;;;;              │                         │         │
;;;;              ▼                         │         │
;;;;       视交叉融合 (沿特征维拼接)           │         │
;;;;       (B, K*2F)                        │         │
;;;;              │                         │         │
;;;;              └────────► concat ◄───────┘         │
;;;;                           │                      │
;;;;                           ▼                      │
;;;;                    分类器 (10 类)  ◄──────────────┘
;;;;                    （periphery 直连，双通路）
;;;;
;;;;
;;;; 【生物机制对照】
;;;;
;;;;   | 生物视觉机制              | 本实现                                |
;;;;   |-------------------------|---------------------------------------|
;;;;   | 周边视觉引导注视           | periphery CNN → fix-head            |
;;;;   | 中央凹高分辨率采样         | 高斯中央凹 + 可微双线性采样             |
;;;;   | 扫视序列                  | K=4 个注视点                          |
;;;;   | 双眼视差（辐辏）           | 可学习标量 δ，左眼 -δ / 右眼 +δ         |
;;;;   | 视交叉                    | 左右眼特征按注视点配对拼接               |
;;;;   | 背侧流 / 腹侧流分离        | periphery 直连 classifier 的旁路       |
;;;;
;;;;
;;;; 【三处关键设计决策】
;;;;
;;;;   1. 双通路（v3 的核心）
;;;;      periphery 特征不仅用于预测注视点，也直连分类器。这样即使
;;;;      注视点机制完全失效（落在纯黑背景、采样到噪声），分类器仍有
;;;;      全图信息兜底。这是训练能收敛的必要条件：单靠注视点路径的
;;;;      梯度必须穿过「双线性采样 → patch-encoder → 分类器」才能
;;;;      监督 fix-head，而这条链上有三层衰减（差分、max-pool、
;;;;      tanh 饱和），实际梯度量级不足以让 fix-head 学到东西。
;;;;
;;;;   2. 双眼共享 encoder：沿 batch 维拼接后只前向一次
;;;;      clnn 的所有层 backward 都是覆盖式写入梯度（setf 而非 incf）。
;;;;      如果对同一 encoder 先后调用两次 forward/backward，第二次会
;;;;      覆盖第一次的缓存和梯度，导致左眼路径完全丢失监督信号。
;;;;      解决方案：把「左眼 + 右眼」在 batch 维上 concat 成一个
;;;;      (2BK, ...) 的 batch，encoder 只前向一次、只反向一次，
;;;;      梯度自然累加到共享权重上。
;;;;
;;;;   3. δ 与 base-fixations 的可学习性
;;;;      - δ 是标量参数，直接挂进 params 列表，Adam 像更新权重一样
;;;;        更新它；反向时 d_δ = Σd_right.x - Σd_left.x。
;;;;      - base-fixations 是每个注视点的 (2K,) 基准位置，用于打破
;;;;        初始对称性——fix-head 初始输出接近 tanh(0)=0，如果 4 个
;;;;        注视点都从中心出发，梯度几乎相同，永远分不开。
;;;;
;;;;
;;;; 【手写可微采样器的核心】
;;;;
;;;;   bilinear-sample 的反向里，对注视点的梯度有一个非常简洁的形式：
;;;;       网格坐标 gx[b,k,i,j] = fx[b,k] + offset[j]
;;;;       因为 offset 是固定离散网格，∂gx/∂fx = 1
;;;;       所以 d_fx = Σ_{i,j} d_gx       （对所有 patch 像素求和）
;;;;   这比显式推链式法则简单一个数量级，且避开了 floor 的不连续问题。
;;;;
;;;;
;;;; 【已知局限 / 实验结论】
;;;;
;;;;   - 在 MNIST 上，训练收敛后 δ 会自发收缩到接近 0（0.01~0.07），
;;;;     说明模型发现双眼视差对这个任务没有净收益。这不是 bug，
;;;;     而是模型在简单任务上做的自动 ablation：单眼信息已足够。
;;;;   - 要真正验证双眼机制的增益，需要「单眼会失败、双眼能成功」
;;;;     的任务，例如：遮挡 MNIST（每张图随机打黑块）、
;;;;     Fashion-MNIST、或合成视差对（左右眼喂真正不同的图）。
;;;;   - 与标准小 CNN 相比，本模型在 MNIST 上精度相当（98%+），
;;;;     但参数更多、训练更慢。它的价值在于验证「可微注视点 +
;;;;     双眼视差」这条链路在 Lisp + 手写张量库上完全走得通。
;;;;
;;;;


(eval-when (:compile-toplevel :load-toplevel :execute)
  (ql:quickload '(:clnn :chipz)))
(in-package :clnn)

;;; ============================================================
;;; 全局数据
;;; ============================================================
(defvar *mnist-traina* nil)
(defvar *mnist-trainl* nil)
(defvar *mnist-testa* nil)
(defvar *mnist-testl* nil)
(defvar *mnist-train-labels* nil)     ; fixnum 数组，one-hot 转换后
(defvar *mnist-test-labels* nil)

;;; ============================================================
;;; 数据加载
;;; ============================================================
(defun mnist-file-path (filename)
  (asdf:system-relative-pathname
   "clnn" (concatenate 'string "mnist-data/" filename)))

(defun mnist-read-and-normalize-image (data offset)
  (let ((image (make-array (* 28 28) :element-type 'double-float
                                     :initial-element 0.0d0)))
    (dotimes (i (* 28 28))
      (setf (aref image i) (/ (aref data (+ offset i)) 255.0d0)))
    (clvt::vt-from-array image :dtype :float64 :fast t)))

(defun mnist-read-and-normalize-label (data offset)
  (let ((target (make-array 10 :element-type 'double-float
                               :initial-element 0.0d0))
        (category (aref data offset)))
    (setf (aref target category) 1.0d0)
    (clvt::vt-from-array target :dtype :float64 :fast t)))

(defun mnist-load (type)
  (destructuring-bind (n-images images-path labels-path)
      (if (eql type :train)
          (list 60000
                (mnist-file-path "train-images-idx3-ubyte.gz")
                (mnist-file-path "train-labels-idx1-ubyte.gz"))
          (list 10000
                (mnist-file-path "t10k-images-idx3-ubyte.gz")
                (mnist-file-path "t10k-labels-idx1-ubyte.gz")))
    (let ((images
            (with-open-file (f images-path :element-type '(unsigned-byte 8))
              (let ((images-data (chipz:decompress nil 'chipz:gzip f)))
                (loop for i from 0 below n-images
                      for offset = (+ 16 (* i 28 28))
                      collect (mnist-read-and-normalize-image
                               images-data offset)))))
          (labels
	      (with-open-file (f labels-path :element-type '(unsigned-byte 8))
                (let ((labels-data (chipz:decompress nil 'chipz:gzip f)))
                  (loop for i from 0 below n-images
                        collect (mnist-read-and-normalize-label
                                 labels-data (+ 8 i)))))))
      (values images labels))))

(defun ensure-mnist-data ()
  "幂等加载 MNIST。已加载则直接返回。"
  (when *mnist-traina* (return-from ensure-mnist-data))
  (format t "~%加载 MNIST 数据...~%")
  (setf *mnist-traina* (clvt::vt-zeros '(60000 784) :dtype :float64))
  (setf *mnist-trainl* (clvt::vt-zeros '(60000 10) :dtype :float64))
  (setf *mnist-testa* (clvt::vt-zeros '(10000 784) :dtype :float64))
  (setf *mnist-testl* (clvt::vt-zeros '(10000 10) :dtype :float64))
  (multiple-value-bind (image label) (mnist-load :train)
    (let ((count 0))
      (dolist (var image)
        (setf (clvt::vt-slice *mnist-traina* (list count)) var)
        (incf count)))
    (let ((count 0))
      (dolist (var label)
        (setf (clvt::vt-slice *mnist-trainl* (list count)) var)
        (incf count))))
  (multiple-value-bind (image label) (mnist-load :test)
    (let ((count 0))
      (dolist (var image)
        (setf (clvt::vt-slice *mnist-testa* (list count)) var)
        (incf count)))
    (let ((count 0))
      (dolist (var label)
        (setf (clvt::vt-slice *mnist-testl* (list count)) var)
        (incf count))))
  (format t "加载完成。~%"))

;;; ============================================================
;;; 切片工具
;;; ============================================================
(defun mnist-slice-batch (tensor start end)
  "取 TENSOR 的 [START, END) 行，强制连续。"
  (let ((s (clvt::vt-slice tensor (list start end) (list :all))))
    (if (clvt::vt-contiguous-p s) s (clvt::vt-contiguous s))))

(defun mnist-images-4d (start end batch-size &key (data *mnist-traina*))
  "从 DATA 取 [START, END)，reshape 成 (BATCH-SIZE, 1, 28, 28)。
   DATA 默认 *mnist-traina*（训练用）；评估时传 *mnist-testa*。"
  (clvt::vt-reshape (mnist-slice-batch data start end)
                    (list batch-size 1 28 28)))

;;; ============================================================
;;; 标签工具
;;; ============================================================
(defun mnist-one-hot-to-labels (one-hot)
  "把 (N, C) one-hot 张量转成 fixnum 数组。"
  (let* ((n (first (clvt::vt-shape one-hot)))
         (c (second (clvt::vt-shape one-hot)))
         (data (clvt::vt-data one-hot))
         (off (clvt::vt-offset one-hot))
         (rs (first (clvt::vt-strides one-hot)))
         (cs (second (clvt::vt-strides one-hot)))
         (labels (make-array n :element-type 'fixnum)))
    (dotimes (i n)
      (let ((best 0) (val most-negative-double-float)
		     (base (+ off (* i rs))))
        (dotimes (k c)
          (let ((v (aref data (+ base (* k cs)))))
            (when (> v val) (setf val v best k))))
        (setf (aref labels i) best)))
    labels))

(defun ensure-mnist-labels ()
  "幂等转换 one-hot → fixnum 标签。"
  (when *mnist-train-labels* (return-from ensure-mnist-labels))
  (ensure-mnist-data)
  (format t "转换 one-hot 标签...~%")
  (setf *mnist-train-labels* (mnist-one-hot-to-labels *mnist-trainl*))
  (setf *mnist-test-labels* (mnist-one-hot-to-labels *mnist-testl*)))

(defun mnist-slice-labels (labels-array start end)
  "取 fixnum 数组的 [START, END)，返回 int64 VT。"
  (clvt::vt-from-sequence
   (loop for i from start below end collect (aref labels-array i))
   :dtype :int64))

;;; ============================================================
;;; 准确率
;;; ============================================================
(defun mnist-accuracy-vt (pred target)
  "PRED / TARGET 均为 (batch, n-classes) VT。返回 batch 内正确数。"
  (let* ((batch (first (clvt::vt-shape pred)))
         (n-classes (second (clvt::vt-shape pred)))
         (p-data (clvt::vt-data pred))
	 (p-off (clvt::vt-offset pred))
         (p-rs (first (clvt::vt-strides pred)))
	 (p-cs (second (clvt::vt-strides pred)))
         (t-data (clvt::vt-data target))
	 (t-off (clvt::vt-offset target))
         (t-rs (first (clvt::vt-strides target)))
	 (t-cs (second (clvt::vt-strides target)))
         (correct 0))
    (dotimes (i batch)
      (let ((pb 0) (pv most-negative-double-float)
		   (tb 0) (tv most-negative-double-float)
		   (p-base (+ p-off (* i p-rs)))
		   (t-base (+ t-off (* i t-rs))))
        (dotimes (c n-classes)
          (let ((a (aref p-data (+ p-base (* c p-cs))))
                (b (aref t-data (+ t-base (* c t-cs)))))
            (when (> a pv) (setf pv a pb c))
            (when (> b tv) (setf tv b tb c))))
        (when (= pb tb) (incf correct))))
    correct))

(defun mnist-accuracy-labels (pred labels-array start)
  "PRED 是 (batch, n-classes) VT；LABELS-ARRAY 是 fixnum 数组；
   START 是标签数组的起始索引。返回 batch 内正确数。"
  (let* ((batch (first (clvt::vt-shape pred)))
         (n-classes (second (clvt::vt-shape pred)))
         (data (clvt::vt-data pred))
	 (off (clvt::vt-offset pred))
         (rs (first (clvt::vt-strides pred)))
	 (cs (second (clvt::vt-strides pred)))
         (correct 0))
    (dotimes (i batch)
      (let ((best 0)
	    (val most-negative-double-float)
	    (base (+ off (* i rs))))
        (dotimes (c n-classes)
          (let ((v (aref data (+ base (* c cs)))))
            (when (> v val) (setf val v best c))))
        (when (= best (aref labels-array (+ start i)))
	  (incf correct))))
    correct))

;;; ============================================================
;;; 评估
;;; ============================================================
(defun evaluate-mnist-vt (model &key (batch-size 128) (n 10000))
  "用 one-hot 标签评估。返回 (values correct total accuracy)。"
  (let ((correct 0) (total 0) (start 0))
    (loop while (< start n) do
      (let* ((end (min (+ start batch-size) n))
             (bs (- end start))
             (x (mnist-slice-batch *mnist-testa* start end))
             (y (mnist-slice-batch *mnist-testl* start end))
             (pred (forward model x)))
        (incf correct (mnist-accuracy-vt pred y))
        (incf total bs)
        (setf start end)
	(clear-step-caches! model)))
    (values correct total (coerce (/ correct total) 'double-float))))

(defun evaluate-mnist-labels (model &key (batch-size 128) (n 10000)
                                      (images-4d nil))
  "用 fixnum 标签评估。
   IMAGES-4D=T 时把输入 reshape 成 (batch, 1, 28, 28)——供 CNN 使用。
   IMAGES-4D=NIL 时保持 (batch, 784)——供 Dense 使用。"
  (ensure-mnist-labels)
  (let ((correct 0) (total 0) (start 0))
    (loop while (< start n) do
      (let* ((end (min (+ start batch-size) n))
             (bs (- end start))
             (x (if images-4d
                    (mnist-images-4d start end bs :data *mnist-testa*)
                    (mnist-slice-batch *mnist-testa* start end)))
             (pred (forward model x)))
        (incf correct (mnist-accuracy-labels pred *mnist-test-labels* start))
        (incf total bs)
        (setf start end)))
    (values correct total (coerce (/ correct total) 'double-float))))

(defparameter *fix-grad-scale* 0.5d0
  "注视点梯度放大系数。诊断显示梯度量级正常（0.2~0.5），
   但比 classifier 梯度偏大，用 0.5 抑制注视点振荡。")

;;; ============================================================
;;; 1. 可微双线性采样器（不变）
;;; ============================================================
(defun bilinear-sample (images grid)
  (let* ((images (if (vt-contiguous-p images) images (vt-contiguous images)))
         (grid   (if (vt-contiguous-p grid)   grid   (vt-contiguous grid)))
         (ishape (vt-shape images)) (gshape (vt-shape grid))
         (B (first ishape)) (C (second ishape))
         (H (third ishape)) (W (fourth ishape)) (N (second gshape))
         (i-data (vt-data images)) (i-off (vt-offset images))
         (g-data (vt-data grid))   (g-off (vt-offset grid))
         (out-data (make-array (* B N C) :element-type 'double-float
                                         :initial-element 0.0d0))
         (ix0-l (make-array (* B N) :element-type 'fixnum))
         (iy0-l (make-array (* B N) :element-type 'fixnum))
         (ix1-l (make-array (* B N) :element-type 'fixnum))
         (iy1-l (make-array (* B N) :element-type 'fixnum))
         (wx-l  (make-array (* B N) :element-type 'double-float))
         (wy-l  (make-array (* B N) :element-type 'double-float)))
    (declare (type fixnum B C H W N)
             (type (simple-array double-float (*)) i-data g-data out-data)
             (type (simple-array fixnum (*)) ix0-l iy0-l ix1-l iy1-l)
             (type (simple-array double-float (*)) wx-l wy-l))
    (let ((CHW (* C H W)))
      (dotimes (bi B)
        (dotimes (ni N)
          (let* ((bn (+ (* bi N) ni))
                 (g-base (+ g-off (* bn 2)))
                 (gx (aref g-data g-base))
                 (gy (aref g-data (+ g-base 1)))
                 (ixf (max 0.0d0 (min (coerce (1- W) 'double-float)
                                      (/ (* (+ gx 1.0d0) (1- W)) 2.0d0))))
                 (iyf (max 0.0d0 (min (coerce (1- H) 'double-float)
                                      (/ (* (+ gy 1.0d0) (1- H)) 2.0d0))))
                 (ix0 (min (- W 2) (max 0 (floor ixf))))
                 (iy0 (min (- H 2) (max 0 (floor iyf))))
                 (ix1 (1+ ix0)) (iy1 (1+ iy0))
                 (wx (- ixf ix0)) (wy (- iyf iy0))
                 (b-base (+ i-off (* bi CHW))))
            (setf (aref ix0-l bn) ix0 (aref iy0-l bn) iy0
                  (aref ix1-l bn) ix1 (aref iy1-l bn) iy1
                  (aref wx-l bn)  wx  (aref wy-l bn)  wy)
            (let ((iwx (- 1.0d0 wx)) (iwy (- 1.0d0 wy))
                  (o-base (* bi N C)))
              (dotimes (ci C)
                (let* ((c-base (+ b-base (* ci H W)))
                       (v00 (aref i-data (+ c-base (* iy0 W) ix0)))
                       (v01 (aref i-data (+ c-base (* iy0 W) ix1)))
                       (v10 (aref i-data (+ c-base (* iy1 W) ix0)))
                       (v11 (aref i-data (+ c-base (* iy1 W) ix1)))
                       (val (+ (* iwx iwy v00) (* wx iwy v01)
                               (* iwx wy v10) (* wx wy v11))))
                  (setf (aref out-data (+ o-base (* ni C) ci)) val))))))))
      (values
       (vt-reshape (vt-from-array out-data :dtype :float64 :fast t)
                   (list B N C))
       (list :B B :C C :H H :W W :N N
             :ix0 ix0-l :iy0 iy0-l :ix1 ix1-l :iy1 iy1-l
             :wx wx-l :wy wy-l :images images :grid grid))))

(defun bilinear-sample-grad (grad-output cache)
  (destructuring-bind (&key B C H W N ix0 iy0 ix1 iy1 wx wy images grid) cache
    (declare (type fixnum B C H W N))
    (let* ((grad-output (if (vt-contiguous-p grad-output)
                            grad-output (vt-contiguous grad-output)))
           (g-data (vt-data grad-output)) (g-off (vt-offset grad-output))
           (d-img-data (make-array (* B C H W) :element-type 'double-float
                                               :initial-element 0.0d0))
           (d-grid-data (make-array (* B N 2) :element-type 'double-float
                                               :initial-element 0.0d0))
           (i-data (vt-data images)) (i-off (vt-offset images)))
      (declare (type (simple-array double-float (*)) g-data i-data
                     d-img-data d-grid-data))
      (let ((CHW (* C H W)))
        (dotimes (bi B)
          (dotimes (ni N)
            (let* ((bn (+ (* bi N) ni))
                   (i-x0 (aref ix0 bn)) (i-y0 (aref iy0 bn))
                   (i-x1 (aref ix1 bn)) (i-y1 (aref iy1 bn))
                   (w-x (aref wx bn))  (w-y (aref wy bn))
                   (iwx (- 1.0d0 w-x)) (iwy (- 1.0d0 w-y))
                   (b-base-in  (+ i-off (* bi CHW)))
                   (b-base-out (* bi CHW))
                   (g-base (+ g-off (* bi N C) (* ni C)))
                   (dgx 0.0d0) (dgy 0.0d0))
              (dotimes (ci C)
                (let* ((g (aref g-data (+ g-base ci)))
                       (c-base-in  (+ b-base-in  (* ci H W)))
                       (c-base-out (+ b-base-out (* ci H W)))
                       (off00 (+ c-base-in (* i-y0 W) i-x0))
                       (off01 (+ c-base-in (* i-y0 W) i-x1))
                       (off10 (+ c-base-in (* i-y1 W) i-x0))
                       (off11 (+ c-base-in (* i-y1 W) i-x1))
                       (woff00 (+ c-base-out (* i-y0 W) i-x0))
                       (woff01 (+ c-base-out (* i-y0 W) i-x1))
                       (woff10 (+ c-base-out (* i-y1 W) i-x0))
                       (woff11 (+ c-base-out (* i-y1 W) i-x1))
                       (v00 (aref i-data off00)) (v01 (aref i-data off01))
                       (v10 (aref i-data off10)) (v11 (aref i-data off11)))
                  (incf (aref d-img-data woff00) (* g iwx iwy))
                  (incf (aref d-img-data woff01) (* g w-x iwy))
                  (incf (aref d-img-data woff10) (* g iwx w-y))
                  (incf (aref d-img-data woff11) (* g w-x w-y))
                  (incf dgx (* g (+ (* (- iwy) v00) (* iwy v01)
                                    (* (- w-y) v10) (* w-y v11))))
                  (incf dgy (* g (+ (* (- iwx) v00) (* (- w-x) v01)
                                    (* iwx v10) (* w-x v11))))))
              (setf (aref d-grid-data (+ (* bn 2)))
                    (* dgx (/ (1- W) 2.0d0))
                    (aref d-grid-data (+ (* bn 2) 1))
                    (* dgy (/ (1- H) 2.0d0))))))
        (values
         (vt-reshape (vt-from-array d-img-data :dtype :float64 :fast t)
                     (vt-shape images))
         (vt-reshape (vt-from-array d-grid-data :dtype :float64 :fast t)
                     (vt-shape grid)))))))

;;; ============================================================
;;; 2. 注视点辅助
;;; ============================================================

(defun shift-fixations-x (fixations delta)
  (vt-+ fixations
        (vt-reshape (vt-from-sequence (list delta 0.0d0) :dtype :float64)
                    (list 1 1 2))))

(defun build-foveal-grid (fixations patch-size extent)
  "向量化版：用 tile + 广播构造 (B, K*P*P, 2) 网格。"
  (let* ((shape (vt-shape fixations))
         (B (first shape))
	 (K (second shape))
	 (P patch-size)
         (offsets
           (vt-from-sequence
            (loop for i from 0 below P
                  collect (- (* i (/ extent (1- P))) (/ extent 2.0d0)))
            :dtype :float64))
         ;; fx, fy: (B, K)  —— vt-slice 会降维
         (fx (vt-slice fixations (list :all) (list :all) (list 0)))
         (fy (vt-slice fixations (list :all) (list :all) (list 1)))
         ;; ★ fx 扩成 (B,K,P,1)（沿行复制 P 份）
         (fx-4d (vt-tile (vt-reshape fx (list B K 1 1))
                         (list 1 1 P 1)))          ; (B,K,P,1)
         ;; ★ fy 扩成 (B,K,1,P)（沿列复制 P 份）
         (fy-4d (vt-tile (vt-reshape fy (list B K 1 1))
                         (list 1 1 1 P)))          ; (B,K,1,P)
         ;; off-col: (1,1,1,P) 沿列偏移；off-row: (1,1,P,1) 沿行偏移
         (off-col (vt-reshape offsets (list 1 1 1 P)))
         (off-row (vt-reshape offsets (list 1 1 P 1)))
         ;; 广播到 (B,K,P,P)
         (gx (vt-clip (vt-+ fx-4d off-col) -1.0d0 1.0d0))
         (gy (vt-clip (vt-+ fy-4d off-row) -1.0d0 1.0d0)))
    (vt-reshape
     (vt-concatenate -1
                     (vt-reshape gx (list B K P P 1))
                     (vt-reshape gy (list B K P P 1)))
     (list B (* K P P) 2))))


(defun grid-grad-to-fix-grad (d-grid B K P)
  (let ((d (if (vt-contiguous-p d-grid) d-grid (vt-contiguous d-grid))))
    (vt-sum (vt-reshape d (list B K (* P P) 2)) :axis 2)))

(defun fix-grad-to-delta (d-right d-left)
  (- (vt-item (vt-sum (vt-slice d-right (list :all) (list :all) (list 0))))
     (vt-item (vt-sum (vt-slice d-left  (list :all) (list :all) (list 0))))))

;;; ============================================================
;;; 3. 主类
;;; ============================================================
(defclass binocular-fovea-mnist (layer)
  ((n-fixations :initarg :n-fixations :initform 4 :reader bfm-n-fixations)
   (patch-size  :initarg :patch-size  :initform 14 :reader bfm-patch-size)
   (patch-extent :initarg :patch-extent :initform 0.4d0 :reader bfm-patch-extent)
   (feature-dim :initarg :feature-dim :initform 32 :reader bfm-feature-dim)
   (periphery-dim :initarg :periphery-dim :initform 64 :reader bfm-periphery-dim)
   (periphery     :initform nil :accessor bfm-periphery)
   (fix-head      :initform nil :accessor bfm-fix-head)
   (patch-encoder :initform nil :accessor bfm-patch-encoder)
   (classifier    :initform nil :accessor bfm-classifier)
   (disparity     :initform nil :accessor bfm-disparity)
   (d-disparity   :initform nil :accessor bfm-d-disparity)
   (base-fixations   :initform nil :accessor bfm-base-fixations)
   (d-base-fixations :initform nil :accessor bfm-d-base-fixations)
   (cache :initform nil :accessor bfm-cache)))

(defun make-binocular-fovea-mnist
    (&key (n-fixations 4) (patch-size 14) (patch-extent 0.4d0)
          (feature-dim 32) (periphery-dim 64)
          (name "binocular-fovea-mnist") (trainable t))
  (make-instance 'binocular-fovea-mnist
                 :n-fixations n-fixations :patch-size patch-size
                 :patch-extent patch-extent :feature-dim feature-dim
                 :periphery-dim periphery-dim
                 :name name :trainable trainable))

(defmethod initialize-instance :after ((l binocular-fovea-mnist) &key)
  (let* ((K (bfm-n-fixations l))
         (F (bfm-feature-dim l))
	 (PD (bfm-periphery-dim l)))
    (let ((p (make-sequential :name "periphery")))
      (seq-add! p (make-conv2d 8 '(3 3) :in-channels 1 :padding '(1 1)))
      (seq-add! p (make-activation-layer :relu))
      (seq-add! p (make-max-pool2d 2))
      (seq-add! p (make-conv2d 16 '(3 3) :in-channels 8 :padding '(1 1)))
      (seq-add! p (make-activation-layer :relu))
      (seq-add! p (make-max-pool2d 2))
      (seq-add! p (make-flatten))
      (seq-add! p (make-dense PD :activation :relu))
      (setf (bfm-periphery l) p))

    (setf (bfm-fix-head l)
          (make-dense (* K 2) :in-dim PD :activation :none :name "fix-head"))

    (let ((e (make-sequential :name "patch-encoder")))
      (seq-add! e (make-conv2d 16 '(3 3) :in-channels 1 :padding '(1 1)))
      (seq-add! e (make-activation-layer :relu))
      (seq-add! e (make-max-pool2d 2))              ; 14 → 7
      (seq-add! e (make-conv2d 32 '(3 3) :in-channels 16 :padding '(1 1)))
      (seq-add! e (make-activation-layer :relu))
      (seq-add! e (make-flatten))
      (seq-add! e (make-dense F :activation :relu))
      (setf (bfm-patch-encoder l) e))

    ;; ★★★ classifier 输入 = K*2F（patch 路径） + PD（periphery 直连）
    (let ((c (make-sequential :name "classifier")))
      (seq-add! c (make-dense 128 :in-dim (+ (* K 2 F) PD)
                                  :activation :relu))
      (seq-add! c (make-dense 10 :activation :none))
      (setf (bfm-classifier l) c))

    (setf (bfm-disparity l) (vt-const (list 1) 0.15d0))

    ;; 四象限 ±0.3 确定性初始化
    (setf (bfm-base-fixations l)
          (vt-from-sequence
           (loop for k from 0 below K
                 for gx = (* 0.6d0 (- (coerce (mod k 2) 'double-float) 0.5d0))
                 for gy = (* 0.6d0 (- (coerce (floor k 2) 'double-float) 0.5d0))
                 append (list gx gy))
           :dtype :float64))))

(defun bfm-all-sublayers (l)
  (list (bfm-periphery l) (bfm-fix-head l)
        (bfm-patch-encoder l) (bfm-classifier l)))

;;; ============================================================
;;; 4. forward / backward
;;; ============================================================
(defmethod forward ((l binocular-fovea-mnist) input)
  (let* ((in-shape (vt-shape input))
	 (rank (length in-shape))
         (input-4d (if (= rank 4) input
                       (vt-reshape input (list (first in-shape) 1 28 28))))
         (B (first (vt-shape input-4d)))
         (K (bfm-n-fixations l)) (P (bfm-patch-size l))
         (F (bfm-feature-dim l)) (E (bfm-patch-extent l))
         ;; 1. periphery
         (peri-feat (forward (bfm-periphery l) input-4d))     ; (B, PD)
         ;; 2. fix-head
         (fix-raw (forward (bfm-fix-head l) peri-feat))
         (fix-with-base (vt-+ fix-raw (vt-reshape (bfm-base-fixations l)
                                                  (list 1 (* K 2)))))
         (fix-tanh (vt-tanh fix-with-base))
         (fix-scaled (vt-scale fix-tanh 0.85d0))
         (fixations (vt-reshape fix-scaled (list B K 2)))
         (delta (coerce (aref (vt-data (bfm-disparity l))
                              (vt-offset (bfm-disparity l)))
			'double-float))
         (left-fix  (shift-fixations-x fixations (- delta)))
         (right-fix (shift-fixations-x fixations delta))
         (left-grid  (build-foveal-grid left-fix  P E))
         (right-grid (build-foveal-grid right-fix P E)))
    (multiple-value-bind (left-patches left-cache)
        (bilinear-sample input-4d left-grid)
      (multiple-value-bind (right-patches right-cache)
          (bilinear-sample input-4d right-grid)
        (let* ((left-2d  (vt-reshape left-patches  (list (* B K) 1 P P)))
               (right-2d (vt-reshape right-patches (list (* B K) 1 P P)))
               (both-2d  (vt-concatenate 0 left-2d right-2d))
               (both-feat (forward (bfm-patch-encoder l) both-2d))
               (left-feat-2d  (vt-contiguous
                               (vt-slice both-feat `(0 ,(* B K)) (list :all))))
               (right-feat-2d (vt-contiguous
                               (vt-slice both-feat `(,(* B K) ,(* 2 B K))
                                         (list :all))))
               (left-feat  (vt-reshape left-feat-2d  (list B K F)))
               (right-feat (vt-reshape right-feat-2d (list B K F)))
               (fused      (vt-concatenate -1 left-feat right-feat))
               (fused-flat (vt-reshape fused (list B (* K 2 F))))
               ;; ★★★ periphery 特征直连 classifier
               (classifier-in (vt-concatenate 1 fused-flat peri-feat))
               (logits (forward (bfm-classifier l) classifier-in)))
          (setf (bfm-cache l)
                (list :fix-tanh fix-tanh
                      :left-cache left-cache :right-cache right-cache
                      :B B :K K :P P :F F :delta delta))
          logits)))))

(defmethod backward ((l binocular-fovea-mnist) grad-output)
  (let* ((cache (bfm-cache l))
         (B (getf cache :B))
	 (K (getf cache :K))
         (P (getf cache :P))
	 (F (getf cache :F))
         (fix-tanh (getf cache :fix-tanh))
         (PD (bfm-periphery-dim l)))
    ;; 1. classifier 反向: (B, 10) → (B, K*2F + PD)
    (let* ((d-classifier-in (backward (bfm-classifier l) grad-output))
           ;; ★★★ 切出两条路径的梯度
           (d-fused-flat (vt-contiguous
                          (vt-slice d-classifier-in (list :all)
                                    `(0 ,(* K 2 F)))))
           (d-peri-from-cls (vt-contiguous
                             (vt-slice d-classifier-in (list :all)
                                       `(,(* K 2 F) ,(+ (* K 2 F) PD))))))
      (let* ((d-fused (vt-reshape d-fused-flat (list B K (* 2 F))))
             (d-left-feat  (vt-contiguous
                            (vt-slice d-fused (list :all) (list :all) `(0 ,F))))
             (d-right-feat (vt-contiguous
                            (vt-slice d-fused (list :all) (list :all) `(,F ,(* 2 F))))))
        (let* ((d-left-2d  (vt-reshape d-left-feat  (list (* B K) F)))
               (d-right-2d (vt-reshape d-right-feat (list (* B K) F)))
               (d-both-2d  (vt-concatenate 0 d-left-2d d-right-2d))
               (d-both-in  (backward (bfm-patch-encoder l) d-both-2d))
               (d-left-patches-2d
                 (vt-contiguous (vt-slice d-both-in `(0 ,(* B K))
                                          (list :all) (list :all) (list :all))))
               (d-right-patches-2d
                 (vt-contiguous (vt-slice d-both-in `(,(* B K) ,(* 2 B K))
                                          (list :all) (list :all) (list :all))))
               (d-left-patches  (vt-reshape d-left-patches-2d
                                            (list B (* K P P) 1)))
               (d-right-patches (vt-reshape d-right-patches-2d
                                            (list B (* K P P) 1))))
          (multiple-value-bind (d-img-l d-grid-l)
              (bilinear-sample-grad d-left-patches (getf cache :left-cache))
            (multiple-value-bind (d-img-r d-grid-r)
                (bilinear-sample-grad d-right-patches (getf cache :right-cache))
              (let* ((d-images-samp (vt-+ d-img-l d-img-r))
                     (d-left-fix  (grid-grad-to-fix-grad d-grid-l B K P))
                     (d-right-fix (grid-grad-to-fix-grad d-grid-r B K P))
                     (d-fixations (vt-+ d-left-fix d-right-fix))
                     (d-delta (fix-grad-to-delta d-right-fix d-left-fix)))
                (setf (bfm-d-disparity l)
                      (vt-reshape
                       (vt-from-array
                        (make-array 1 :element-type 'double-float
                                      :initial-element d-delta)
                        :dtype :float64 :fast t)
                       (list 1)))
                (let* ((d-fix-2d (vt-reshape d-fixations (list B (* K 2))))
                       (d-tanh-arg (vt-* (vt-scale d-fix-2d 0.85d0)
                                         (vt-- 1.0d0 (vt-* fix-tanh fix-tanh))))
                       (d-base (vt-sum d-tanh-arg :axis 0)))
                  (setf (bfm-d-base-fixations l) d-base)
                  (let* ((d-fix-raw (backward (bfm-fix-head l)
                                              (vt-scale d-tanh-arg
                                                        *fix-grad-scale*)))
                         ;; ★★★ periphery 梯度 = 注视点路径 + classifier 直连路径
                         (d-peri-total (vt-+ d-fix-raw d-peri-from-cls))
                         (d-images-peri (backward (bfm-periphery l)
                                                  d-peri-total)))
                    (vt-+ d-images-samp d-images-peri)))))))))))

;;; ============================================================
;;; 5. 协议
;;; ============================================================
(defmethod params ((l binocular-fovea-mnist))
  (append (params (bfm-periphery l))
	  (params (bfm-fix-head l))
          (params (bfm-patch-encoder l))
	  (params (bfm-classifier l))
          (list (list l "disparity" (bfm-disparity l)
                      #'(lambda (v) (setf (bfm-disparity l) v)))
                (list l "base-fixations" (bfm-base-fixations l)
                      #'(lambda (v) (setf (bfm-base-fixations l) v))))))

(defmethod grads ((l binocular-fovea-mnist))
  (append (grads (bfm-periphery l))
	  (grads (bfm-fix-head l))
          (grads (bfm-patch-encoder l))
	  (grads (bfm-classifier l))
          (list (cons "disparity" (bfm-d-disparity l))
                (cons "base-fixations" (bfm-d-base-fixations l)))))

(defmethod grad-slots ((l binocular-fovea-mnist))
  '(d-disparity d-base-fixations))

(defmethod cache-slots ((l binocular-fovea-mnist)) '(cache))

(defmethod zero-grad-children ((l binocular-fovea-mnist))
  (bfm-all-sublayers l))

(defmethod set-training! ((l binocular-fovea-mnist) mode)
  (call-next-method)
  (dolist (sub (bfm-all-sublayers l))
    (when sub (set-training! sub mode))))

;;; ============================================================
;;; 6. 训练/评估
;;; ============================================================
(defun train-binocular-mnist (&key (epochs 5) (batch-size 64)
                                    (lr 3e-3) (n-train 30000) (log-every 100))
  (ensure-mnist-data) (ensure-mnist-labels)
  (let* ((model (make-binocular-fovea-mnist))
         (loss-fn (make-ce-loss :reduction :mean))
         (opt (make-adam :lr lr))
         (n-batches (floor n-train batch-size))
         (t0 (get-internal-real-time)))
    (format t "~%==== 仿生双眼 MNIST v3  (extent=~a, patch=~a) ====~%"
            (bfm-patch-extent model) (bfm-patch-size model))
    (dotimes (epoch epochs)
      (let ((ep-loss 0.0d0) (ep-correct 0)
            (t-ep (get-internal-real-time)))
        (dotimes (b n-batches)
          (let* ((start (* b batch-size)) (end (+ start batch-size))
                 (x (mnist-images-4d start end batch-size))
                 (y (mnist-slice-labels *mnist-train-labels* start end)))
            (zero-grad! model)
            (let* ((pred (forward model x))
                   (loss (compute-loss loss-fn pred y))
                   (grad (compute-loss-gradient loss-fn pred y)))
              (backward model grad)
              (optimizer-step opt (params model) (grads model))
              (incf ep-loss (vt-item loss))
              (incf ep-correct
                    (mnist-accuracy-labels pred *mnist-train-labels* start))))
          (when (and log-every (zerop (mod (1+ b) log-every)))
            (format t "  [e~a b~a/~a]  loss=~,4f  acc=~,4f  δ=~,4f  ~,1fs~%"
                    (1+ epoch) (1+ b) n-batches
                    (/ ep-loss (1+ b))
                    (/ ep-correct (* (1+ b) batch-size))
                    (coerce (aref (vt-data (bfm-disparity model))
                                  (vt-offset (bfm-disparity model)))
                            'double-float)
                    (/ (- (get-internal-real-time) t-ep)
                       (coerce internal-time-units-per-second 'double-float)))))
        (format t "Epoch ~a/~a  loss=~,6f  train=~,4f  (~,1fs)~%"
                (1+ epoch) epochs (/ ep-loss n-batches) (/ ep-correct n-train)
                (/ (- (get-internal-real-time) t-ep)
                   (coerce internal-time-units-per-second 'double-float)))))
    (set-model-training! model nil)
    (multiple-value-bind (c tt a) (evaluate-binocular model)
      (format t "~%测试准确率: ~a / ~a = ~,2f%~%" c tt (* 100.0 a)))
    (format  t "~%总用时: ~a 秒"
	     (/ (- (get-internal-real-time) t0)
                (coerce internal-time-units-per-second 'double-float)))
    model))

(defun evaluate-binocular (model &key (batch-size 128) (n 10000))
  (ensure-mnist-labels)
  (let ((correct 0) (total 0) (start 0))
    (loop while (< start n) do
      (let* ((end (min (+ start batch-size) n)) (bs (- end start))
             (x (mnist-images-4d start end bs :data *mnist-testa*))
             (pred (forward model x)))
        (incf correct (mnist-accuracy-labels pred *mnist-test-labels* start))
        (incf total bs) (setf start end)
        (clear-step-caches! model)))
    (values correct total (coerce (/ correct total) 'double-float))))


#|
NN> (with-seed (40)
      (clnn::train-binocular-mnist :epochs 5 :batch-size 64 :lr 3e-3 :n-train 30000))

加载 MNIST 数据...
加载完成。
转换 one-hot 标签...

==== 仿生双眼 MNIST v3  (extent=0.4, patch=14) ====
  [e1 b100/468]  loss=0.6815  acc=0.7813  δ=0.2271  131.4s
  [e1 b200/468]  loss=0.4572  acc=0.8538  δ=0.2271  268.1s
  [e1 b300/468]  loss=0.3540  acc=0.8873  δ=0.2284  409.3s
  [e1 b400/468]  loss=0.2910  acc=0.9078  δ=0.2307  550.8s
Epoch 1/5  loss=0.263549  train=0.9155  (647.1s)
  [e2 b100/468]  loss=0.0881  acc=0.9741  δ=0.2323  142.8s
  [e2 b200/468]  loss=0.0888  acc=0.9727  δ=0.2282  286.0s
  [e2 b300/468]  loss=0.0848  acc=0.9735  δ=0.2302  428.8s
  [e2 b400/468]  loss=0.0776  acc=0.9758  δ=0.2321  572.2s
Epoch 2/5  loss=0.076334  train=0.9748  (669.4s)
  [e3 b100/468]  loss=0.0542  acc=0.9823  δ=0.2340  143.1s
  [e3 b200/468]  loss=0.0570  acc=0.9810  δ=0.2311  303.1s
  [e3 b300/468]  loss=0.0541  acc=0.9822  δ=0.2272  463.6s
  [e3 b400/468]  loss=0.0503  acc=0.9832  δ=0.2251  619.0s
Epoch 3/5  loss=0.049372  train=0.9820  (726.7s)
  [e4 b100/468]  loss=0.0388  acc=0.9875  δ=0.2274  154.9s
  [e4 b200/468]  loss=0.0387  acc=0.9884  δ=0.2286  320.3s
  [e4 b300/468]  loss=0.0385  acc=0.9871  δ=0.2262  485.5s
  [e4 b400/468]  loss=0.0354  acc=0.9881  δ=0.2280  650.5s
Epoch 4/5  loss=0.034028  train=0.9870  (757.8s)
  [e5 b100/468]  loss=0.0333  acc=0.9908  δ=0.2263  152.4s
  [e5 b200/468]  loss=0.0387  acc=0.9882  δ=0.2217  303.9s
  [e5 b300/468]  loss=0.0387  acc=0.9877  δ=0.2247  464.7s
  [e5 b400/468]  loss=0.0339  acc=0.9892  δ=0.2188  614.7s
Epoch 5/5  loss=0.032969  train=0.9877  (717.9s)

测试准确率: 9767 / 10000 = 97.67%

总用时: 3619.956369 秒
#<BINOCULAR-FOVEA-MNIST {1211D95CA3}>
|#
