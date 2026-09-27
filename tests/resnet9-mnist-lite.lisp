;;;; resnet9-mnist-tiny.lisp
;;;; ResNet-9 mini on MNIST —— 内存安全版（峰值 < 300 MB）
;;;;
;;;; 关键设计（与前几版的区别）：
;;;;   1. prep-conv 后立刻 MaxPool 降到 14×14，避免在 28×28 上做重卷积
;;;;   2. 通道数 8/16（不是 16/32），im2col 尺寸再降 4 倍
;;;;   3. batch-size 32（不是 128），col 矩阵再降 4 倍
;;;;   4. 数据集用精简 loader，只加载 N 个样本
;;;;
;;;; 用法：
;;;;   (in-package :nn)
;;;;   (load "resnet9-mnist-tiny.lisp")
;;;;   (train-resnet9-tiny)

(eval-when (:compile-toplevel :load-toplevel :execute)
  (ql:quickload '(:chipz :clnn)))
(in-package :nn)

;;; ============================================================================
;;; 1. 精简 MNIST 加载（同前）
;;; ============================================================================

(defun mnist-lite-path (filename)
  (asdf:system-relative-pathname
   "clnn" (concatenate 'string "mnist-data/" filename)))

(defun load-mnist-lite (type n-samples)
  (let* ((images-name (if (eq type :train)
                          "train-images-idx3-ubyte.gz"
                          "t10k-images-idx3-ubyte.gz"))
         (labels-name (if (eq type :train)
                          "train-labels-idx1-ubyte.gz"
                          "t10k-labels-idx1-ubyte.gz"))
         (n-pixels (* n-samples 28 28))
         (images (make-array n-pixels :element-type 'double-float
                                      :initial-element 0.0d0))
         (labels (make-array n-samples :element-type 'fixnum
                                       :initial-element 0)))
    (with-open-file (f (mnist-lite-path images-name)
                       :element-type '(unsigned-byte 8))
      (let ((raw (chipz:decompress nil 'chipz:gzip f)))
        (dotimes (i n-pixels)
          (setf (aref images i)
                (/ (aref raw (+ 16 i)) 255.0d0)))))
    (with-open-file (f (mnist-lite-path labels-name)
                       :element-type '(unsigned-byte 8))
      (let ((raw (chipz:decompress nil 'chipz:gzip f)))
        (dotimes (i n-samples)
          (setf (aref labels i) (aref raw (+ 8 i))))))
    (values
     (vt-reshape (vt-from-array images :dtype :float64 :fast t)
                 (list n-samples 1 28 28))
     labels)))

;;; ============================================================================
;;; 2. 切片工具
;;; ============================================================================

(defun slice-images (images start end)
  (let ((s (vt-slice images (list start end)
                            (list :all) (list :all) (list :all))))
    (if (vt-contiguous-p s) s (vt-contiguous s))))

(defun slice-labels (labels start end)
  (vt-from-sequence
   (loop for i from start below end collect (aref labels i))
   :dtype :int64))

(defun accuracy-labels (pred labels start)
  (let* ((batch (first (vt-shape pred)))
         (n-classes (second (vt-shape pred)))
         (p-data (vt-data pred))
         (p-off  (vt-offset pred))
         (p-rs   (first  (vt-strides pred)))
         (p-cs   (second (vt-strides pred)))
         (correct 0))
    (dotimes (i batch)
      (let ((best 0) (val most-negative-double-float)
                     (base (+ p-off (* i p-rs))))
        (dotimes (c n-classes)
          (let ((v (aref p-data (+ base (* c p-cs)))))
            (when (> v val) (setf val v best c))))
        (when (= best (aref labels (+ start i)))
          (incf correct))))
    correct))

;;; ============================================================================
;;; 3. 模型（内存优化版）
;;; ============================================================================

(defun make-residual-block (channels name)
  "3×3 Conv + BN + ReLU 的两层残差块。"
  (let ((inner (make-sequential :name (concatenate 'string name "-inner"))))
    (seq-add! inner (make-conv2d channels '(3 3)
                                 :in-channels channels :padding '(1 1)
                                 :use-bias nil
                                 :name (concatenate 'string name "-c1")))
    (seq-add! inner (make-batch-norm channels
                                     :name (concatenate 'string name "-bn1")))
    (seq-add! inner (make-activation-layer :relu
                                           :name (concatenate 'string name "-r1")))
    (seq-add! inner (make-conv2d channels '(3 3)
                                 :in-channels channels :padding '(1 1)
                                 :use-bias nil
                                 :name (concatenate 'string name "-c2")))
    (seq-add! inner (make-batch-norm channels
                                     :name (concatenate 'string name "-bn2")))
    (make-residual inner :name name)))

(defun make-resnet9-tiny ()
  "内存安全的 ResNet-9 mini。
   分辨率轨迹：28×28 → 14×14 → 7×7。
   im2col 峰值 < 20 MB（batch=32 时）。"
  (let ((m (make-sequential :name "resnet9-tiny")))

    ;; ---- prep: 28×28, 1→8 ----
    (seq-add! m (make-conv2d 8 '(3 3)
                             :in-channels 1
                             :padding '(1 1)
                             :use-bias nil
                             :name "prep-conv"))
    (seq-add! m (make-batch-norm 8 :name "prep-bn"))
    (seq-add! m (make-activation-layer :relu :name "prep-relu"))

    ;; ★ 关键：立刻降分辨率 28→14，后续所有重卷积都在 14×14 上做
    (seq-add! m (make-max-pool2d 2 :name "downsample"))

    ;; ---- trans: 14×14, 8→16 ----
    (seq-add! m (make-conv2d 16 '(3 3)
                             :in-channels 8
                             :padding '(1 1)
                             :use-bias nil
                             :name "trans-conv"))
    (seq-add! m (make-batch-norm 16 :name "trans-bn"))
    (seq-add! m (make-activation-layer :relu :name "trans-relu"))

    ;; ---- block1: 14×14, 16→16 ----
    (seq-add! m (make-residual-block 16 "b1"))
    (seq-add! m (make-activation-layer :relu :name "b1-out"))
    (seq-add! m (make-max-pool2d 2 :name "pool1"))       ; 14→7

    ;; ---- block2: 7×7, 16→16 ----
    (seq-add! m (make-residual-block 16 "b2"))
    (seq-add! m (make-activation-layer :relu :name "b2-out"))

    ;; ---- 分类头 ----
    (seq-add! m (make-global-avg-pool2d :name "gap"))    ; 7×7×16 → 16
    (seq-add! m (make-dense 10 :activation :none :name "fc"))
    m))

;;; ============================================================================
;;; 4. 评估
;;; ============================================================================

(defun evaluate-tiny (model test-x test-y n-test &key (batch-size 64))
  (let ((correct 0) (total 0) (start 0))
    (with-training nil
      (loop while (< start n-test) do
        (let* ((end (min (+ start batch-size) n-test))
               (x (slice-images test-x start end))
               (pred (forward model x)))
          (incf correct (accuracy-labels pred test-y start))
          (incf total (- end start))
          (setf start end)
          (clear-step-caches! model))))
    (values correct total
            (coerce (/ correct total) 'double-float))))

;;; ============================================================================
;;; 5. 训练
;;; ============================================================================

(defun train-resnet9-tiny (&key (epochs 5)
                                 (batch-size 32)         ; ★ 32 而非 128
                                 (lr 5.0d-2)
                                 (momentum 0.9d0)
                                 (weight-decay 5.0d-4)
                                 (nesterov t)
                                 (n-train 10000)
                                 (n-test 2000)
                                 (log-every 40))
  (format t "~&加载 MNIST 子集（train=~a, test=~a）...~%" n-train n-test)
  (multiple-value-bind (train-x train-y) (load-mnist-lite :train n-train)
    (multiple-value-bind (test-x test-y) (load-mnist-lite :test n-test)

      (let* ((model     (make-resnet9-tiny))
             (opt       (make-sgd :lr lr :momentum momentum
                                  :nesterov nesterov
                                  :weight-decay weight-decay))
             (scheduler (make-cosine-annealing-lr opt epochs :eta-min 0.0d0))
             (loss-fn   (make-ce-loss :reduction :mean))
             (n-batches (floor n-train batch-size))
             (t0 (get-internal-real-time)))

        ;; ★ 触发延迟初始化
        (build-model model (vt-zeros '(1 1 28 28)))
        (set-model-training! model t)

        (format t "~%============================================================~%")
        (format t "  ResNet-9 tiny on MNIST（内存安全版）~%")
        (format t "============================================================~%")
        (format t "  train=~a  test=~a  batch=~a  epochs=~a~%"
                n-train n-test batch-size epochs)
        (format t "  优化器: SGD(lr=~a, momentum=~a, nesterov=~a, wd=~a)~%"
                lr momentum nesterov weight-decay)
        (format t "  参数量: ~a~%" (param-count model))
        (format t "  im2col 峰值 < 20 MB（vs 完整版 810 MB）~%")
        (format t "============================================================~%~%")

        (dotimes (epoch epochs)
          (let ((epoch-loss 0.0d0)
                (epoch-correct 0)
                (t-epoch (get-internal-real-time)))
            (dotimes (b n-batches)
              (let* ((start (* b batch-size))
                     (end   (+ start batch-size))
                     (x (slice-images train-x start end))
                     (y (slice-labels train-y start end)))
                (zero-grad! model)
                (let* ((pred (forward model x))
                       (loss (compute-loss loss-fn pred y))
                       (grad (compute-loss-gradient loss-fn pred y)))
                  (backward model grad)
                  (optimizer-step opt (params model) (grads model))
                  (incf epoch-loss (coerce (vt-item loss) 'double-float))
                  (incf epoch-correct (accuracy-labels pred train-y start))))
              (when (zerop (mod (1+ b) log-every))
                (format t "  [e~a b~a/~a] loss=~,4f acc=~,4f ~,1fs~%"
                        (1+ epoch) (1+ b) n-batches
                        (/ epoch-loss (1+ b))
                        (/ (float epoch-correct)
                           (float (* (1+ b) batch-size)))
                        (/ (- (get-internal-real-time) t-epoch)
                           (coerce internal-time-units-per-second 'double-float)))))
            (scheduler-step! scheduler)
            (format t "Epoch ~a/~a  lr=~,5f  loss=~,4f  train=~,4f  (~,1fs)~%"
                    (1+ epoch) epochs
                    (scheduler-get-lr scheduler)
                    (/ epoch-loss n-batches)
                    (/ (float epoch-correct) (float n-train))
                    (/ (- (get-internal-real-time) t-epoch)
                       (coerce internal-time-units-per-second 'double-float)))))

        (multiple-value-bind (c tt a)
            (evaluate-tiny model test-x test-y n-test :batch-size 64)
          (declare (ignore tt))
          (format t "~%============================================================~%")
          (format t "  测试准确率: ~a / ~a = ~,2f%~%" c n-test (* 100.0 a))
          (format t "  总耗时: ~,1fs~%"
                  (/ (- (get-internal-real-time) t0)
                     (coerce internal-time-units-per-second 'double-float)))
          (format t "============================================================~%"))
        model))))



(train-resnet9-tiny )

#| 
加载 MNIST 子集（train=10000, test=2000）...

============================================================
  ResNet-9 tiny on MNIST（内存安全版）
============================================================
  train=10000  test=2000  batch=32  epochs=5
  优化器: SGD(lr=0.05, momentum=0.9, nesterov=T, wd=5.0e-4)
  参数量: 10786
  im2col 峰值 < 20 MB（vs 完整版 810 MB）
============================================================

  [e1 b40/312] loss=1.6146 acc=0.4805 17.4s
  [e1 b80/312] loss=1.0978 acc=0.6586 35.0s
  [e1 b120/312] loss=0.8429 acc=0.7432 54.1s
  [e1 b160/312] loss=0.7068 acc=0.7857 73.0s
  [e1 b200/312] loss=0.6157 acc=0.8159 91.9s
  [e1 b240/312] loss=0.5487 acc=0.8365 111.7s
  [e1 b280/312] loss=0.5039 acc=0.8511 132.1s
Epoch 1/5  lr=0.04523  loss=0.4738  train=0.8590  (147.1s)
  [e2 b40/312] loss=0.1610 acc=0.9570 19.9s
  [e2 b80/312] loss=0.1468 acc=0.9586 40.0s
  [e2 b120/312] loss=0.1439 acc=0.9599 59.6s
  [e2 b160/312] loss=0.1400 acc=0.9617 79.9s
  [e2 b200/312] loss=0.1365 acc=0.9631 99.8s
  [e2 b240/312] loss=0.1350 acc=0.9626 120.1s
  [e2 b280/312] loss=0.1370 acc=0.9616 139.9s
Epoch 2/5  lr=0.03273  loss=0.1360  train=0.9599  (156.2s)
  [e3 b40/312] loss=0.0958 acc=0.9711 20.1s
  [e3 b80/312] loss=0.0882 acc=0.9738 40.5s
  [e3 b120/312] loss=0.0840 acc=0.9758 61.6s
  [e3 b160/312] loss=0.0837 acc=0.9756 86.5s
  [e3 b200/312] loss=0.0848 acc=0.9755 108.4s
  [e3 b240/312] loss=0.0843 acc=0.9757 129.0s
  [e3 b280/312] loss=0.0859 acc=0.9749 148.8s
Epoch 3/5  lr=0.01727  loss=0.0844  train=0.9739  (165.6s)
  [e4 b40/312] loss=0.0721 acc=0.9789 20.9s
  [e4 b80/312] loss=0.0666 acc=0.9809 44.5s
  [e4 b120/312] loss=0.0628 acc=0.9818 66.4s
  [e4 b160/312] loss=0.0599 acc=0.9830 86.9s
  [e4 b200/312] loss=0.0585 acc=0.9837 108.0s
  [e4 b240/312] loss=0.0590 acc=0.9828 128.7s
  [e4 b280/312] loss=0.0580 acc=0.9828 149.5s
Epoch 4/5  lr=0.00477  loss=0.0566  train=0.9818  (165.5s)
  [e5 b40/312] loss=0.0480 acc=0.9836 20.7s
  [e5 b80/312] loss=0.0419 acc=0.9863 41.9s
  [e5 b120/312] loss=0.0419 acc=0.9870 63.8s
  [e5 b160/312] loss=0.0400 acc=0.9881 84.9s
  [e5 b200/312] loss=0.0392 acc=0.9886 105.1s
  [e5 b240/312] loss=0.0371 acc=0.9895 126.0s
  [e5 b280/312] loss=0.0356 acc=0.9906 146.5s
Epoch 5/5  lr=0.00000  loss=0.0340  train=0.9899  (162.6s)

============================================================
  测试准确率: 1960 / 2000 = 98.00%
  总耗时: 811.4s
============================================================
|#
