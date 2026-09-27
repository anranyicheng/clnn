;;;; resnet9-cifar-tiny.lisp
;;;; ============================================================================
;;;; ResNet-9 mini on CIFAR-10 —— 内存安全版
;;;; ============================================================================
;;;;
;;;; 关键改动（相对原版）：
;;;;   1. prep 之后立即 MaxPool 2×2，把 32×32 降到 16×16
;;;;      重 conv 全在 16×16 上做，im2col 内存降 4 倍
;;;;   2. 默认 batch-size 128 → 32，内存再降 4 倍
;;;;   3. 默认 n-train 10000 → 5000，数据缓存减半
;;;;
;;;; 内存对比：
;;;;   原版  峰值 ~3 GB
;;;;   本版  峰值 ~200 MB（15× 下降）
;;;;
;;;; 运行：
;;;;   (in-package :nn)
;;;;   (load "resnet9-cifar-tiny.lisp")
;;;;   (run-resnet9-cifar)
;;;;
;;;; 预期：5 epoch 内 test-acc 55%+，总耗时 ~8–15 分钟（CPU）

(in-package :nn)

;;; ============================================================================
;;; 1. 数据加载（与原版相同）
;;; ============================================================================

(defun cifar10-data-dir ()
  "默认数据目录。用 (setf (cifar10-data-dir) ...) 覆盖。"
  "~/cifar-10-binary/")

(defun read-uint8-file (path n)
  (let ((buf (make-array n :element-type '(unsigned-byte 8))))
    (with-open-file (f path :element-type '(unsigned-byte 8)
                            :if-does-not-exist :error)
      (read-sequence buf f))
    buf))

(defun load-cifar10-images (path n-images)
  (let* ((n-pixels (* n-images 3072))
         (raw  (read-uint8-file path n-pixels))
         (norm (make-array n-pixels :element-type 'double-float)))
    (dotimes (i n-pixels)
      (setf (aref norm i) (/ (aref raw i) 255.0d0)))
    (vt-reshape (vt-from-array norm :dtype :float64 :fast t)
                (list n-images 3 32 32))))

(defun load-cifar10-labels (path n)
  (let* ((raw (read-uint8-file path n))
         (out (make-array n :element-type 'fixnum)))
    (dotimes (i n)
      (setf (aref out i) (aref raw i)))
    out))

(defun load-cifar10 (&key (n-train 5000) (n-test 1000)
                       (dir (cifar10-data-dir)))
  (format t "~&加载 CIFAR-10 数据（train=~a test=~a）...~%" n-train n-test)
  (let* ((train-x (load-cifar10-images
                   (merge-pathnames "train-images.bin" dir) n-train))
         (train-y (load-cifar10-labels
                   (merge-pathnames "train-labels.bin" dir) n-train))
         (test-x  (load-cifar10-images
                   (merge-pathnames "test-images.bin" dir) n-test))
         (test-y  (load-cifar10-labels
                   (merge-pathnames "test-labels.bin" dir) n-test)))
    (format t "加载完成。~%")
    (values train-x train-y test-x test-y)))

;;; ============================================================================
;;; 2. 切片与准确率工具（与原版相同）
;;; ============================================================================

(defun cifar-slice-images (images start end)
  (let ((s (vt-slice images (list start end)
                     (list :all) (list :all) (list :all))))
    (if (vt-contiguous-p s) s (vt-contiguous s))))

(defun cifar-slice-labels (labels start end)
  (vt-from-sequence
   (loop for i from start below end collect (aref labels i))
   :dtype :int64))

(defun cifar-accuracy (pred labels start)
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
;;; 3. 模型构造（★ 关键改动：early downsample）
;;; ============================================================================

(defun make-residual-block (channels name)
  "3×3 Conv + BN + ReLU 的两层残差块（无最后激活）。"
  (let* ((inner (make-sequential :name (concatenate 'string name "-inner"))))
    (seq-add! inner (make-conv2d channels '(3 3)
                                 :in-channels channels
                                 :padding '(1 1)
                                 :use-bias nil
                                 :name (concatenate 'string name "-c1")))
    (seq-add! inner (make-batch-norm channels
                                     :name (concatenate 'string name "-bn1")))
    (seq-add! inner (make-activation-layer :relu
                                           :name (concatenate 'string name "-r1")))
    (seq-add! inner (make-conv2d channels '(3 3)
                                 :in-channels channels
                                 :padding '(1 1)
                                 :use-bias nil
                                 :name (concatenate 'string name "-c2")))
    (seq-add! inner (make-batch-norm channels
                                     :name (concatenate 'string name "-bn2")))
    (make-residual inner :name name)))

(defun make-resnet9-cifar-tiny (&key (n-classes 10))
  "内存安全的 ResNet-9 mini。
   分辨率轨迹：32×32 → 16×16 → 8×8 → 4×4。
   ★ 关键：prep 之后立即下采样到 16×16，所有重 conv 都在 ≤16×16 上做。"
  (let ((m (make-sequential :name "resnet9-cifar-tiny")))

    ;; ---- prep: 3 → 32，保持 32×32（im2col: 32×32×3×9 ≈ 28K/张，很便宜）----
    (seq-add! m (make-conv2d 32 '(3 3)
                             :in-channels 3
                             :padding '(1 1)
                             :use-bias nil
                             :name "prep-conv"))
    (seq-add! m (make-batch-norm 32 :name "prep-bn"))
    (seq-add! m (make-activation-layer :relu :name "prep-relu"))

    ;; ★★★ 早期下采样：32×32 → 16×16
    ;;   之后的 conv 的 im2col 缓存都小 4 倍
    (seq-add! m (make-max-pool2d 2 :name "downsample"))

    ;; ---- transition: 32 → 64，在 16×16 上 ----
    (seq-add! m (make-conv2d 64 '(3 3)
                             :in-channels 32
                             :padding '(1 1)
                             :use-bias nil
                             :name "trans-conv"))
    (seq-add! m (make-batch-norm 64 :name "trans-bn"))
    (seq-add! m (make-activation-layer :relu :name "trans-relu"))

    ;; ---- block1: 16×16, 64ch ----
    (seq-add! m (make-residual-block 64 "block1"))
    (seq-add! m (make-activation-layer :relu :name "block1-out"))
    (seq-add! m (make-max-pool2d 2 :name "pool1"))       ; 16→8

    ;; ---- block2: 8×8, 64ch ----
    (seq-add! m (make-residual-block 64 "block2"))
    (seq-add! m (make-activation-layer :relu :name "block2-out"))
    (seq-add! m (make-max-pool2d 2 :name "pool2"))       ; 8→4

    ;; ---- block3: 4×4, 64ch ----
    (seq-add! m (make-residual-block 64 "block3"))
    (seq-add! m (make-activation-layer :relu :name "block3-out"))
    ;; 不再 pool：4×4 已经很小

    ;; ---- 分类头 ----
    (seq-add! m (make-global-avg-pool2d :name "gap"))    ; 4×4×64 → 64
    (seq-add! m (make-dense n-classes
                            :activation :none
                            :name "fc"))
    m))

;;; ============================================================================
;;; 4. 评估
;;; ============================================================================

(defun evaluate-resnet9-cifar (model test-x test-y
                               &key (batch-size 32) (n-test 1000))
  (let ((correct 0) (total 0) (start 0))
    (with-training nil
      (loop while (< start n-test) do
        (let* ((end (min (+ start batch-size) n-test))
               (x (cifar-slice-images test-x start end))
               (pred (forward model x)))
          (incf correct (cifar-accuracy pred test-y start))
          (incf total (- end start))
          (setf start end)
          (clear-step-caches! model))))
    (values correct total
            (coerce (/ correct total) 'double-float))))

;;; ============================================================================
;;; 5. 训练循环
;;; ============================================================================

(defun train-resnet9-cifar (&key (epochs 5)
                              (batch-size 32)         ; ★ 128 → 32
                              (lr 5.0d-2)             ; batch 变小，lr 略降
                              (momentum 0.9d0)
                              (weight-decay 5.0d-4)
                              (nesterov t)
                              (n-train 5000)          ; ★ 10000 → 5000
                              (n-test 1000)           ; ★ 2000 → 1000
                              (log-every 20))
  (multiple-value-bind (train-x train-y test-x test-y)
      (load-cifar10 :n-train n-train :n-test n-test)

    (let* ((model     (make-resnet9-cifar-tiny))
           (opt       (make-sgd :lr lr
                                :momentum momentum
                                :nesterov nesterov
                                :weight-decay weight-decay))
           (scheduler (make-cosine-annealing-lr opt epochs :eta-min 0.0d0))
           (loss-fn   (make-ce-loss :reduction :mean))
           (n-batches (floor n-train batch-size))
           (t0 (get-internal-real-time)))

      (build-model model (vt-zeros (list 1 3 32 32)))
      (set-model-training! model t)

      (format t "~%============================================================~%")
      (format t "  ResNet-9 tiny on CIFAR-10（内存安全版）~%")
      (format t "============================================================~%")
      (format t "  train=~a  test=~a  batch=~a  epochs=~a~%"
              n-train n-test batch-size epochs)
      (format t "  优化器: SGD(lr=~a, momentum=~a, nesterov=~a, wd=~a)~%"
              lr momentum nesterov weight-decay)
      (format t "  调度器: CosineAnnealingLR(T_max=~a)~%" epochs)
      (format t "  参数量: ~a~%" (param-count model))
      (format t "  分辨率轨迹: 32→16→8→4~%")
      (format t "============================================================~%~%")

      (dotimes (epoch epochs)
        (let ((epoch-loss 0.0d0)
              (epoch-correct 0)
              (t-epoch (get-internal-real-time)))
          (dotimes (b n-batches)
            (let* ((start (* b batch-size))
                   (end   (+ start batch-size))
                   (x (cifar-slice-images train-x start end))
                   (y (cifar-slice-labels train-y start end)))
              (zero-grad! model)
              (let* ((pred (forward model x))
                     (loss (compute-loss loss-fn pred y))
                     (grad (compute-loss-gradient loss-fn pred y)))
                (backward model grad)
                (optimizer-step opt (params model) (grads model))
                (incf epoch-loss (coerce (vt-item loss) 'double-float))
                (incf epoch-correct (cifar-accuracy pred train-y start))))
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
          (evaluate-resnet9-cifar model test-x test-y
                                  :batch-size batch-size :n-test n-test)
        (declare (ignore tt))
        (format t "~%============================================================~%")
        (format t "  测试准确率: ~a / ~a = ~,2f%~%" c n-test (* 100.0 a))
        (format t "  总耗时: ~,1fs~%"
                (/ (- (get-internal-real-time) t0)
                   (coerce internal-time-units-per-second 'double-float)))
        (format t "============================================================~%"))
      model)))

;;; ============================================================================
;;; 6. 入口
;;; ============================================================================

(defun run-resnet9-cifar ()
  "训练 + 评估，返回训练好的模型。"
  (train-resnet9-cifar :epochs 5
                       :batch-size 32
                       :lr 5.0d-2
                       :n-train 5000
                       :n-test 1000))








;;;; resnet9-cifar.lisp
;;;; ============================================================================
;;;; ResNet-9 on CIFAR-10
;;;; ============================================================================
;;;;
;;;; 【本示例的独特价值】
;;;;   此前所有 demo（MNIST、RL、文本分类）都未覆盖两个组件：
;;;;     1. RESIDUAL 容器       —— 库中唯一从未使用过的容器层
;;;;     2. SGD / NESTEROV / COSINE-ANNEALING —— 一直用 Adam，从未验证别的
;;;;   本示例一次性覆盖这两块，并顺带验证：
;;;;     - BatchNorm 在深层网络多步训练下的 running stats 累积
;;;;     - GlobalAvgPool 在大尺寸特征图（4×4×64）上的反向
;;;;     - 每个 epoch 调用 scheduler-step! 的用法
;;;;
;;;; 【架构】ResNet-9 mini（通道数保持 64，让 residual 能直接相加）
;;;;
;;;;   prep:   Conv(3→32, 3×3, pad=1) + BN + ReLU              [32×32×32]
;;;;   trans:  Conv(32→64, 3×3, pad=1) + BN + ReLU             [32×32×64]
;;;;   block1: Residual[Conv+BN+ReLU+Conv+BN] + ReLU + MaxPool(2)  [16×16×64]
;;;;   block2: Residual[Conv+BN+ReLU+Conv+BN] + ReLU + MaxPool(2)  [ 8×8×64]
;;;;   block3: Residual[Conv+BN+ReLU+Conv+BN] + ReLU + MaxPool(2)  [ 4×4×64]
;;;;   GAP:    GlobalAvgPool                                   [64]
;;;;   fc:     Dense(64→10)                                    [10]
;;;;
;;;;   所有 residual 块内部通道数不变（64→64），
;;;;   因此可直接使用库的 MAKE-RESIDUAL（forward 做 input + block(input)）。
;;;;   通道变化只发生在 prep 和 trans 这两个非 residual 位置。
;;;;
;;;; 【运行】
;;;;   (in-package :nn)
;;;;   (load "resnet9-cifar.lisp")
;;;;   (train-resnet9-cifar)                  ; 默认 10000 训练 / 2000 测试
;;;;   (train-resnet9-cifar :n-train 50000)   ; 全量（内存 ~1.2GB）
;;;;
;;;; 【预期】10 epoch 内 train-acc > 70%，test-acc > 60%

(in-package :nn)

;;; ============================================================================
;;; 1. 数据加载（raw binary）
;;; ============================================================================

(defun cifar10-data-dir ()
  "默认数据目录。可用 (setf (cifar10-data-dir) ...) 覆盖（需要定义 defparameter）。"
  "~/cifar-10-binary/")

(defun read-uint8-file (path n)
  "读 PATH 的前 N 个字节，返回 (unsigned-byte 8) 数组。"
  (let ((buf (make-array n :element-type '(unsigned-byte 8))))
    (with-open-file (f path :element-type '(unsigned-byte 8)
                            :if-does-not-exist :error)
      (read-sequence buf f))
    buf))

(defun load-cifar10-images (path n-images)
  "读 N-IMAGES 张图像（每张 3072 字节，NCHW 布局），归一化到 [0,1]。
   返回形状 (N, 3, 32, 32) 的 float64 张量。"
  (let* ((n-pixels (* n-images 3072))
         (raw  (read-uint8-file path n-pixels))
         (norm (make-array n-pixels :element-type 'double-float)))
    (dotimes (i n-pixels)
      (setf (aref norm i) (/ (aref raw i) 255.0d0)))
    (vt-reshape (vt-from-array norm :dtype :float64 :fast t)
                (list n-images 3 32 32))))

(defun load-cifar10-labels (path n)
  "读 N 个 uint8 标签，返回 fixnum 数组。"
  (let* ((raw (read-uint8-file path n))
         (out (make-array n :element-type 'fixnum)))
    (dotimes (i n)
      (setf (aref out i) (aref raw i)))
    out))

(defun load-cifar10 (&key (n-train 10000) (n-test 2000)
                            (dir (cifar10-data-dir)))
  "加载 CIFAR-10 的前 N-TRAIN / N-TEST 个样本。
   返回 (values train-x train-y test-x test-y)：
     TRAIN-X / TEST-X 形状 (N, 3, 32, 32) float64
     TRAIN-Y / TEST-Y fixnum 数组"
  (format t "~&加载 CIFAR-10 数据（train=~a test=~a）...~%" n-train n-test)
  (let* ((train-x (load-cifar10-images
                   (merge-pathnames "train-images.bin" dir) n-train))
         (train-y (load-cifar10-labels
                   (merge-pathnames "train-labels.bin" dir) n-train))
         (test-x  (load-cifar10-images
                   (merge-pathnames "test-images.bin" dir) n-test))
         (test-y  (load-cifar10-labels
                   (merge-pathnames "test-labels.bin" dir) n-test)))
    (format t "加载完成。~%")
    (values train-x train-y test-x test-y)))

;;; ============================================================================
;;; 2. 切片与准确率工具
;;; ============================================================================

(defun cifar-slice-images (images start end)
  "取 [START, END) 的样本，返回连续 (B, 3, 32, 32)。"
  (let ((s (vt-slice images (list start end)
                            (list :all) (list :all) (list :all))))
    (if (vt-contiguous-p s) s (vt-contiguous s))))

(defun cifar-slice-labels (labels start end)
  "取 fixnum 数组的 [START, END)，返回 int64 VT。"
  (vt-from-sequence
   (loop for i from start below end collect (aref labels i))
   :dtype :int64))

(defun cifar-accuracy (pred labels start)
  "PRED 形状 (B, 10)；LABELS fixnum 数组；START 是标签数组的起始索引。
   返回 batch 内正确样本数。"
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
;;; 3. 模型构造
;;; ============================================================================

(defun make-residual-block (channels name)
  "构造一个 3×3 Conv + BN + ReLU 的两层残差块（不含最后 ReLU）。
   两层都保持 CHANNELS 通道数，使得残差相加成立。
   使用 USE-BIAS=NIL（因为 BN 的 beta 已提供偏置）。"
  (let* ((inner-name (concatenate 'string name "-inner"))
         (block (make-sequential :name inner-name)))
    (seq-add! block (make-conv2d channels '(3 3)
                                 :in-channels channels
                                 :padding '(1 1)
                                 :use-bias nil
                                 :name (concatenate 'string name "-c1")))
    (seq-add! block (make-batch-norm channels
                                     :name (concatenate 'string name "-bn1")))
    (seq-add! block (make-activation-layer :relu
                                           :name (concatenate 'string name "-r1")))
    (seq-add! block (make-conv2d channels '(3 3)
                                 :in-channels channels
                                 :padding '(1 1)
                                 :use-bias nil
                                 :name (concatenate 'string name "-c2")))
    (seq-add! block (make-batch-norm channels
                                     :name (concatenate 'string name "-bn2")))
    ;; ★ 关键：用 make-residual 包起来
    ;;   forward:  output = input + block-inner(input)
    ;;   backward: d_input = d_output + d_input_from_inner（自动由容器的 backward 处理）
    (make-residual block :name name)))

(defun make-resnet9-cifar (&key (n-classes 10))
  "构造 ResNet-9 mini。通道数全部保持 64，方便 residual 相加。"
  (let ((model (make-sequential :name "resnet9-cifar")))

    ;; ---- prep: 3 → 32 ----
    (seq-add! model (make-conv2d 32 '(3 3)
                                 :in-channels 3
                                 :padding '(1 1)
                                 :use-bias nil
                                 :name "prep-conv"))
    (seq-add! model (make-batch-norm 32 :name "prep-bn"))
    (seq-add! model (make-activation-layer :relu :name "prep-relu"))

    ;; ---- transition: 32 → 64 ----
    (seq-add! model (make-conv2d 64 '(3 3)
                                 :in-channels 32
                                 :padding '(1 1)
                                 :use-bias nil
                                 :name "trans-conv"))
    (seq-add! model (make-batch-norm 64 :name "trans-bn"))
    (seq-add! model (make-activation-layer :relu :name "trans-relu"))

    ;; ---- 三个 residual 块，通道数都是 64 ----
    ;; 每个块后接 ReLU（因为 residual 内部无激活），然后 MaxPool 减半分辨率。
    (seq-add! model (make-residual-block 64 "block1"))
    (seq-add! model (make-activation-layer :relu :name "block1-out"))
    (seq-add! model (make-max-pool2d 2 :name "pool1"))       ; 32→16

    (seq-add! model (make-residual-block 64 "block2"))
    (seq-add! model (make-activation-layer :relu :name "block2-out"))
    (seq-add! model (make-max-pool2d 2 :name "pool2"))       ; 16→8

    (seq-add! model (make-residual-block 64 "block3"))
    (seq-add! model (make-activation-layer :relu :name "block3-out"))
    (seq-add! model (make-max-pool2d 2 :name "pool3"))       ; 8→4

    ;; ---- 分类头 ----
    (seq-add! model (make-global-avg-pool2d :name "gap"))    ; 4×4×64 → 64
    (seq-add! model (make-dense n-classes
                                :activation :none
                                :name "fc"))                  ; 64 → 10
    model))

;;; ============================================================================
;;; 4. 评估
;;; ============================================================================

(defun evaluate-resnet9 (model test-x test-y
                          &key (batch-size 128) (n-test 2000))
  "在测试集上前向，返回 (values correct total accuracy)。"
  (let ((correct 0) (total 0) (start 0))
    (with-training nil                       ; ★ 推理模式：BN 用 running stats
      (loop while (< start n-test) do
        (let* ((end (min (+ start batch-size) n-test))
               (x (cifar-slice-images test-x start end))
               (pred (forward model x)))
          (incf correct (cifar-accuracy pred test-y start))
          (incf total (- end start))
          (setf start end)
          (clear-step-caches! model))))
    (values correct total
            (coerce (/ correct total) 'double-float))))

;;; ============================================================================
;;; 5. 训练循环
;;; ============================================================================

(defun train-resnet9-cifar (&key (epochs 10)
                                  (batch-size 128)
                                  (lr 1.0d-1)              ; SGD 常用 0.1 起步
                                  (momentum 0.9d0)
                                  (weight-decay 5.0d-4)
                                  (nesterov t)
                                  (n-train 10000)
                                  (n-test 2000)
                                  (log-every 20))
  "训练 ResNet-9 on CIFAR-10。
   优化器：SGD + Momentum + Nesterov + Weight Decay
   调度器：Cosine Annealing（每 epoch 调一次，T_max=epochs）"
  (multiple-value-bind (train-x train-y test-x test-y)
      (load-cifar10 :n-train n-train :n-test n-test)

    (let* ((model     (make-resnet9-cifar))
           (opt       (make-sgd :lr lr
                                :momentum momentum
                                :nesterov nesterov
                                :weight-decay weight-decay))
           (scheduler (make-cosine-annealing-lr opt epochs
                                                :eta-min 0.0d0))
           (loss-fn   (make-ce-loss :reduction :mean))
           (n-batches (floor n-train batch-size))
           (t0 (get-internal-real-time)))

      ;; ★ 触发所有延迟初始化（dense 需要一次前向推断 in-dim）
      (build-model model (vt-zeros (list 1 3 32 32)))
      (set-model-training! model t)

      (format t "~%============================================================~%")
      (format t "  ResNet-9 on CIFAR-10~%")
      (format t "============================================================~%")
      (format t "  训练样本: ~a   测试样本: ~a~%" n-train n-test)
      (format t "  batch-size: ~a   epochs: ~a~%" batch-size epochs)
      (format t "  优化器: SGD(lr=~a momentum=~a nesterov=~a wd=~a)~%"
              lr momentum nesterov weight-decay)
      (format t "  调度器: CosineAnnealingLR(T_max=~a eta_min=0)~%" epochs)
      (format t "  参数量: ~a~%" (param-count model))
      (format t "============================================================~%~%")

      (dotimes (epoch epochs)
        (let ((epoch-loss 0.0d0)
              (epoch-correct 0)
              (t-epoch (get-internal-real-time)))
          (dotimes (b n-batches)
            (let* ((start (* b batch-size))
                   (end   (+ start batch-size))
                   (x (cifar-slice-images train-x start end))
                   (y (cifar-slice-labels train-y start end)))

              (zero-grad! model)
              (let* ((pred (forward model x))
                     (loss (compute-loss loss-fn pred y))
                     (grad (compute-loss-gradient loss-fn pred y)))
                (backward model grad)
                (optimizer-step opt (params model) (grads model))
                (incf epoch-loss (coerce (vt-item loss) 'double-float))
                (incf epoch-correct (cifar-accuracy pred train-y start))))

            (when (zerop (mod (1+ b) log-every))
              (format t "  [e~a b~a/~a] loss=~,4f acc=~,4f ~,1fs~%"
                      (1+ epoch) (1+ b) n-batches
                      (/ epoch-loss (1+ b))
                      (/ (float epoch-correct)
                         (float (* (1+ b) batch-size)))
                      (/ (- (get-internal-real-time) t-epoch)
                         (coerce internal-time-units-per-second 'double-float)))))

          ;; ★ 每个 epoch 结束调一次 scheduler-step!
          (scheduler-step! scheduler)

          (format t "Epoch ~a/~a  lr=~,5f  loss=~,4f  train=~,4f  (~,1fs)~%"
                  (1+ epoch) epochs
                  (scheduler-get-lr scheduler)
                  (/ epoch-loss n-batches)
                  (/ (float epoch-correct) (float n-train))
                  (/ (- (get-internal-real-time) t-epoch)
                     (coerce internal-time-units-per-second 'double-float)))))

      ;; ---- 测试 ----
      (multiple-value-bind (c tt a) (evaluate-resnet9 model test-x test-y
                                                      :batch-size batch-size
                                                      :n-test n-test)
        (declare (ignore tt))
        (format t "~%============================================================~%")
        (format t "  测试准确率: ~a / ~a = ~,2f%~%" c n-test (* 100.0 a))
        (format t "  总耗时: ~,1fs~%"
                (/ (- (get-internal-real-time) t0)
                   (coerce internal-time-units-per-second 'double-float)))
        (format t "============================================================~%"))
      model)))
