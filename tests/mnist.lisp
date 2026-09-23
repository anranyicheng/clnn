
;;;; mnist-clnn.lisp

;;; ============================================================
;;; 1. 模型
;;; ============================================================
(defun make-mnist-model (&key (hidden1 256) (hidden2 128))
  (let ((m (nn:make-sequential)))
    (nn:seq-add! m (nn:make-dense hidden1 :in-dim 784 :activation :relu))
    (nn:seq-add! m (nn:make-dense hidden2 :activation :relu))
    (nn:seq-add! m (nn:make-dense 10 :activation :none))
    m))
(defun mnist-file-path (filename)
  (asdf:system-relative-pathname
   "clnn" (concatenate 'string "mnist-data/" filename)))

(defun mnist-read-and-normalize-image (data offset)
  (let ((image (make-array (* 28 28)
                           :element-type 'double-float
                           :initial-element 0.0d0)))
    (dotimes (i (* 28 28))
      (setf (aref image i)
	    (/ (aref data (+ offset i)) 255.0d0)))
    (clvt::vt-from-sequence  image :dtype :float64)))

(defun mnist-read-and-normalize-label (data offset)
  (let ((target (make-array 10
                            :element-type 'double-float
                            :initial-element 0.0d0))   ; 设置初始值都为-1，
	(category (aref data offset)))
    ;; 将标记好的图片数字结果记录，形式如 #(-1.0 -1.0 -1.0 -1.0 -1.0 1.0 -1.0 -1.0 -1.0 -1.0) 表示数字为5
    (setf (aref target category) 1.0d0)
    (clvt::vt-from-sequence target :dtype :float64)))

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
	    (with-open-file (f images-path
                               :element-type '(unsigned-byte 8))
	      ;; 这里解压并读取二进制数据，得到#(0 0 8 3 0 0 234 96 0 0 0 28 0 0 0 28 0 0 0 0 0......0)
              (let ((images-data (chipz:decompress nil 'chipz:gzip f)))  
                (loop for i from 0 below n-images
                      for offset = (+ 16 (* i 28 28))
                      collect (mnist-read-and-normalize-image
			       images-data offset)))))
          (labels (with-open-file (f labels-path
                                    :element-type '(unsigned-byte 8))
                    (let ((labels-data (chipz:decompress nil 'chipz:gzip f)))
                      (loop for i from 0 below n-images
                            collect (mnist-read-and-normalize-label
				     labels-data (+ 8 i)))))))
      (values images labels))))  ;; 返回的都是列表形式的数据, 其中列表的每一个元素都是向量
(defun initial-data ()
  (defparameter *traina* (clvt::vt-zeros '(60000 784) :dtype :float64))
  (defparameter *trainl* (clvt::vt-zeros '(60000 10) :dtype :float64))
  (defparameter *testa* (clvt::vt-zeros '(10000 784) :dtype :float64))
  (defparameter *testl* (clvt::vt-zeros '(10000 10) :dtype :float64))
  (multiple-value-bind (image label)
      (mnist-load :train)
    (let* ((count 0))
      (dolist (var image)
	(setf (clvt::vt-slice *traina* (list count)) var)
	(incf count)))
    (let* ((count 0))
      (dolist (var label)
	(setf (clvt::vt-slice *trainl* (list count)) var)
	(incf count))))
  (multiple-value-bind (image label)
      (mnist-load :test)
    (let* ((count 0))
      (dolist (var image)
	(setf (clvt::vt-slice *testa* (list count)) var)
	(incf count)))
    (let* ((count 0))
      (dolist (var label)
	(setf (clvt::vt-slice *testl* (list count)) var)
	(incf count)))))

;;; ============================================================
;;; 2. 数据切分与评估
;;; ============================================================
(defun slice-batch (tensor start end)
  "取 TENSOR 的 [START, END) 行，返回 (batch, ...) 张量。"
  (clvt::vt-slice tensor (list start end) (list :all)))

(defun batch-accuracy (pred target &key (n-classes 10))
  (let ((batch (first (clvt::vt-shape pred)))
        (correct 0))
    (dotimes (i batch)
      (let ((pred-best 0)
	    (pred-val most-negative-double-float)
	    (tgt-best 0)  (tgt-val most-negative-double-float))
        (dotimes (c n-classes)
          (let ((pv (coerce (clvt::vt-ref pred i c) 'double-float))
                (tv (coerce (clvt::vt-ref target i c) 'double-float)))
            (when (> pv pred-val) (setf pred-val pv pred-best c))
            (when (> tv tgt-val) (setf tgt-val tv tgt-best c))))
        (when (= pred-best tgt-best) (incf correct))))
    correct))

(defun evaluate-mnist (model &key (batch-size 500))
  (let ((n 10000) (correct 0) (total 0))
    (loop for start from 0 below n by batch-size
          for end = (min (+ start batch-size) n)
          for x = (slice-batch *testa* start end)
          for y = (slice-batch *testl* start end)
          for pred = (nn:forward model x)
          do (incf correct (batch-accuracy pred y))
             (incf total (- end start)))
    (values correct total (coerce (/ correct total) 'double-float))))

;;; ============================================================
;;; 3. 训练
;;; ============================================================
(defun train-mnist (&key (epochs 5) (batch-size 128) (lr 1e-3)
                      (hidden1 256) (hidden2 128))
  (let* ((model    (make-mnist-model :hidden1 hidden1 :hidden2 hidden2))
         (loss-fn  (nn:make-mse-loss :reduction :mean))
         (opt      (nn:make-adam :lr lr))
         (n-train  60000)
         (n-batches (floor n-train batch-size))
         (t0 (get-internal-real-time)))
    (format t "~%=== MNIST 训练开始 ===~%")
    (format t "模型: 784 → ~a → ~a → 10~%" hidden1 hidden2)
    (format t "优化器: Adam  lr=~a~%" lr)
    (format t "批大小: ~a  每 epoch ~a 批  共 ~a epoch~%~%"
            batch-size n-batches epochs)

    (dotimes (epoch epochs)
      (let ((epoch-loss 0.0d0)
            (epoch-correct 0)
            (t-epoch (get-internal-real-time)))
        (dotimes (b n-batches)
          (let* ((start (* b batch-size))
                 (end   (+ start batch-size))
                 (x (slice-batch *traina* start end))
                 (y (slice-batch *trainl* start end)))
            (nn:zero-grad! model)
            (let* ((pred (nn:forward model x))
                   (loss (nn:compute-loss loss-fn pred y))
                   (grad (nn:compute-loss-gradient loss-fn pred y)))
              (nn:backward model grad)
              (nn:optimizer-step opt (nn:params model) (nn:grads model))
              (incf epoch-loss    (clvt::vt-item loss))
              (incf epoch-correct (batch-accuracy pred y)))))
        (let ((elapsed (/ (- (get-internal-real-time) t-epoch)
                          (coerce internal-time-units-per-second 'double-float))))
          (format t "Epoch ~a/~a  loss=~,6f  acc=~,4f  (~,1fs)~%"
                  (1+ epoch) epochs
                  (/ epoch-loss n-batches)
                  (/ epoch-correct n-train)
                  elapsed))))

    (multiple-value-bind (correct total acc) (evaluate-mnist model)
      (format t "~%测试集准确率: ~a / ~a = ~,2f%~%" correct total (* 100.0 acc)))
    (let ((total-time (/ (- (get-internal-real-time) t0)
                         (coerce internal-time-units-per-second 'double-float))))
      (format t "总训练时间: ~,1f 秒~%" total-time))
    model))

(defun run-mnist (&key (epochs 5) (batch-size 128) (lr 1e-3))
  (format t "========加载数据======")
  (initial-data)
  (format t "========加载完成======")
  (train-mnist :epochs epochs :batch-size batch-size :lr lr))

























;;;; mnist-cnn.lisp
;;;; 依赖：clnn 已加载，*traina* / *trainl* / *testa* / *testl* 已准备好。

;;; ============================================================
;;; 1. 数据辅助
;;; ============================================================
(defun slice-batch (tensor start end)
  "取 TENSOR 的 [START, END) 行。"
  (clvt::vt-slice tensor (list start end) (list :all)))

(defun images-to-4d (x batch-size)
  "把 (batch, 784) reshape 成 (batch, 1, 28, 28)。"
  (clvt::vt-reshape x (list batch-size 1 28 28)))

;;; ============================================================
;;; 2. 快速准确率（绕过 vt-ref 泛型 dispatch）
;;; ============================================================
(defun batch-accuracy-fast (pred target &key (n-classes 10))
  "PRED / TARGET 形状均为 (batch, n-classes)。"
  (let* ((batch   (first (clvt::vt-shape pred)))
         (p-data  (clvt::vt-data pred))
         (p-off   (clvt::vt-offset pred))
         (p-str   (clvt::vt-strides pred))
         (p-rs    (first p-str))
         (p-cs    (second p-str))
         (t-data  (clvt::vt-data target))
         (t-off   (clvt::vt-offset target))
         (t-str   (clvt::vt-strides target))
         (t-rs    (first t-str))
         (t-cs    (second t-str))
         (correct 0))
    (declare (type fixnum batch correct p-rs p-cs t-rs t-cs))
    (dotimes (i batch)
      (let ((pred-best 0) (pred-val most-negative-double-float)
            (tgt-best  0) (tgt-val  most-negative-double-float)
            (p-base (+ p-off (the fixnum (* i p-rs))))
            (t-base (+ t-off (the fixnum (* i t-rs)))))
        (declare (type fixnum pred-best tgt-best p-base t-base))
        (dotimes (c n-classes)
          (let ((pv (aref p-data (+ p-base (the fixnum (* c p-cs)))))
                (tv (aref t-data (+ t-base (the fixnum (* c t-cs))))))
            (when (> pv pred-val) (setf pred-val pv pred-best c))
            (when (> tv tgt-val) (setf tgt-val tv tgt-best c))))
        (when (= pred-best tgt-best) (incf correct))))
    correct))

;;; ============================================================
;;; 3. 模型
;;; ============================================================
(defun make-mnist-cnn ()
  "Conv(1→32, 3x3, pad=1) → ReLU → MaxPool(2)
   Conv(32→64, 3x3, pad=1) → ReLU → MaxPool(2)
   Flatten → Dense(128) → ReLU → Dense(10)"
  (let ((m (nn:make-sequential)))
    ;; Conv 1: 1 → 32, 28x28 保持
    (nn:seq-add! m (nn:make-conv2d 32 '(3 3)
                                    :in-channels 1
                                    :stride '(1 1)
                                    :padding '(1 1)))
    (nn:seq-add! m (nn:make-activation-layer :relu))
    (nn:seq-add! m (nn:make-max-pool2d 2))         ; 28 → 14
    ;; Conv 2: 32 → 64, 14x14 保持
    (nn:seq-add! m (nn:make-conv2d 64 '(3 3)
                                    :in-channels 32
                                    :stride '(1 1)
                                    :padding '(1 1)))
    (nn:seq-add! m (nn:make-activation-layer :relu))
    (nn:seq-add! m (nn:make-max-pool2d 2))         ; 14 → 7
    ;; Flatten: (batch, 64, 7, 7) → (batch, 3136)
    (nn:seq-add! m (nn:make-flatten))
    ;; Dense 分类头
    (nn:seq-add! m (nn:make-dense 128 :activation :relu))
    (nn:seq-add! m (nn:make-dense 10  :activation :none))
    m))

;;; ============================================================
;;; 4. 评估
;;; ============================================================
(defun evaluate-mnist-cnn (model &key (batch-size 500))
  (let ((n 10000) (correct 0) (total 0))
    (loop for start from 0 below n by batch-size
          for end = (min (+ start batch-size) n)
          for bs  = (- end start)
          for x   = (images-to-4d (slice-batch *testa* start end) bs)
          for y   = (slice-batch *testl* start end)
          for pred = (nn:forward model x)
          do (incf correct (batch-accuracy-fast pred y))
             (incf total bs))
    (values correct total (coerce (/ correct total) 'double-float))))

;;; ============================================================
;;; 5. 训练
;;; ============================================================
(defun train-mnist-cnn (&key (epochs 10) (batch-size 128) (lr 1e-3)
                               (log-every 50))
  (let* ((model    (make-mnist-cnn))
         (loss-fn  (nn:make-mse-loss :reduction :mean))
         (opt      (nn:make-adam :lr lr))
         (n-train  60000)
         (n-batches (floor n-train batch-size))
         (t0 (get-internal-real-time)))
    (format t "~%=== MNIST CNN 训练开始 ===~%")
    (format t "模型: Conv(1→32,3x3,p=1) → ReLU → Pool(2) →~%")
    (format t "      Conv(32→64,3x3,p=1) → ReLU → Pool(2) →~%")
    (format t "      Flatten → Dense(128) → ReLU → Dense(10)~%")
    (format t "优化器: Adam  lr=~a~%" lr)
    (format t "批大小: ~a  每 epoch ~a 批  共 ~a epoch~%~%"
            batch-size n-batches epochs)

    (dotimes (epoch epochs)
      (let ((epoch-loss 0.0d0)
            (epoch-correct 0)
            (t-epoch (get-internal-real-time)))
        (dotimes (b n-batches)
          (let* ((start (* b batch-size))
                 (end   (+ start batch-size))
                 (x (images-to-4d (slice-batch *traina* start end) batch-size))
                 (y (slice-batch *trainl* start end)))
            (nn:zero-grad! model)
            (let* ((pred (nn:forward model x))
                   (loss (nn:compute-loss loss-fn pred y))
                   (grad (nn:compute-loss-gradient loss-fn pred y)))
              (nn:backward model grad)
              (nn:optimizer-step opt (nn:params model) (nn:grads model))
              (incf epoch-loss    (clvt::vt-item loss))
              (incf epoch-correct (batch-accuracy-fast pred y))))
          (when (zerop (mod (1+ b) log-every))
            (format t "  [epoch ~a  batch ~a/~a]  loss=~,4f  acc=~,4f  ~,1fs~%"
                    (1+ epoch) (1+ b) n-batches
                    (/ epoch-loss (1+ b))
                    (/ epoch-correct (* (1+ b) batch-size))
                    (/ (- (get-internal-real-time) t-epoch)
                       (coerce internal-time-units-per-second 'double-float)))))
        (let ((elapsed (/ (- (get-internal-real-time) t-epoch)
                          (coerce internal-time-units-per-second 'double-float))))
          (format t "Epoch ~a/~a  loss=~,6f  train-acc=~,4f  (~,1fs)~%"
                  (1+ epoch) epochs
                  (/ epoch-loss n-batches)
                  (/ epoch-correct n-train)
                  elapsed))))

    (multiple-value-bind (correct total acc) (evaluate-mnist-cnn model)
      (format t "~%测试集准确率: ~a / ~a = ~,2f%~%" correct total (* 100.0 acc)))
    (let ((total-time (/ (- (get-internal-real-time) t0)
                         (coerce internal-time-units-per-second 'double-float))))
      (format t "总训练时间: ~,1f 秒~%" total-time))
    model))

(defun run-mnist-cnn (&key (epochs 10) (batch-size 128) (lr 1e-3))
  (unless (and (boundp '*traina*) (boundp '*trainl*)
               (boundp '*testa*)  (boundp '*testl*))
    (format t "~%数据未加载，调用 (initial-data)...~%")
    (initial-data))
  (train-mnist-cnn :epochs epochs :batch-size batch-size :lr lr))










;;;; mnist-cnn-small.lisp
;;;; 小 CNN + CE loss，几分钟内跑完 MNIST，验证 clnn 的 CNN 全流程。
;;;; 依赖：clnn 已加载，*traina* / *trainl* / *testa* / *testl* 已准备好。

;;; ============================================================
;;; 1. 标签转换（one-hot → 整数，一次性缓存）
;;; ============================================================
(defparameter *train-labels* nil)
(defparameter *test-labels* nil)

(defun one-hot-to-labels (one-hot)
  "把 (N, C) one-hot 张量转成 fixnum 简单数组。"
  (let* ((n (first (clvt::vt-shape one-hot)))
         (c (second (clvt::vt-shape one-hot)))
         (data (clvt::vt-data one-hot))
         (off  (clvt::vt-offset one-hot))
         (rs   (first  (clvt::vt-strides one-hot)))
         (cs   (second (clvt::vt-strides one-hot)))
         (labels (make-array n :element-type 'fixnum)))
    (declare (type fixnum n c))
    (dotimes (i n)
      (let ((best 0) (val most-negative-double-float)
            (base (+ off (* i rs))))
        (declare (type fixnum best base))
        (dotimes (k c)
          (let ((v (aref data (+ base (* k cs)))))
            (when (> v val) (setf val v best k))))
        (setf (aref labels i) best)))
    labels))

(defun ensure-labels-cached ()
  (unless *train-labels*
    (format t "转换 one-hot 标签（一次性）...~%")
    (setf *train-labels* (one-hot-to-labels *trainl*))
    (setf *test-labels*  (one-hot-to-labels *testl*))))

(defun slice-labels (labels-array start end)
  (clvt::vt-from-sequence
   (loop for i from start below end collect (aref labels-array i))
   :dtype :int64))

;;; ============================================================
;;; 2. 数据 reshape / 切片
;;; ============================================================
(defun slice-batch (tensor start end)
  (clvt::vt-slice tensor (list start end) (list :all)))

(defun images-to-4d (x batch-size)
  (clvt::vt-reshape x (list batch-size 1 28 28)))

;;; ============================================================
;;; 3. 快速准确率（读底层数组，绕过 vt-ref）
;;; ============================================================
(defun batch-accuracy-fast (pred labels-array start)
  "PRED 形状 (batch, 10)；LABELS-ARRAY 是 fixnum 数组；START/END 是标签数组中的索引。"
  (let* ((batch (first (clvt::vt-shape pred)))
         (data (clvt::vt-data pred))
         (off  (clvt::vt-offset pred))
         (rs   (first  (clvt::vt-strides pred)))
         (cs   (second (clvt::vt-strides pred)))
         (correct 0))
    (declare (type fixnum batch correct))
    (dotimes (i batch)
      (let ((best 0) (val most-negative-double-float)
            (base (+ off (* i rs))))
        (declare (type fixnum best base))
        (dotimes (c 10)
          (let ((v (aref data (+ base (* c cs)))))
            (when (> v val) (setf val v best c))))
        (when (= best (aref labels-array (+ start i)))
          (incf correct))))
    correct))

;;; ============================================================
;;; 4. 小模型（FLOPs 约为原 32/64 网络的 1/30）
;;; ============================================================
(defun make-mnist-cnn-small ()
  "Conv(1→8,3x3,p=1) → ReLU → Pool(2)
   Conv(8→16,3x3,p=1) → ReLU → Pool(2)
   Flatten → Dense(784→32) → ReLU → Dense(32→10)"
  (let ((m (nn:make-sequential)))
    (nn:seq-add! m (nn:make-conv2d 8 '(3 3) :in-channels 1
                                    :stride '(1 1) :padding '(1 1)))
    (nn:seq-add! m (nn:make-activation-layer :relu))
    (nn:seq-add! m (nn:make-max-pool2d 2))
    (nn:seq-add! m (nn:make-conv2d 16 '(3 3) :in-channels 8
                                    :stride '(1 1) :padding '(1 1)))
    (nn:seq-add! m (nn:make-activation-layer :relu))
    (nn:seq-add! m (nn:make-max-pool2d 2))
    (nn:seq-add! m (nn:make-flatten))
    (nn:seq-add! m (nn:make-dense 32 :activation :relu))
    (nn:seq-add! m (nn:make-dense 10 :activation :none))
    m))

;;; ============================================================
;;; 5. 训练
;;; ============================================================
(defun train-mnist-cnn-small (&key (epochs 5) (batch-size 128) (lr 1e-3)
                                     (n-train 60000) (log-every 100))
  (ensure-labels-cached)
  (let* ((model (make-mnist-cnn-small))
         (loss-fn (nn:make-ce-loss :reduction :mean))   ; ← CE 而非 MSE
         (opt (nn:make-adam :lr lr))
         (n-batches (floor n-train batch-size))
         (t0 (get-internal-real-time)))
    (format t "~%=== MNIST CNN-small 训练 (CE loss) ===~%")
    (format t "模型: Conv(1→8) → Pool → Conv(8→16) → Pool → Dense(32) → Dense(10)~%")
    (format t "样本: ~a  批: ~a  每 epoch ~a 批  ~a epoch~%"
            n-train batch-size n-batches epochs)
    (format t "优化器: Adam  lr=~a~%~%" lr)

    (dotimes (epoch epochs)
      (let ((epoch-loss 0.0d0)
            (epoch-correct 0)
            (t-epoch (get-internal-real-time)))
        (dotimes (b n-batches)
          (let* ((start (* b batch-size))
                 (end   (+ start batch-size))
                 (x (images-to-4d (slice-batch *traina* start end) batch-size))
                 (y (slice-labels *train-labels* start end)))
            (nn:zero-grad! model)
            (let* ((pred (nn:forward model x))
                   (loss (nn:compute-loss loss-fn pred y))
                   (grad (nn:compute-loss-gradient loss-fn pred y)))
              (nn:backward model grad)
              (nn:optimizer-step opt (nn:params model) (nn:grads model))
              (incf epoch-loss (clvt::vt-item loss))
              (incf epoch-correct
                    (batch-accuracy-fast pred *train-labels* start))))
          (when (zerop (mod (1+ b) log-every))
            (format t "  [epoch ~a batch ~a/~a]  loss=~,4f  acc=~,4f  ~,1fs~%"
                    (1+ epoch) (1+ b) n-batches
                    (/ epoch-loss (1+ b))
                    (/ epoch-correct (* (1+ b) batch-size))
                    (/ (- (get-internal-real-time) t-epoch)
                       (coerce internal-time-units-per-second 'double-float)))))
        (format t "Epoch ~a/~a  loss=~,6f  train-acc=~,4f  (~,1fs)~%"
                (1+ epoch) epochs
                (/ epoch-loss n-batches)
                (/ epoch-correct n-train)
                (/ (- (get-internal-real-time) t-epoch)
                   (coerce internal-time-units-per-second 'double-float)))))

    ;; 测试集评估
    (let ((correct 0) (total 0))
      (loop for start from 0 below 10000 by 500
            for end = (min (+ start 500) 10000)
            for bs  = (- end start)
            for x   = (images-to-4d (slice-batch *testa* start end) bs)
            for pred = (nn:forward model x)
            do (incf correct (batch-accuracy-fast pred *test-labels* start))
               (incf total bs))
      (format t "~%测试准确率: ~a / ~a = ~,2f%~%"
              correct total (* 100.0 (/ correct total))))

    (format t "总训练时间: ~,1f 秒~%"
            (/ (- (get-internal-real-time) t0)
               (coerce internal-time-units-per-second 'double-float)))
    model))

(defun run-mnist-cnn-small (&key (epochs 5) (batch-size 128) (lr 1e-3)
                                   (n-train 60000))
  (unless (and (boundp '*traina*) (boundp '*trainl*)
               (boundp '*testa*)  (boundp '*testl*))
    (format t "~%数据未加载，调用 (initial-data)...~%")
    (initial-data))
  (train-mnist-cnn-small :epochs epochs :batch-size batch-size
                         :lr lr :n-train n-train))
