;;;; mnist-common.lisp
;;;; MNIST 数据加载 + 共享工具。所有 demo 都依赖此文件。
;;;; 加载顺序：先加载本文件，再加载 mnist-demos.lisp。

(eval-when (:compile-toplevel :load-toplevel :execute)
  (ql:quickload '(:clnn :chipz)))
(in-package :clnn)
(in-package :nn)

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
    (clvt::vt-from-sequence image :dtype :float64)))

(defun mnist-read-and-normalize-label (data offset)
  (let ((target (make-array 10 :element-type 'double-float
                               :initial-element 0.0d0))
        (category (aref data offset)))
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
            (with-open-file (f images-path :element-type '(unsigned-byte 8))
              (let ((images-data (chipz:decompress nil 'chipz:gzip f)))
                (loop for i from 0 below n-images
                      for offset = (+ 16 (* i 28 28))
                      collect (mnist-read-and-normalize-image
                               images-data offset)))))
          (labels (with-open-file (f labels-path :element-type '(unsigned-byte 8))
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

(defun mnist-images-4d (start end batch-size)
  "从 *mnist-traina* 取 [START, END)，reshape 成 (BATCH, 1, 28, 28)。"
  (clvt::vt-reshape (mnist-slice-batch *mnist-traina* start end)
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
         (p-data (clvt::vt-data pred)) (p-off (clvt::vt-offset pred))
         (p-rs (first (clvt::vt-strides pred))) (p-cs (second (clvt::vt-strides pred)))
         (t-data (clvt::vt-data target)) (t-off (clvt::vt-offset target))
         (t-rs (first (clvt::vt-strides target))) (t-cs (second (clvt::vt-strides target)))
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
         (data (clvt::vt-data pred)) (off (clvt::vt-offset pred))
         (rs (first (clvt::vt-strides pred))) (cs (second (clvt::vt-strides pred)))
         (correct 0))
    (dotimes (i batch)
      (let ((best 0) (val most-negative-double-float)
            (base (+ off (* i rs))))
        (dotimes (c n-classes)
          (let ((v (aref data (+ base (* c cs)))))
            (when (> v val) (setf val v best c))))
        (when (= best (aref labels-array (+ start i))) (incf correct))))
    correct))

;;; ============================================================
;;; 评估
;;; ============================================================
(defun evaluate-mnist-vt (model &key (batch-size 500) (n 10000))
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
        (setf start end)))
    (values correct total (coerce (/ correct total) 'double-float))))

(defun evaluate-mnist-labels (model &key (batch-size 500) (n 10000))
  "用 fixnum 标签评估。"
  (ensure-mnist-labels)
  (let ((correct 0) (total 0) (start 0))
    (loop while (< start n) do
      (let* ((end (min (+ start batch-size) n))
             (bs (- end start))
             (x (mnist-slice-batch *mnist-testa* start end))
             (pred (forward model x)))
        (incf correct (mnist-accuracy-labels pred *mnist-test-labels* start))
        (incf total bs)
        (setf start end)))
    (values correct total (coerce (/ correct total) 'double-float))))









;;;; mnist-demos.lisp
;;;; 4 个 MNIST demo。依赖 mnist-common.lisp 已加载。
(in-package :nn)

;;; ============================================================
;;; DEMO 1: Dense 网络 (MSE)
;;; ============================================================
(defun make-dense-mnist (&key (hidden1 256) (hidden2 128))
  (let ((m (make-sequential)))
    (seq-add! m (make-dense hidden1 :in-dim 784 :activation :relu))
    (seq-add! m (make-dense hidden2 :activation :relu))
    (seq-add! m (make-dense 10 :activation :none))
    m))

(defun train-dense-mnist (&key (epochs 5) (batch-size 128) (lr 1e-3)
                                (hidden1 256) (hidden2 128))
  (ensure-mnist-data)
  (let* ((model (make-dense-mnist :hidden1 hidden1 :hidden2 hidden2))
         (loss-fn (make-mse-loss :reduction :mean))
         (opt (make-adam :lr lr))
         (n-train 60000)
         (n-batches (floor n-train batch-size))
         (t0 (get-internal-real-time)))
    (format t "~%=== Dense MNIST (MSE) ===~%")
    (format t "模型: 784→~a→~a→10  lr=~a  batch=~a  epochs=~a~%~%"
            hidden1 hidden2 lr batch-size epochs)
    (dotimes (epoch epochs)
      (let ((epoch-loss 0.0d0) (epoch-correct 0)
            (t-epoch (get-internal-real-time)))
        (dotimes (b n-batches)
          (let* ((start (* b batch-size))
                 (end (+ start batch-size))
                 (x (mnist-slice-batch *mnist-traina* start end))
                 (y (mnist-slice-batch *mnist-trainl* start end)))
            (zero-grad! model)
            (let* ((pred (forward model x))
                   (loss (compute-loss loss-fn pred y))
                   (grad (compute-loss-gradient loss-fn pred y)))
              (backward model grad)
              (optimizer-step opt (params model) (grads model))
              (incf epoch-loss (vt-item loss))
              (incf epoch-correct (mnist-accuracy-vt pred y)))))
        (format t "Epoch ~a/~a  loss=~,6f  train=~,4f  (~,1fs)~%"
                (1+ epoch) epochs
                (/ epoch-loss n-batches) (/ epoch-correct n-train)
                (/ (- (get-internal-real-time) t-epoch)
                   (coerce internal-time-units-per-second 'double-float)))))
    (multiple-value-bind (c tt a) (evaluate-mnist-vt model)
      (format t "~%测试准确率: ~a / ~a = ~,2f%~%" c tt (* 100.0 a)))
    (format t "总耗时: ~,1f 秒~%"
            (/ (- (get-internal-real-time) t0)
               (coerce internal-time-units-per-second 'double-float)))
    model))

;;; ============================================================
;;; DEMO 2: CNN 32/64 (MSE)
;;; ============================================================
(defun make-cnn-mnist ()
  (let ((m (make-sequential)))
    (seq-add! m (make-conv2d 32 '(3 3) :in-channels 1
                              :stride '(1 1) :padding '(1 1)))
    (seq-add! m (make-activation-layer :relu))
    (seq-add! m (make-max-pool2d 2))
    (seq-add! m (make-conv2d 64 '(3 3) :in-channels 32
                              :stride '(1 1) :padding '(1 1)))
    (seq-add! m (make-activation-layer :relu))
    (seq-add! m (make-max-pool2d 2))
    (seq-add! m (make-flatten))
    (seq-add! m (make-dense 128 :activation :relu))
    (seq-add! m (make-dense 10 :activation :none))
    m))

(defun train-cnn-mnist (&key (epochs 10) (batch-size 128) (lr 1e-3)
                              (log-every 50))
  (ensure-mnist-data)
  (let* ((model (make-cnn-mnist))
         (loss-fn (make-mse-loss :reduction :mean))
         (opt (make-adam :lr lr))
         (n-train 60000)
         (n-batches (floor n-train batch-size))
         (t0 (get-internal-real-time)))
    (format t "~%=== CNN MNIST 32/64 (MSE) ===~%")
    (format t "lr=~a  batch=~a  epochs=~a~%~%" lr batch-size epochs)
    (dotimes (epoch epochs)
      (let ((epoch-loss 0.0d0) (epoch-correct 0)
            (t-epoch (get-internal-real-time)))
        (dotimes (b n-batches)
          (let* ((start (* b batch-size))
                 (end (+ start batch-size))
                 (x (mnist-images-4d start end batch-size))
                 (y (mnist-slice-batch *mnist-trainl* start end)))
            (zero-grad! model)
            (let* ((pred (forward model x))
                   (loss (compute-loss loss-fn pred y))
                   (grad (compute-loss-gradient loss-fn pred y)))
              (backward model grad)
              (optimizer-step opt (params model) (grads model))
              (incf epoch-loss (vt-item loss))
              (incf epoch-correct (mnist-accuracy-vt pred y))))
          (when (zerop (mod (1+ b) log-every))
            (format t "  [e~a b~a/~a]  loss=~,4f  acc=~,4f  ~,1fs~%"
                    (1+ epoch) (1+ b) n-batches
                    (/ epoch-loss (1+ b))
                    (/ epoch-correct (* (1+ b) batch-size))
                    (/ (- (get-internal-real-time) t-epoch)
                       (coerce internal-time-units-per-second 'double-float)))))
        (format t "Epoch ~a/~a  loss=~,6f  train=~,4f  (~,1fs)~%"
                (1+ epoch) epochs
                (/ epoch-loss n-batches) (/ epoch-correct n-train)
                (/ (- (get-internal-real-time) t-epoch)
                   (coerce internal-time-units-per-second 'double-float)))))
    (multiple-value-bind (c tt a) (evaluate-mnist-vt model)
      (format t "~%测试准确率: ~a / ~a = ~,2f%~%" c tt (* 100.0 a)))
    (format t "总耗时: ~,1f 秒~%"
            (/ (- (get-internal-real-time) t0)
               (coerce internal-time-units-per-second 'double-float)))
    model))

;;; ============================================================
;;; DEMO 3: CNN 8/16 (CE)
;;; ============================================================
(defun make-cnn-mnist-small ()
  (let ((m (make-sequential)))
    (seq-add! m (make-conv2d 8 '(3 3) :in-channels 1
                              :stride '(1 1) :padding '(1 1)))
    (seq-add! m (make-activation-layer :relu))
    (seq-add! m (make-max-pool2d 2))
    (seq-add! m (make-conv2d 16 '(3 3) :in-channels 8
                              :stride '(1 1) :padding '(1 1)))
    (seq-add! m (make-activation-layer :relu))
    (seq-add! m (make-max-pool2d 2))
    (seq-add! m (make-flatten))
    (seq-add! m (make-dense 32 :activation :relu))
    (seq-add! m (make-dense 10 :activation :none))
    m))

(defun train-cnn-mnist-small (&key (epochs 5) (batch-size 128) (lr 1e-3)
                                     (n-train 60000) (log-every 100))
  (ensure-mnist-data)
  (ensure-mnist-labels)
  (let* ((model (make-cnn-mnist-small))
         (loss-fn (make-ce-loss :reduction :mean))
         (opt (make-adam :lr lr))
         (n-batches (floor n-train batch-size))
         (t0 (get-internal-real-time)))
    (format t "~%=== CNN Small 8/16 (CE) ===~%")
    (format t "样本=~a  batch=~a  lr=~a  epochs=~a~%~%"
            n-train batch-size lr epochs)
    (dotimes (epoch epochs)
      (let ((epoch-loss 0.0d0) (epoch-correct 0)
            (t-epoch (get-internal-real-time)))
        (dotimes (b n-batches)
          (let* ((start (* b batch-size))
                 (end (+ start batch-size))
                 (x (mnist-images-4d start end batch-size))
                 (y (mnist-slice-labels *mnist-train-labels* start end)))
            (zero-grad! model)
            (let* ((pred (forward model x))
                   (loss (compute-loss loss-fn pred y))
                   (grad (compute-loss-gradient loss-fn pred y)))
              (backward model grad)
              (optimizer-step opt (params model) (grads model))
              (incf epoch-loss (vt-item loss))
              (incf epoch-correct
                    (mnist-accuracy-labels pred *mnist-train-labels* start)))))
        (format t "Epoch ~a/~a  loss=~,6f  train=~,4f  (~,1fs)~%"
                (1+ epoch) epochs
                (/ epoch-loss n-batches) (/ epoch-correct n-train)
                (/ (- (get-internal-real-time) t-epoch)
                   (coerce internal-time-units-per-second 'double-float)))))
    (multiple-value-bind (c tt a) (evaluate-mnist-labels model)
      (format t "~%测试准确率: ~a / ~a = ~,2f%~%" c tt (* 100.0 a)))
    (format t "总耗时: ~,1f 秒~%"
            (/ (- (get-internal-real-time) t0)
               (coerce internal-time-units-per-second 'double-float)))
    model))

;;; ============================================================
;;; DEMO 4: CNN + BN 对比 (Flatten vs GAP)
;;; ============================================================
(defun make-cnn-bn-flatten ()
  (let ((m (make-sequential)))
    (seq-add! m (make-conv2d 8 '(3 3) :in-channels 1 :padding '(1 1)))
    (seq-add! m (make-batch-norm 8))
    (seq-add! m (make-activation-layer :relu))
    (seq-add! m (make-max-pool2d 2))
    (seq-add! m (make-conv2d 16 '(3 3) :in-channels 8 :padding '(1 1)))
    (seq-add! m (make-batch-norm 16))
    (seq-add! m (make-activation-layer :relu))
    (seq-add! m (make-max-pool2d 2))
    (seq-add! m (make-flatten))
    (seq-add! m (make-dense 10 :activation :none))
    m))

(defun make-cnn-bn-gap ()
  (let ((m (make-sequential)))
    (seq-add! m (make-conv2d 8 '(3 3) :in-channels 1 :padding '(1 1)))
    (seq-add! m (make-batch-norm 8))
    (seq-add! m (make-activation-layer :relu))
    (seq-add! m (make-max-pool2d 2))
    (seq-add! m (make-conv2d 16 '(3 3) :in-channels 8 :padding '(1 1)))
    (seq-add! m (make-batch-norm 16))
    (seq-add! m (make-activation-layer :relu))
    (seq-add! m (make-max-pool2d 2))
    (seq-add! m (make-global-avg-pool2d))
    (seq-add! m (make-dense 10 :activation :none))
    m))

(defun describe-cnn-layer (l)
  (typecase l
    (conv2d (format nil "Conv(~a→~a,~ax~a)"
                    (conv-in-channels l) (conv-out-channels l)
                    (first (conv-kernel-size l)) (second (conv-kernel-size l))))
    (batch-norm (format nil "BN(~a)" (bn-num-features l)))
    (activation-layer (string-upcase (symbol-name (activation-kind l))))
    (max-pool2d (format nil "MaxPool(~a)" (first (pool-kernel-size l))))
    (global-avg-pool2d "GAP")
    (flatten "Flatten")
    (dense (format nil "Dense(~a→~a)" (or (dense-in-dim l) '?) (dense-out-dim l)))
    (t (class-name (class-of l)))))

(defun describe-cnn-model (model)
  (format nil "~{~a~^ → ~}" (mapcar #'describe-cnn-layer (seq-layers model))))

(defun train-cnn-and-eval (model &key (n-train 20000) (epochs 3)
                                    (batch-size 128) (lr 1e-3)
                                    (label "model"))
  (ensure-mnist-data)
  (ensure-mnist-labels)
  (let* ((loss-fn (make-ce-loss :reduction :mean))
         (opt (make-adam :lr lr))
         (n-batches (floor n-train batch-size))
         (train-accs '())
         (t-total (get-internal-real-time)))
    (dotimes (epoch epochs)
      (let ((epoch-loss 0.0d0) (epoch-correct 0)
            (t-epoch (get-internal-real-time)))
        (dotimes (b n-batches)
          (let* ((start (* b batch-size))
                 (end (+ start batch-size))
                 (x (mnist-images-4d start end batch-size))
                 (y (mnist-slice-labels *mnist-train-labels* start end)))
            (zero-grad! model)
            (let* ((pred (forward model x))
                   (loss (compute-loss loss-fn pred y))
                   (grad (compute-loss-gradient loss-fn pred y)))
              (backward model grad)
              (optimizer-step opt (params model) (grads model))
              (incf epoch-loss (vt-item loss))
              (incf epoch-correct
                    (mnist-accuracy-labels pred *mnist-train-labels* start)))))
        (let ((acc (/ epoch-correct n-train)))
          (push acc train-accs)
          (format t "  [~a] epoch ~a/~a  loss=~,4f  train=~,4f  (~,1fs)~%"
                  label (1+ epoch) epochs
                  (/ epoch-loss n-batches) acc
                  (/ (- (get-internal-real-time) t-epoch)
                     (coerce internal-time-units-per-second 'double-float))))))
    (set-model-training! model nil)
    (multiple-value-bind (c tt a) (evaluate-mnist-labels model)
      (let ((total-time (/ (- (get-internal-real-time) t-total)
                           (coerce internal-time-units-per-second 'double-float))))
        (format t "  [~a] 测试准确率: ~,2f%  总耗时 ~,1fs~%"
                label (* 100.0 a) total-time)
        (values model (nreverse train-accs) a total-time)))))

(defun compare-cnn-tails (&key (n-train 20000) (epochs 3)
                                  (batch-size 128) (lr 1e-3))
  (format t "~%============================================================~%")
  (format t "  CNN 尾部对比  n=~a  epochs=~a  batch=~a  lr=~a~%"
          n-train epochs batch-size lr)
  (format t "============================================================~%")
  (let* ((m-flatten (make-cnn-bn-flatten))
         (m-gap (make-cnn-bn-gap)))
    (format t "~%模型 A (Flatten): ~a~%" (describe-cnn-model m-flatten))
    (format t "模型 B (GAP):     ~a~%~%" (describe-cnn-model m-gap))
    (format t "--- 训练 A ---~%")
    (multiple-value-bind (model-a accs-a test-a time-a)
        (train-cnn-and-eval m-flatten :n-train n-train :epochs epochs
                             :batch-size batch-size :lr lr :label "Flatten")
      (format t "~%--- 训练 B ---~%")
      (multiple-value-bind (model-b accs-b test-b time-b)
          (train-cnn-and-eval m-gap :n-train n-train :epochs epochs
                               :batch-size batch-size :lr lr :label "GAP")
        (format t "~%============================================================~%")
        (format t "  汇总~%")
        (format t "============================================================~%")
        (format t "~24a  ~12a  ~12a~%" "指标" "Flatten" "GAP")
        (dotimes (i epochs)
          (format t "~24a  ~11,4f%  ~11,4f%~%"
                  (format nil "Epoch ~a train-acc" (1+ i))
                  (* 100.0 (nth i accs-a)) (* 100.0 (nth i accs-b))))
        (format t "~24a  ~11,2f%  ~11,2f%~%" "测试准确率"
                (* 100.0 test-a) (* 100.0 test-b))
        (format t "~24a  ~10,1fs  ~10,1fs~%" "总耗时" time-a time-b)
        (format t "~24a  ~10a  ~10a~%" "参数量"
                (param-count model-a) (param-count model-b))
        (values model-a model-b accs-a accs-b test-a test-b)))))
