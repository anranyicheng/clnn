;;;; smoke-mnist-convergence.lisp — 真实 MNIST 子集上的端到端收敛冒烟测试
;;;;
;;;; 目的：验证 batchnorm（训练态统计/反向公式）、ce-loss、优化器等改动
;;;; 没有破坏实际训练。用 4000 张训练图、240 步、一个 9k 参数的小 CNN，
;;;; 断言 loss 显著下降且测试集准确率 > 80%。
;;;;
;;;; 需要 chipz 解压 .gz（quicklisp 可自动安装）。
;;;;
;;;; 用法:
;;;;   sbcl --dynamic-space-size 6144 --non-interactive \
;;;;        --load tests/smoke-mnist-convergence.lisp
(require :asdf)
(ql:quickload '(:clnn :chipz))
(in-package :nn)

;;; ---- 读 MNIST（与 examples/mnist.lisp 类似的解压逻辑，但只取子集）----
(defun mnist-path (f)
  (asdf:system-relative-pathname "clnn" (concatenate 'string "mnist-data/" f)))

(defun gunzip-bytes (path)
  (with-open-file (f path :direction :input :element-type '(unsigned-byte 8))
    (chipz:decompress nil 'chipz:gzip f)))

(defun be-int (bytes off n)
  (let ((v 0))
    (dotimes (i n v) (setf v (+ (* v 256) (aref bytes (+ off i)))))))

(defun load-images (path limit)
  (let* ((b (gunzip-bytes path))
         (n (be-int b 4 4))
         (rows (be-int b 8 4))
         (cols (be-int b 12 4))
         (n (min n limit))
         (data (make-array (* n rows cols) :element-type 'double-float)))
    (dotimes (i (length data))
      (setf (aref data i) (/ (aref b (+ 16 i)) 255.0d0)))
    (values (vt-reshape (vt-from-array data :fast t) (list n 1 rows cols)) n)))

(defun load-labels (path limit)
  (let* ((b (gunzip-bytes path))
         (n (min (be-int b 4 4) limit))
         (a (make-array n :element-type '(signed-byte 64))))
    (dotimes (i n) (setf (aref a i) (aref b (+ 8 i))))
    (vt-from-array a :dtype :int64)))

(defun sub-batch (x start size)
  (vt-slice x (list start (+ start size))))

(format t "~%=== 载入 MNIST 子集 ===~%")
(multiple-value-bind (xall n) (load-images (mnist-path "train-images-idx3-ubyte.gz") 4000)
  (let* ((yall (load-labels (mnist-path "train-labels-idx1-ubyte.gz") 4000))
         (vx  (load-images (mnist-path "t10k-images-idx3-ubyte.gz") 1000))
         (vy  (load-labels (mnist-path "t10k-labels-idx1-ubyte.gz") 1000))
         (m (make-sequential))
         (loss (make-ce-loss))
         (opt (make-adam :lr 1e-3))
         (batch 64)
         (steps 240))
    (declare (ignore n))
    (format t "训练样本 ~a, 测试样本 1000~%" (first (vt-shape xall)))
    ;; 小型 CNN：conv -> relu -> bn -> pool -> conv -> relu -> bn -> pool -> flatten -> dense
    (seq-add! m (make-conv2d 8 3 :in-channels 1 :padding 1))
    (seq-add! m (make-activation-layer :relu))
    (seq-add! m (make-batch-norm 8))
    (seq-add! m (make-max-pool2d 2))
    (seq-add! m (make-conv2d 16 3 :in-channels 8 :padding 1))
    (seq-add! m (make-activation-layer :relu))
    (seq-add! m (make-batch-norm 16))
    (seq-add! m (make-max-pool2d 2))
    (seq-add! m (make-flatten))
    (seq-add! m (make-dense 10 :in-dim 784 :activation :none))

    (set-global-training! t)
    (build-model m (vt-zeros (list 1 1 28 28)))
    (format t "参数量 = ~a~%" (param-count m))

    (let ((losses '()))
      (dotimes (s steps)
        (let* ((start (mod (* s batch) (- (first (vt-shape xall)) batch)))
               (xb (sub-batch xall start batch))
               (yb (vt-slice yall (list start (+ start batch)))))
          (zero-grad! m)
          (let ((out (forward m xb)))
            (push (vt-item (compute-loss loss out yb)) losses)
            (backward m (compute-loss-gradient loss out yb))
            (model-update! m opt))))
      ;; LOSSES 是 push 出来的（最新在前），先转成时间顺序再取首尾窗口
      (let* ((chron (reverse losses))
             (n (length chron))
             (first10 (/ (reduce #'+ (subseq chron 0 10)) 10.0d0))
             (last10  (/ (reduce #'+ (subseq chron (- n 10))) 10.0d0)))
        (format t "~%训练 loss: 前 10 步均值 ~,4f -> 后 10 步均值 ~,4f~%" first10 last10)
        (assert (< last10 (* 0.7d0 first10)) () "loss 未显著下降")))

    ;; 测试集准确率（推理模式）
    (set-model-training! m nil)
    (let ((correct 0) (total 0))
      (dotimes (s 15)
        (let* ((start (* s 64))
               (xb (sub-batch vx start 64))
               (yb (sub-batch vy start 64))
               (pred (forward m xb))
               (rows (vt-to-list pred)))
          (dotimes (i 64)
            (let* ((row (nth i rows))
                   (best (let ((bi 0) (bv (first row)))
                           (loop for v in row for j from 0
                                 when (> v bv) do (setf bv v bi j))
                           bi))
                   (want (coerce (vt-ref yb i) 'fixnum)))
              (incf total)
              (when (= best want) (incf correct))))))
      (format t "测试准确率 = ~a/~a = ~,2f%~%"
              correct total (* 100.0d0 (/ correct total)))
      (assert (> (/ correct total) 0.80d0) () "准确率低于 80%"))
    (format t "~%=== MNIST 冒烟测试通过 ===~%")
    (finish-output)))
