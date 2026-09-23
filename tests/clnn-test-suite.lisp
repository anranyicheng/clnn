;;;; clnn-test-suite.lisp
;;;;
;;;; clnn 神经网络库回归测试套件。
;;;;
;;;; 用法:
;;;;   sbcl --load clnn-test-suite.lisp
;;;; 或:
;;;;   (load "clnn-test-suite.lisp")
;;;;   (nn::run-all-tests)
;;;;
;;;; 结构:
;;;;   1. 通用辅助函数
;;;;   2. 测试框架 (deftest 宏)
;;;;   3. 8 个已修复 bug 的回归测试
;;;;   4. 18 个已确认无 bug 的行为回归测试
;;;;   5. 1 个端到端集成测试
;;;;   6. run-all-tests 入口

(require :asdf)
(asdf:load-system :clnn :force t)

(in-package :nn)

;;; ============================================================
;;; 1. 通用辅助
;;; ============================================================

(defun t-flat (x) (vt-to-list (vt-flatten x)))

(defun t-inner (a b) (vt-item (vt-sum (vt-* a b))))

(defun vt-equal-p (a b)
  (and (equal (vt-shape a) (vt-shape b))
       (equal (vt-to-list a) (vt-to-list b))))

(defun approx-eq (a b &key (tol 1.0d-9))
  (< (abs (- a b)) (* tol (max 1.0d0 (abs a) (abs b)))))

(defun all-zeros-p (x)
  (cond ((null x) t)
        ((listp x) (every #'all-zeros-p x))
        ((numberp x) (= x 0.0d0))
        (t nil)))

(defun max-rel-err (a b)
  (let ((m 0.0d0))
    (loop for x in a for y in b
          for s = (max 1.0d0 (abs x) (abs y))
          do (setf m (max m (/ (abs (- x y)) s))))
    m))

(defun t-get-row (w i)
  (let ((row '()))
    (dotimes (j (second (vt-shape w)))
      (push (vt-ref w i j) row))
    (nreverse row)))

;;; ============================================================
;;; 2. 测试框架
;;; ============================================================

(defparameter *test-pass* 0)
(defparameter *test-fail* 0)
(defmacro deftest (name &body body)
  `(defun ,name ()
     (handler-case
         (progn ,@body
                (incf *test-pass*)
                (format t "~&[PASS] ~a~%" ',name))
       (error (e)
         (incf *test-fail*)
         (format t "~&[FAIL] ~a: ~a~%" ',name e)))))

;;; ============================================================
;;; 3. 已修复 bug 的回归测试
;;; ============================================================

;;; ---- Bug 1: flops-estimate conv2d 偏置统计逻辑反 ----
(deftest test-flops-conv2d-bias
  (flet ((check (use-bias expected)
           (let* ((c (make-conv2d 8 '(3 3) :in-channels 3 :use-bias use-bias))
                  (x (vt-random-normal (list 1 3 8 8))))
             (forward c x)
             (multiple-value-bind (macs params)
                 (flops-estimate c (list 1 3 8 8))
               (declare (ignore macs))
               (assert (= params expected) ()
                       "use-bias=~a: 期望 ~a, 实际 ~a" use-bias expected params)))))
    (check t   224)
    (check nil 216)))

;;; ---- Bug 2: stop-gradient-node 前向忽略输入 ----
(deftest test-stop-gradient-forward
  (let* ((a (vt-const (list 2 3) 1.0d0))
         (b (vt-const (list 2 3) 9.0d0))
         (node (make-stop-gradient)))
    (assert (vt-equal-p (forward node b) b) () "forward 应返回传入 input")
    (assert (vt-equal-p (forward node a) a) () "forward 应返回传入 input")
    ;; 修复后：backward 返回与 grad 同形状的全零张量（不是 NIL），
    ;; 这样 sequential 才能把零梯度继续传给上游层，而不是崩溃。
    (let ((g (backward node (vt-ones (list 2 3)))))
      (assert g () "backward 应返回非 NIL（零张量）")
      (assert (equal (vt-shape g) '(2 3)) ()
              "backward 应返回与 grad 同形状的零张量")
      (assert (all-zeros-p (vt-to-list g)) ()
              "backward 应返回全零张量"))))

;;; ---- Bug 3: sequential + stop-gradient 的 nil 梯度 ----
(deftest test-sequential-stop-gradient
  (let* ((d1 (make-dense 4 :in-dim 3 :activation :none))
         (sg (make-stop-gradient))
         (d2 (make-dense 2 :activation :none))
         (m (make-sequential)))
    (seq-add! m d1)
    (seq-add! m sg)
    (seq-add! m d2)
    (let ((out (forward m (vt-random-normal (list 1 3)))))
      (backward m (vt-ones (vt-shape out)))
      (let ((dw1 (dense-dw d1)))
        (assert dw1 () "d1 的 dw 应为非 NIL")
        (assert (equal (vt-shape dw1) '(3 4)) () "d1 dw 形状应为 (3 4)")
        (assert (all-zeros-p (vt-to-list dw1)) ()
                "stop-gradient 之前应得到零梯度")))))

;;; ---- Bug 4: batch-norm 在 batch=1 时错误更新 running-var ----
(deftest test-bn-batch1-running-var
  (let* ((nf 3)
         (bn (make-batch-norm nf :momentum 0.1d0)))
    ;; 先跑一次 batch=2 触发初始化
    (with-training t (forward bn (vt-random-normal (list 2 nf))))
    (let ((rv0 (vt-to-list (bn-running-var bn))))
      ;; 连跑 5 次 batch=1
      (dotimes (i 5)
        (with-training t (forward bn (vt-random-normal (list 1 nf)))))
      (let ((rv5 (vt-to-list (bn-running-var bn))))
        (assert (equal rv0 rv5) ()
                "batch=1 时 running-var 不应改变：~a -> ~a" rv0 rv5)))))

;;; ---- Bug 5: flops-estimate LSTM/GRU MACs 漏乘 T ----
(deftest test-flops-lstm-gru-t
  (let* ((is 3) (hs 4) (seq 5))
    (let ((lstm (make-lstm is hs)))
      (multiple-value-bind (macs params) (flops-estimate lstm (list 2 seq is))
        (declare (ignore params))
        (assert (= macs (* seq 4 hs (+ is hs))) ()
                "LSTM MACs 应计入 T")))
    (let ((gru (make-gru is hs)))
      (multiple-value-bind (macs params) (flops-estimate gru (list 2 seq is))
        (declare (ignore params))
        (assert (= macs (* seq 3 hs (+ is hs))) ()
                "GRU MACs 应计入 T")))))

;;; ---- Bug 6: LSTM h-0 / c-0 梯度被丢弃 ----
(deftest test-lstm-h0-c0-grad
  (let* ((batch 2) (seq 3) (is 4) (hs 5)
         (h0 (vt-random-normal (list batch hs)))
         (c0 (vt-random-normal (list batch hs)))
         (lstm (make-lstm is hs :h-0 h0 :c-0 c0))
         (x (vt-random-normal (list batch seq is))))
    (forward lstm x)
    (multiple-value-bind (g dh0 dc0)
        (backward lstm (vt-ones (list batch seq hs)))
      (assert g () "grad-input 应为非 NIL")
      (assert dh0 () "dh-0 应为非 NIL")
      (assert dc0 () "dc-0 应为非 NIL")
      (assert (equal (vt-shape dh0) (list batch hs)) ())
      (assert (equal (vt-shape dc0) (list batch hs)) ()))
    ;; 未传 h-0 / c-0 时应返回 NIL
    (let ((lstm2 (make-lstm is hs)))
      (forward lstm2 x)
      (multiple-value-bind (g dh0 dc0)
          (backward lstm2 (vt-ones (list batch seq hs)))
        (declare (ignore g))
        (assert (null dh0) () "未传 h-0 时 dh-0 应为 NIL")
        (assert (null dc0) () "未传 c-0 时 dc-0 应为 NIL")))))

;;; ---- Bug 7: flops-estimate MHA 参数量漏算偏置 ----
(deftest test-flops-mha-bias
  (let* ((d 8) (nh 2) (input-shape (list 2 5 d)))
    (flet ((report (use-bias expected)
             (let ((m (make-multi-head-attention d nh :use-bias use-bias)))
               (forward m (list (vt-random-normal input-shape)
                                (vt-random-normal input-shape)
                                (vt-random-normal input-shape)))
               (multiple-value-bind (macs params) (flops-estimate m input-shape)
                 (declare (ignore macs))
                 (assert (= params expected) ()
                         "use-bias=~a: 期望 ~a, 实际 ~a"
                         use-bias expected params)))))
      (report t   (+ (* 4 d d) (* 4 d)))
      (report nil (* 4 d d)))))

;;; ---- Bug 8: KL divergence 梯度被 clip 吞掉 ----
(deftest test-kl-gradient
  (let* ((shape '(3 4))
         (loss (make-kl-divergence-loss :reduction :mean))
         (predicted (vt-random-normal shape))
         (target (vt-softmax (vt-random-normal shape))))
    ;; 数值梯度 (采样 4 个位置)
    (let* ((flat-p (t-flat predicted))
           (eps 1.0d-6)
           (analytic (t-flat (compute-loss-gradient loss predicted target)))
           (max-err 0.0d0))
      (loop for i in '(0 3 7 11)
            for num = (flet ((eval-at (delta)
                               (let* ((p (copy-list flat-p))
                                      (_ (setf (nth i p) (+ (nth i flat-p) delta)))
                                      (z (vt-reshape (vt-from-sequence p) shape)))
                                 (declare (ignore _))
                                 (vt-item (compute-loss loss z target)))))
                        (/ (- (eval-at eps) (eval-at (- eps))) (* 2.0d0 eps)))
            do (setf max-err (max max-err
                                  (/ (abs (- num (nth i analytic)))
                                     (max 1.0d0 (abs num) (abs (nth i analytic)))))))
      (assert (< max-err 1.0d-4) ()
              "KL 解析梯度与数值梯度不一致，最大相对误差 ~a" max-err))))

;;; ============================================================
;;; 4. 已确认无 bug 的行为回归测试
;;; ============================================================

;;; ---- RNN 时间步切片返回 2D（撤回旧结论） ----
(deftest test-rnn-slice-shape
  (let* ((x (vt-random-normal (list 2 3 4)))
         (slice (vt-slice x (list :all) (list 0) (list :all))))
    (assert (equal (vt-shape slice) '(2 4)) ()
            "vt-slice 单元素索引应压缩维度，实际 ~a" (vt-shape slice)))
  ;; LSTM / GRU / rnn-sequence forward 应能跑通
  (let ((lstm (make-lstm 4 5))
        (gru (make-gru 4 5))
        (rnn (make-rnn-sequence 4 5)))
    (forward lstm (vt-random-normal (list 2 3 4)))
    (forward gru  (vt-random-normal (list 2 3 4)))
    (forward rnn  (vt-random-normal (list 2 3 4)))))

;;; ---- 序列化往返 ----
(deftest test-serialization-roundtrip
  (let* ((m (make-sequential))
         (d (make-dense 3 :in-dim 2 :activation :none))
         (x (vt-random-normal (list 1 2)))
         (path "/tmp/clnn-test-serialization.lisp"))
    (forward d x)
    (seq-add! m d)
    (save-model m path)
    (let* ((m2 (load-model path))
           (d2 (first (seq-layers m2))))
      (assert (vt-equal-p (dense-weights d) (dense-weights d2)) ()
              "权重往返应一致")
      (assert (vt-equal-p (dense-bias d) (dense-bias d2)) ()
              "偏置往返应一致"))))

;;; ---- AMSGrad vt-map 多张量 ----
(deftest test-amsgrad
  (let ((a (vt-from-sequence '(1.0d0 5.0d0 3.0d0)))
        (b (vt-from-sequence '(4.0d0 2.0d0 6.0d0))))
    (assert (equal (vt-to-list (vt-map #'max a b)) '(4.0d0 5.0d0 6.0d0)) ()
            "vt-map 应支持多张量"))
  ;; AMSGrad 实跑
  (let* ((d (make-dense 3 :in-dim 2 :activation :none))
         (opt (make-adam :lr 0.1d0 :amsgrad t))
         (x (vt-random-normal (list 4 2)))
         (y (vt-random-normal (list 4 3))))
    (forward d x)
    (dotimes (i 3)
      (zero-grad! d)
      (let* ((out (forward d x))
             (g (vt-scale (vt-- out y) 0.5d0)))
        (backward d g))
      (optimizer-step opt (params d) (grads d)))))

;;; ---- transformer-block set-training! 传播 ----
(deftest test-tb-training-prop
  (let* ((d 8) (nh 2) (seq 4) (batch 2)
         (tb (make-transformer-block d nh :ffn-dim 16 :dropout-rate 0.5d0))
         (x (vt-random-normal (list batch seq d))))
    (set-model-training! tb t)
    (let ((o1 (forward tb x))
          (o2 (forward tb x)))
      (assert (not (vt-equal-p o1 o2)) () "训练模式下两次输出应不同"))
    (set-model-training! tb nil)
    (dolist (sub (list (tb-mha tb) (tb-ffn1 tb) (tb-ffn2 tb)
                       (tb-ln1 tb) (tb-ln2 tb) (tb-drop1 tb) (tb-drop2 tb)))
      (assert (null (training-p sub)) () "set-model-training! nil 后子层应为推理模式"))
    (let ((o1 (forward tb x))
          (o2 (forward tb x)))
      (assert (vt-equal-p o1 o2) () "推理模式下两次输出应相同"))))

;;; ---- im2col / col2im 伴随性 ----
(deftest test-im2col-col2im
  (flet ((check-adjoint (kh kw sh sw ph pw H W)
           (let* ((x (vt-random-normal (list 1 2 H W)))
                  (col-shape (vt-shape (im2col x kh kw sh sw ph pw)))
                  (col (vt-random-normal col-shape))
                  (lhs (t-inner (im2col x kh kw sh sw ph pw) col))
                  (rhs (t-inner x (col2im col kh kw sh sw ph pw (vt-shape x))))
                  (rel (/ (abs (- lhs rhs)) (max 1.0d0 (abs lhs) (abs rhs)))))
             (assert (< rel 1.0d-10) ()
                     "伴随性不成立 (~a): rel=~a" (list kh kw sh sw ph pw) rel))))
    (check-adjoint 1 1 1 1 0 0 4 4)
    (check-adjoint 3 3 1 1 0 0 5 5)
    (check-adjoint 3 3 2 2 0 0 6 6)
    (check-adjoint 3 3 1 1 1 1 5 5)
    (check-adjoint 5 5 3 3 2 2 8 8)))

;;; ---- BN ND 路径与 2D 路径一致 ----
(deftest test-bn-nd-vs-2d
  (let* ((n 2) (c 3) (h 4) (w 5)
         (bn-nd (make-batch-norm c :momentum 0.1d0))
         (bn-2d (make-batch-norm c :momentum 0.1d0))
         (x4d (vt-random-normal (list n c h w)))
         (x2d (vt-reshape (vt-transpose x4d '(0 2 3 1))
                          (list (* n h w) c))))
    (let ((y-nd (with-training t (forward bn-nd x4d)))
          (y-2d (with-training t (forward bn-2d x2d))))
      (declare (ignore y-nd y-2d))
      (assert (approx-eq (vt-item (vt-sum (bn-running-mean bn-nd)))
                         (vt-item (vt-sum (bn-running-mean bn-2d))))
              () "running-mean 应一致")
      (assert (approx-eq (vt-item (vt-sum (bn-running-var bn-nd)))
                         (vt-item (vt-sum (bn-running-var bn-2d))))
              () "running-var 应一致"))))

;;; ---- copy-network 深拷贝独立性 ----
(deftest test-copy-network-deep
  (let* ((d (make-dense 4 :in-dim 3 :activation :none))
         (x (vt-random-normal (list 1 3))))
    (forward d x)
    (let* ((d2 (copy-network d))
           (w (dense-weights d))
           (w2 (dense-weights d2)))
      (setf (vt-ref w 0 0) 12345.0d0)
      (assert (/= (vt-ref w2 0 0) 12345.0d0) ()
              "深拷贝后副本不应受原模型修改影响"))))

;;; ---- params / grads 契约 ----
(deftest test-params-grads-consistency
  (flet ((check (label layer)
           (let ((p (params layer)) (g (grads layer)))
             (assert (= (length p) (length g)) ()
                     "~a: params 长度 ~a != grads 长度 ~a"
                     label (length p) (length g))
             (assert (equal (mapcar #'second p) (mapcar #'car g)) ()
                     "~a: 名称顺序不一致" label))))
    (let ((d (make-dense 4 :in-dim 3 :activation :none)))
      (forward d (vt-random-normal (list 1 3)))
      (check "Dense" d))
    (let ((c (make-conv2d 8 '(3 3) :in-channels 3)))
      (forward c (vt-random-normal (list 1 3 8 8)))
      (check "Conv2d" c))
    (let ((bn (make-batch-norm 3)))
      (forward bn (vt-random-normal (list 2 3)))
      (check "BatchNorm" bn))
    (let ((ln (make-layer-norm '(4))))
      (forward ln (vt-random-normal (list 2 4)))
      (check "LayerNorm" ln))
    (let ((e (make-embedding 10 4)))
      (forward e (vt-from-sequence '(1 2 3) :dtype :int64))
      (check "Embedding" e))
    (let ((l (make-lstm 3 5)))
      (forward l (vt-random-normal (list 2 4 3)))
      (check "LSTM" l))
    (let ((g (make-gru 3 5)))
      (forward g (vt-random-normal (list 2 4 3)))
      (check "GRU" g))
    (let ((m (make-multi-head-attention 8 2)))
      (forward m (list (vt-random-normal (list 2 4 8))
                       (vt-random-normal (list 2 4 8))
                       (vt-random-normal (list 2 4 8))))
      (check "MHA" m))))

;;; ---- SDPA 反向数值验证 ----
(deftest test-sdpa-grad
  (let* ((batch 2) (seq 3) (dk 4) (dv 4)
         (q (vt-random-normal (list batch seq dk)))
         (k (vt-random-normal (list batch seq dk)))
         (v (vt-random-normal (list batch seq dv)))
         (grad-out (vt-random-normal (list batch seq dv))))
    (multiple-value-bind (out attn dropped mask)
        (sdpa-forward q k v nil 0.0d0 nil)
      (declare (ignore out mask))
      (multiple-value-bind (dq dk dv)
          (sdpa-backward grad-out q k v attn dropped nil)
        (flet ((check (which target analytic idxs)
                 (let* ((shape (vt-shape target))
                        (f (t-flat target))
                        (eps 1.0d-6))
                   (dolist (i idxs)
                     (flet ((eval-at (delta)
                              (let* ((g (copy-list f))
                                     (_ (setf (nth i g) (+ (nth i f) delta)))
                                     (tensor (vt-reshape (vt-from-sequence g) shape)))
                                (declare (ignore _))
                                (multiple-value-bind (o a d m)
                                    (sdpa-forward
                                     (if (eq which :q) tensor q)
                                     (if (eq which :k) tensor k)
                                     (if (eq which :v) tensor v)
                                     nil 0.0d0 nil)
                                  (declare (ignore a d m))
                                  (t-inner o grad-out)))))
                       (let* ((num (/ (- (eval-at eps) (eval-at (- eps))) (* 2.0d0 eps)))
                              (ana (nth i analytic))
                              (rel (/ (abs (- num ana))
                                      (max 1.0d0 (abs num) (abs ana)))))
                         (assert (< rel 1.0d-4) ()
                                 "SDPA ~a 梯度误差偏大: rel=~a" which rel)))))))
          (check :q q (t-flat dq) '(0 5 10))
          (check :k k (t-flat dk) '(1 6 11))
          (check :v v (t-flat dv) '(2 7 12)))))))

;;; ---- MHA 端到端反向 ----
(deftest test-mha-grad
  (let* ((batch 2) (seq 3) (d 8) (nh 2)
         (mha (make-multi-head-attention d nh :dropout-rate 0.0d0))
         (q (vt-random-normal (list batch seq d)))
         (k (vt-random-normal (list batch seq d)))
         (v (vt-random-normal (list batch seq d)))
         (grad-out (vt-random-normal (list batch seq d))))
    (forward mha (list q k v))
    (multiple-value-bind (dq dk dv)
        (backward mha grad-out)
      (declare (ignore dk dv))
      ;; 数值微分核对 dq 的若干位置
      (let* ((shape (vt-shape q))
             (f (t-flat q))
             (eps 1.0d-6)
             (dq-flat (t-flat dq)))
        (dolist (i '(0 7 15))
          (flet ((eval-at (delta)
                   (let* ((g (copy-list f))
                          (_ (setf (nth i g) (+ (nth i f) delta)))
                          (qq (vt-reshape (vt-from-sequence g) shape)))
                     (declare (ignore _))
                     (t-inner (forward mha (list qq k v)) grad-out))))
            (let* ((num (/ (- (eval-at eps) (eval-at (- eps))) (* 2.0d0 eps)))
                   (ana (nth i dq-flat))
                   (rel (/ (abs (- num ana)) (max 1.0d0 (abs num) (abs ana)))))
              (assert (< rel 1.0d-4) ()
                      "MHA dq[~a] 误差偏大: rel=~a" i rel))))))))

;;; ---- Embedding 反向 ----
(deftest test-embedding-grad
  (let* ((ne 5) (ed 3)
         (emb (make-embedding ne ed))
         (idx (vt-from-sequence '(1 2 1 3) :dtype :int64)))
    (forward emb idx)
    (backward emb (vt-ones (list 4 ed)))
    (let ((dw (emb-dw emb)))
      (assert (every (lambda (x) (approx-eq x 0.0d0))
                     (t-get-row dw 0)) () "未出现的词梯度应为 0")
      (assert (every (lambda (x) (approx-eq x 2.0d0))
                     (t-get-row dw 1)) () "出现 2 次的词梯度应为 2")
      (assert (every (lambda (x) (approx-eq x 1.0d0))
                     (t-get-row dw 2)) () "出现 1 次的词梯度应为 1"))))

;;; ---- Conv2d 反向数值验证 ----
(deftest test-conv2d-grad
  (let* ((conv (make-conv2d 3 '(3 3) :in-channels 2
                            :stride '(1 1) :padding '(1 1)))
         (x (vt-random-normal (list 1 2 5 5))))
    (let* ((y (forward conv x))
           (grad-out (vt-random-normal (vt-shape y)))
           (d-input (backward conv grad-out))
           (dw (conv-dw conv)))
      (flet ((check (target analytic idxs which)
               (let* ((shape (vt-shape target))
                      (f (t-flat target))
                      (eps 1.0d-5))
                 (dolist (i idxs)
                   (flet ((eval-at (delta)
                            (let* ((g (copy-list f))
                                   (_ (setf (nth i g) (+ (nth i f) delta)))
                                   (tensor (vt-reshape (vt-from-sequence g) shape)))
                              (declare (ignore _))
                              (if (eq which :x)
                                  (t-inner (forward conv tensor) grad-out)
                                  (progn
                                    (setf (conv-weights conv) tensor)
                                    (t-inner (forward conv x) grad-out))))))
                     (let* ((num (/ (- (eval-at eps) (eval-at (- eps))) (* 2.0d0 eps)))
                            (ana (nth i analytic))
                            (rel (/ (abs (- num ana)) (max 1.0d0 (abs num) (abs ana)))))
                       (assert (< rel 1.0d-4) ()
                               "Conv2d ~a[~a] 误差偏大: rel=~a" which i rel)))))))
        (check x (t-flat d-input) '(0 10 20) :x)
        (check (conv-weights conv) (t-flat dw) '(0 5 10) :w)))))

;;; ---- 池化层反向 ----
(deftest test-pool-grad
  (flet ((check (label pool x)
           (let* ((y (forward pool x))
                  (g (vt-random-normal (vt-shape y)))
                  (dx (backward pool g))
                  (shape (vt-shape x))
                  (f (t-flat x))
                  (eps 1.0d-6)
                  (ana (t-flat dx)))
             (dolist (i '(0 3 7))
               (flet ((eval-at (delta)
                        (let* ((gg (copy-list f))
                               (_ (setf (nth i gg) (+ (nth i f) delta)))
                               (tensor (vt-reshape (vt-from-sequence gg) shape)))
                          (declare (ignore _))
                          (t-inner (forward pool tensor) g))))
                 (let* ((num (/ (- (eval-at eps) (eval-at (- eps))) (* 2.0d0 eps)))
                        (a (nth i ana))
                        (rel (/ (abs (- num a)) (max 1.0d0 (abs num) (abs a)))))
                   (assert (< rel 1.0d-4) ()
                           "~a[~a] 误差偏大: rel=~a" label i rel)))))))
    (check "max-pool2d" (make-max-pool2d 2) (vt-random-normal (list 1 2 4 4)))
    (check "max-pool2d-pad" (make-max-pool2d 3 :stride 1 :padding 1)
           (vt-random-normal (list 1 2 5 5)))
    (check "avg-pool2d" (make-avg-pool2d 2) (vt-random-normal (list 1 2 4 4)))
    (check "avg-pool2d-pad" (make-avg-pool2d 3 :stride 1 :padding 1)
           (vt-random-normal (list 1 2 5 5)))
    (check "global-avg-pool2d" (make-global-avg-pool2d)
           (vt-random-normal (list 2 3 4 5)))))

;;; ---- LSTM 端到端反向 ----
(deftest test-lstm-grad
  (let* ((batch 2) (seq 3) (is 4) (hs 5)
         (lstm (make-lstm is hs))
         (x (vt-random-normal (list batch seq is))))
    (multiple-value-bind (out h c)
        (forward lstm x)
      (declare (ignore h c))
      (let* ((grad-out (vt-random-normal (vt-shape out)))
             (d-input (backward lstm grad-out))
             (dwih (lstm-dweight-ih lstm)))
        (flet ((check (target analytic idxs which)
                 (let* ((shape (vt-shape target))
                        (f (t-flat target))
                        (eps 1.0d-6))
                   (dolist (i idxs)
                     (flet ((eval-at (delta)
                              (let* ((g (copy-list f))
                                     (_ (setf (nth i g) (+ (nth i f) delta)))
                                     (tensor (vt-reshape (vt-from-sequence g) shape)))
                                (declare (ignore _))
                                (if (eq which :x)
                                    (t-inner (forward lstm tensor) grad-out)
                                    (progn
                                      (setf (lstm-weight-ih lstm) tensor)
                                      (multiple-value-bind (o hh cc)
                                          (forward lstm x)
                                        (declare (ignore hh cc))
                                        (t-inner o grad-out)))))))
                       (let* ((num (/ (- (eval-at eps) (eval-at (- eps))) (* 2.0d0 eps)))
                              (ana (nth i analytic))
                              (rel (/ (abs (- num ana)) (max 1.0d0 (abs num) (abs ana)))))
                         (assert (< rel 1.0d-4) ()
                                 "LSTM ~a[~a] 误差偏大: rel=~a" which i rel)))))))
          (check x (t-flat d-input) '(0 5 10) :x)
          (check (lstm-weight-ih lstm) (t-flat dwih) '(0 20 40) :w))))))

;;; ---- GRU 端到端反向 ----
(deftest test-gru-grad
  (let* ((batch 2) (seq 3) (is 4) (hs 5)
         (gru (make-gru is hs))
         (x (vt-random-normal (list batch seq is))))
    (multiple-value-bind (out h)
        (forward gru x)
      (declare (ignore h))
      (let* ((grad-out (vt-random-normal (vt-shape out)))
             (d-input (backward gru grad-out)))
        (let* ((shape (vt-shape x))
               (f (t-flat x))
               (eps 1.0d-6)
               (ana (t-flat d-input)))
          (dolist (i '(0 5 10))
            (flet ((eval-at (delta)
                     (let* ((g (copy-list f))
                            (_ (setf (nth i g) (+ (nth i f) delta)))
                            (tensor (vt-reshape (vt-from-sequence g) shape)))
                       (declare (ignore _))
                       (t-inner (forward gru tensor) grad-out))))
              (let* ((num (/ (- (eval-at eps) (eval-at (- eps))) (* 2.0d0 eps)))
                     (a (nth i ana))
                     (rel (/ (abs (- num a)) (max 1.0d0 (abs num) (abs a)))))
                (assert (< rel 1.0d-4) ()
                        "GRU d-input[~a] 误差偏大: rel=~a" i rel)))))))))

;;; ---- 优化器数值 ----
(deftest test-optimizers
  (flet ((run-opt (opt p0 g n)
           (let* ((owner (make-instance 'layer :name "opt" :trainable t))
                  (cell (list (vt-from-sequence p0 :dtype :float64)))
                  (history '()))
             (dotimes (i n)
               (optimizer-step opt
                               (list (list owner "w" (car cell)
                                           #'(lambda (v) (setf (car cell) v))))
                               (list (cons "w" (vt-from-sequence g :dtype :float64))))
               (push (t-flat (car cell)) history))
             (first (last (nreverse history))))))
    ;; SGD 无动量
    (let ((last (run-opt (make-sgd :lr 0.1d0) '(1.0d0) '(0.1d0) 3)))
      (assert (approx-eq (first last) 0.97d0) () "SGD 末步应为 0.97"))
    ;; SGD + momentum
    (let ((last (run-opt (make-sgd :lr 0.1d0 :momentum 0.9d0)
                         '(1.0d0) '(0.1d0) 3)))
      (assert (approx-eq (first last) 0.9439d0 :tol 1.0d-6) ()
              "SGD momentum 末步应为 0.9439"))
    ;; Nesterov
    (let ((last (run-opt (make-sgd :lr 0.1d0 :momentum 0.9d0 :nesterov t)
                         '(1.0d0) '(0.1d0) 3)))
      (assert (approx-eq (first last) 0.91951d0 :tol 1.0d-6) ()
              "SGD nesterov 末步应为 0.91951"))
    ;; Adagrad
    (let ((last (run-opt (make-adagrad :lr 0.1d0 :eps 1.0d-8)
                         '(1.0d0) '(0.1d0) 3)))
      (assert (approx-eq (first last) 0.7715543d0 :tol 1.0d-5) ()
              "Adagrad 末步应为 ~0.77155"))
    ;; RMSprop
    (let ((last (run-opt (make-rmsprop :lr 0.1d0 :alpha 0.99d0 :eps 1.0d-8)
                         '(1.0d0) '(0.1d0) 2)))
      (assert (approx-eq (first last) -0.70888d0 :tol 1.0d-4) ()
              "RMSprop 末步应为 ~-0.70888"))
    ;; 零梯度: AdamW 应收缩
    (let ((last (run-opt (make-adamw :lr 0.1d0) '(1.0d0) '(0.0d0) 5)))
      (assert (approx-eq (first last) (expt 0.999d0 5) :tol 1.0d-9) ()
              "AdamW 零梯度应乘 (1 - lr*wd)"))))

;;; ---- 学习率调度器 ----
(deftest test-schedulers
  (flet ((sched-seq (s n)
           (let ((lrs '()))
             (dotimes (i n)
               (scheduler-step! s)
               (push (optimizer-lr (scheduler-optimizer s)) lrs))
             (nreverse lrs)))
         (check-seq (got expected &key (tol 1.0d-9))
           (assert (and (= (length got) (length expected))
                        (every (lambda (a b) (approx-eq a b :tol tol))
                               got expected))
                   () "序列不一致: 期望 ~a, 实际 ~a" expected got)))
    (check-seq (sched-seq (make-step-lr (make-sgd :lr 1.0d0) 3 :gamma 0.5d0) 6)
               '(1.0d0 1.0d0 0.5d0 0.5d0 0.5d0 0.25d0))
    (check-seq (sched-seq (make-exponential-lr (make-sgd :lr 1.0d0) :gamma 0.5d0) 4)
               '(0.5d0 0.25d0 0.125d0 0.0625d0))
    (check-seq (sched-seq (make-cosine-annealing-lr (make-sgd :lr 1.0d0) 4
                                                    :eta-min 0.0d0) 4)
               '(0.8535533905932737d0 0.5d0 0.14644660940672627d0 0.0d0))
    (check-seq (sched-seq (make-warmup-cosine-lr (make-sgd :lr 1.0d0) 2 6
                                                 :min-lr 0.0d0) 6)
               '(0.5d0 1.0d0 0.8535533905932737d0 0.5d0
                 0.14644660940672627d0 0.0d0))))

;;; ============================================================
;;; 5. 集成测试
;;; ============================================================

(defun make-cluster-data (n dim)
  (let* ((x0 (vt-random-normal (list n dim)))
         (x1 (vt-+ (vt-random-normal (list n dim))
                   (vt-const (list dim) 2.0d0)))
         (x (vt-concatenate 0 x0 x1))
         (y (vt-from-sequence (append (make-list n :initial-element 0)
                                      (make-list n :initial-element 1))
                              :dtype :int64)))
    (values x y)))

(defun compute-accuracy (model x y n-samples n-classes)
  (let ((pred (forward model x))
        (correct 0))
    (dotimes (i n-samples)
      (let ((best-idx 0) (best-val most-negative-double-float))
        (dotimes (c n-classes)
          (let ((v (coerce (vt-ref pred i c) 'double-float)))
            (when (> v best-val)
              (setf best-val v best-idx c))))
        (when (= best-idx (coerce (vt-ref y i) 'fixnum))
          (incf correct))))
    (values correct n-samples)))

(deftest test-integration
  (let* ((n 100) (dim 8) (n-classes 2))
    (multiple-value-bind (x y) (make-cluster-data n dim)
      (let* ((model (make-sequential))
             (loss-fn (make-ce-loss :reduction :mean))
             (opt (make-adam :lr 0.01d0)))
        (seq-add! model (make-dense 16 :in-dim dim :activation :relu))
        (seq-add! model (make-dense n-classes :activation :none))
        ;; 训练 100 步
        (let ((losses '()))
          (dotimes (i 100)
            (zero-grad! model)
            (let* ((pred (forward model x))
                   (lv (vt-item (compute-loss loss-fn pred y)))
                   (g (compute-loss-gradient loss-fn pred y)))
              (backward model g)
              (optimizer-step opt (params model) (grads model))
              (push lv losses)))
          (let* ((first-10 (subseq losses 90 100))
                 (last-10 (subseq losses 0 10))
                 (avg-first (/ (reduce #'+ first-10) 10.0d0))
                 (avg-last (/ (reduce #'+ last-10) 10.0d0)))
            (assert (< avg-last (* 0.5d0 avg-first)) ()
                    "loss 应显著下降: 前 ~a, 后 ~a" avg-first avg-last)))
        ;; 训练后准确率
        (multiple-value-bind (c tt) (compute-accuracy model x y (* 2 n) n-classes)
          (assert (> (/ c tt) 0.9d0) ()
                  "训练后准确率应 > 90%, 实际 ~a/~a" c tt))
        ;; 保存/加载
        (let ((path "/tmp/clnn-integration.lisp"))
          (save-model model path)
          (let* ((model2 (load-model path))
                 (p1 (forward model x))
                 (p2 (forward model2 x)))
            (assert (vt-equal-p p1 p2) () "保存/加载后预测应一致")))
        ;; 推理模式确定性
        (set-model-training! model nil)
        (let ((p1 (forward model x))
              (p2 (forward model x)))
          (assert (vt-equal-p p1 p2) () "推理模式两次预测应一致"))))))

;;; ============================================================
;;; 6. 入口
;;; ============================================================

(defun run-all-tests ()
  (setf *test-pass* 0)
  (setf *test-fail* 0)
  (format t "~%========================================~%")
  (format t "  clnn 回归测试套件~%")
  (format t "========================================~%")

  (format t "~%--- 已修复 bug 的回归 ---~%")
  (test-flops-conv2d-bias)
  (test-stop-gradient-forward)
  (test-sequential-stop-gradient)
  (test-bn-batch1-running-var)
  (test-flops-lstm-gru-t)
  (test-lstm-h0-c0-grad)
  (test-flops-mha-bias)
  (test-kl-gradient)

  (format t "~%--- 已确认无 bug 的行为回归 ---~%")
  (test-rnn-slice-shape)
  (test-serialization-roundtrip)
  (test-amsgrad)
  (test-tb-training-prop)
  (test-im2col-col2im)
  (test-bn-nd-vs-2d)
  (test-copy-network-deep)
  (test-params-grads-consistency)
  (test-sdpa-grad)
  (test-mha-grad)
  (test-embedding-grad)
  (test-conv2d-grad)
  (test-pool-grad)
  (test-lstm-grad)
  (test-gru-grad)
  (test-optimizers)
  (test-schedulers)

  (format t "~%--- 集成测试 ---~%")
  (test-integration)

  (format t "~%========================================~%")
  (format t "  结果: ~a PASS / ~a FAIL~%" *test-pass* *test-fail*)
  (format t "========================================~%")
  (values *test-pass* *test-fail*))


(run-all-tests)
