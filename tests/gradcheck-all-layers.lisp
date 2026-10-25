;;;; gradcheck-all-layers.lisp — 全层有限差分梯度校验
;;;;
;;;; 对每一层、每一类损失做「解析梯度 vs 中心差分数值梯度」对拍。
;;;; 覆盖：dense 的 11 种激活、conv2d 的 5 种 stride/padding 组合、
;;;;       max/avg/global-avg 池化、batch-norm 的 2D/ND/无仿射、
;;;;       layer-norm、SDPA/MHA、transformer-block、残差、
;;;;       LSTM/GRU/rnn-sequence、embedding、7 类损失的三种归约。
;;;;
;;;; 用法:
;;;;   sbcl --dynamic-space-size 4096 --non-interactive \
;;;;        --load tests/gradcheck-all-layers.lisp
;;;;
;;;; 任何一项失配都会打印 [GRAD-FAIL] 行并计入末尾汇总。
;;;; 注意：本脚本会做大量前向/反向，执行时间约数分钟。
(require :asdf)
(asdf:load-system :clnn)
(in-package :nn)

(defparameter *fails* '())

(defun note-fail (fmt &rest args)
  (let ((s (apply #'format nil fmt args)))
    (push s *fails*)
    (format t "~&[GRAD-FAIL] ~a~%" s)))

(defun rel-err (a b)
  (/ (abs (- a b)) (max 1.0d-8 (abs a) (abs b))))

(defun flat (x) (vt-to-list (vt-flatten x)))

(defun sum-prod (a b)
  (let ((fa (flat a)) (fb (flat b)) (s 0.0d0))
    (mapc (lambda (x y) (incf s (* x y))) fa fb)
    s))

(defun vt-random-like (x) (vt-random (vt-shape x)))

(defun shape> (n &optional (scale 1.0d0))
  (vt-scale (vt-random n) scale))

(defun get-flat (tensor i)
  (aref (vt-data tensor) (+ (vt-offset tensor) i)))

(defun set-flat! (tensor i value)
  (setf (aref (vt-data tensor) (+ (vt-offset tensor) i))
        (coerce value 'double-float)))

(defun numeric-d (fn tensor i eps)
  "中心差分：对 TENSOR 第 I 个元素求 FN 的导数（FN 无参，读当前张量状态）。"
  (let* ((orig (get-flat tensor i))
         (h (* eps (max 1.0d0 (abs orig))))
         (lp (progn (set-flat! tensor i (+ orig h)) (funcall fn)))
         (lm (progn (set-flat! tensor i (- orig h)) (funcall fn))))
    (set-flat! tensor i orig)
    (/ (- lp lm) (* 2.0d0 h))))

(defun sample-indices (size samples)
  (if (<= size samples)
      (loop for i below size collect i)
      (loop for k below samples collect (floor (* k size) samples))))

(defun near-zero-p (v) (< (abs v) 1.0d-8))

(defun check-param-grads (label layer fwd bwd input eps tol samples &key loss-fn)
  "LOSS-FN 缺省为 sum(out * r)。比较每个参数张量的解析梯度与数值梯度。"
  (let* ((out (funcall fwd layer input))
         (r (vt-random (vt-shape out)))
         (target (or loss-fn (lambda (o) (sum-prod o r)))))
    (funcall bwd layer r)
    (loop for p in (params layer)
          for g in (grads layer)
          for name = (second p)
          for pv = (third p)
          for gv = (cdr g)
          do (cond
               ((or (null pv) (null gv))
                (unless (and (null pv) (null gv))
                  (note-fail "~a/~a: 参数或梯度为 NIL (p有值=~a g有值=~a)"
                             label name (not (null pv)) (not (null gv)))))
               (t
                (let ((ana (flat gv))
                      (size (reduce #'* (vt-shape pv))))
                  (dolist (i (sample-indices size samples))
                    (let* ((num (numeric-d (lambda () (funcall target (funcall fwd layer input)))
                                           pv i eps))
                           (an (nth i ana))
                           (e (rel-err an num)))
                      ;; b_k 使 scores 整体平移，softmax 后不变 => 梯度恒为 0；
                      ;; 此时数值差分只剩浮点噪声，用绝对阈值判定。
                      (when (and (> e tol)
                                 (not (and (near-zero-p an) (near-zero-p num))))
                        (note-fail "~a/~a[~a]: 解析=~,6g 数值=~,6g 相对误差=~,2e"
                                   label name i an num e))))))))))

(defun param-grad-check (label build fwd bwd input &key (eps 1.0d-6)
                                                    (tol 1.0d-4) (samples 8))
  (handler-case
      (check-param-grads label (funcall build) fwd bwd input eps tol samples)
    (error (e) (note-fail "~a: 抛错 ~a" label e))))

(defun input-grad-check (label layer input &key (eps 1.0d-6) (tol 1.0d-4)
                                        (samples 10))
  (handler-case
      (let* ((out (forward layer input))
             (r (vt-random (vt-shape out)))
             (dx (backward layer r))
             (ana (flat dx)))
        (dolist (i (sample-indices (reduce #'* (vt-shape input)) samples))
          (let* ((num (numeric-d (lambda () (sum-prod (forward layer input) r))
                                 input i eps))
                 (e (rel-err (nth i ana) num)))
            (when (> e tol)
              (note-fail "~a/dx[~a]: 解析=~,6g 数值=~,6g 相对误差=~,2e"
                         label i (nth i ana) num e)))))
    (error (e) (note-fail "~a: 抛错 ~a" label e))))

(defun scalar-loss (loss pred target)
  "把任意 reduction 的损失压成标量（:none 时求和），用于数值差分。"
  (vt-item (vt-sum (compute-loss loss pred target))))

(defun loss-grad-check (label loss pred target &key (eps 1d-6) (tol 1d-4)
                                                  (samples 8))
  (handler-case
      (let* ((g (compute-loss-gradient loss pred target))
             (ana (flat g)))
        (dolist (i (sample-indices (reduce #'* (vt-shape pred)) samples))
          (let* ((num (numeric-d (lambda () (scalar-loss loss pred target))
                                 pred i eps))
                 (e (rel-err (nth i ana) num)))
            (when (> e tol)
              (note-fail "~a[~a]: 解析=~,6g 数值=~,6g 误差=~,2e"
                         label i (nth i ana) num e)))))
    (error (e) (note-fail "~a: 抛错 ~a" label e))))

;;; ============ 1. dense 各激活 ============
(format t "~%########## 1. dense 各激活函数梯度 ##########~%")
(dolist (act '(:none :relu :leaky-relu :sigmoid :tanh :gelu :swish :mish
               :softplus :hard-tanh :hard-sigmoid))
  (param-grad-check (format nil "dense/~a" act)
                    (lambda () (make-dense 5 :in-dim 6 :activation act))
                    (lambda (l x) (forward l x))
                    (lambda (l g) (backward l g))
                    (shape> (list 3 6))))

;;; ============ 2. conv2d ============
(format t "~%########## 2. conv2d 梯度 ##########~%")
(dolist (cfg '((3 1 0) (3 1 1) (3 2 1) (1 1 0) (2 2 0)))
  (destructuring-bind (k s p) cfg
    (param-grad-check (format nil "conv2d/k~a/s~a/p~a" k s p)
                      (lambda () (make-conv2d 3 k :in-channels 2
                                              :stride s :padding p))
                      (lambda (l x) (forward l x))
                      (lambda (l g) (backward l g))
                      (shape> (list 2 2 7 7)))))

;;; ============ 3. 池化 / flatten ============
(format t "~%########## 3. 池化/flatten 输入梯度 ##########~%")
(input-grad-check "max-pool-dx" (make-max-pool2d 2) (shape> (list 1 2 6 6)))
(input-grad-check "max-pool-dx-pad" (make-max-pool2d 3 :stride 2 :padding 1)
                  (shape> (list 1 2 7 7)))
(input-grad-check "avg-pool-dx" (make-avg-pool2d 2) (shape> (list 1 2 6 6)))
(input-grad-check "avg-pool-dx-pad" (make-avg-pool2d 3 :stride 2 :padding 1)
                  (shape> (list 1 2 7 7)))
(input-grad-check "gap-dx" (make-global-avg-pool2d) (shape> (list 2 3 4 4)))
(input-grad-check "flatten-dx" (make-flatten) (shape> (list 2 3 4 4)))

;;; ============ 4. 归一化 ============
(format t "~%########## 4. batch-norm / layer-norm 梯度 ##########~%")
(param-grad-check "bn-2d"
                  (lambda () (make-batch-norm 4))
                  (lambda (l x) (with-training t (forward l x)))
                  (lambda (l g) (backward l g))
                  (shape> (list 6 4)))
(param-grad-check "bn-nd"
                  (lambda () (make-batch-norm 3))
                  (lambda (l x) (with-training t (forward l x)))
                  (lambda (l g) (backward l g))
                  (shape> (list 2 3 4 4)))
(param-grad-check "bn-2d-noaffine"
                  (lambda () (make-batch-norm 4 :affine nil))
                  (lambda (l x) (with-training t (forward l x)))
                  (lambda (l g) (backward l g))
                  (shape> (list 6 4)))
(param-grad-check "ln"
                  (lambda () (make-layer-norm '(4)))
                  (lambda (l x) (forward l x))
                  (lambda (l g) (backward l g))
                  (shape> (list 5 4)))
(param-grad-check "ln-3d"
                  (lambda () (make-layer-norm '(4)))
                  (lambda (l x) (forward l x))
                  (lambda (l g) (backward l g))
                  (shape> (list 2 5 4)))
(input-grad-check "bn-dx" (make-batch-norm 4) (shape> (list 6 4)))

;;; ============ 5. 注意力 ============
(format t "~%########## 5. 注意力梯度 ##########~%")
(let ((x (shape> (list 2 4 8))))
  (param-grad-check "mha"
                    (lambda () (make-multi-head-attention 8 2))
                    (lambda (l q) (forward l (list q q q)))
                    (lambda (l g) (backward l g))
                    x)
  (param-grad-check "mha-nobias"
                    (lambda () (make-multi-head-attention 8 2 :use-bias nil))
                    (lambda (l q) (forward l (list q q q)))
                    (lambda (l g) (backward l g))
                    x))

(let* ((q (shape> (list 2 4 8)))
       (k (shape> (list 2 4 8)))
       (v (shape> (list 2 4 8)))
       (sdpa (make-scaled-dot-product-attention)))
  (labels ((chk (name tensor)
             (let* ((out (forward sdpa (list q k v)))
                    (r (vt-random (vt-shape out))))
               (forward sdpa (list q k v))
               (multiple-value-bind (dq dk dv) (backward sdpa r)
                 (let ((dt (ecase name (:q dq) (:k dk) (:v dv))))
                   (dolist (i (sample-indices (reduce #'* (vt-shape tensor)) 6))
                     (let* ((num (numeric-d
                                  (lambda () (sum-prod (forward sdpa (list q k v)) r))
                                  tensor i 1d-6))
                            (an (nth i (flat dt)))
                            (e (rel-err an num)))
                       (when (> e 1d-4)
                         (note-fail "sdpa/d~a[~a]: 解析=~,6g 数值=~,6g 误差=~,2e"
                                    name i an num e)))))))))
    (chk :q q) (chk :k k) (chk :v v)))

;;; ============ 6. 残差 / transformer-block ============
(format t "~%########## 6. 残差 / transformer-block 梯度 ##########~%")
(param-grad-check "residual"
                  (lambda () (make-residual (make-dense 6 :in-dim 6 :activation :tanh)))
                  (lambda (l x) (forward l x))
                  (lambda (l g) (backward l g))
                  (shape> (list 4 6)))

(let ((tb (make-transformer-block 8 2 :dropout-rate 0.0d0)))
  (set-model-training! tb t)
  (setf (dropout-p (tb-drop1 tb)) 0.0d0)
  (setf (dropout-p (tb-drop2 tb)) 0.0d0)
  (param-grad-check "transformer-block"
                    (lambda () (make-transformer-block 8 2 :dropout-rate 0.0d0))
                    (lambda (l x) (forward l x))
                    (lambda (l g) (backward l g))
                    (shape> (list 2 4 8)))
  (setf (dropout-p (tb-drop1 tb)) 0.0d0)
  (setf (dropout-p (tb-drop2 tb)) 0.0d0)
  (param-grad-check "transformer-block-tb"
                    (lambda () tb)
                    (lambda (l x) (forward l x))
                    (lambda (l g) (backward l g))
                    (shape> (list 2 4 8))))

;;; ============ 7. RNN ============
(format t "~%########## 7. RNN 系列梯度 ##########~%")
(param-grad-check "lstm"
                  (lambda () (make-lstm 5 4))
                  (lambda (l x) (forward l x))
                  (lambda (l g) (backward l g))
                  (shape> (list 2 3 5)))
(param-grad-check "gru"
                  (lambda () (make-gru 5 4))
                  (lambda (l x) (forward l x))
                  (lambda (l g) (backward l g))
                  (shape> (list 2 3 5)))
(param-grad-check "rnn-seq"
                  (lambda () (make-rnn-sequence 5 4))
                  (lambda (l x) (forward l x))
                  (lambda (l g) (backward l g))
                  (shape> (list 2 3 5)))
(input-grad-check "lstm-dx" (make-lstm 5 4) (shape> (list 2 3 5)))
(input-grad-check "gru-dx" (make-gru 5 4) (shape> (list 2 3 5)))
(input-grad-check "rnn-seq-dx" (make-rnn-sequence 5 4) (shape> (list 2 3 5)))

;;; ============ 8. embedding ============
(format t "~%########## 8. embedding 梯度 ##########~%")
(param-grad-check "embedding"
                  (lambda () (make-embedding 5 4))
                  (lambda (l i) (forward l i))
                  (lambda (l g) (backward l g))
                  (vt-from-sequence '(0 2 1) :dtype :int64)
                  :samples 6)
;; scale-grad-by-freq 的语义是「梯度乘以 1/频次」，因此
;; 数值梯度 × (1/频次) 才是期望值；这里用独立实现验证该语义。
(let* ((idx (vt-from-sequence '(0 0 1) :dtype :int64))
       (layer (make-embedding 5 4 :scale-grad-by-freq t))
       (out (forward layer idx))
       (r (vt-random (vt-shape out))))
  (backward layer r)
  (let ((ana (flat (cdr (first (grads layer)))))
        (freq (let ((h (make-hash-table)))
                (dotimes (i 3) (incf (gethash (vt-ref idx i) h 0)))
                h))
        (w (emb-weight layer)))
    (dolist (i (sample-indices (* 5 4) 6))
      (let* ((row (floor i 4))
             (num (numeric-d (lambda () (sum-prod (forward layer idx) r)) w i 1d-6))
             (expect (* num (/ 1.0d0 (coerce (gethash row freq 1) 'double-float))))
             (e (rel-err (nth i ana) expect)))
        (when (> e 1d-4)
          (note-fail "embedding/freq[~a]: 解析=~,6g 期望(数值/频次)=~,6g 误差=~,2e"
                     i (nth i ana) expect e))))))

;;; ============ 9. 损失函数 ============
(format t "~%########## 9. 损失函数梯度 ##########~%")
(let ((p (shape> (list 4 5))) (tgt (shape> (list 4 5))))
  (dolist (red '(:mean :sum :none))
    (loss-grad-check (format nil "mse/~a" red) (make-mse-loss :reduction red)
                     (vt-copy p) (vt-copy tgt))
    (loss-grad-check (format nil "huber/~a" red) (make-huber-loss :reduction red)
                     (vt-copy p) (vt-copy tgt))
    (loss-grad-check (format nil "smooth-l1/~a" red)
                     (make-smooth-l1-loss :reduction red)
                     (vt-copy p) (vt-copy tgt))
    (loss-grad-check (format nil "cosine/~a" red)
                     (make-cosine-similarity-loss :reduction red)
                     (vt-copy p) (vt-copy tgt))))

(let ((p (vt-scale (vt-+ (shape> (list 4 5)) (vt-const (list 4 5) 2.0d0)) 0.2d0))
      (tgt (vt-scale (vt-+ (shape> (list 4 5)) (vt-const (list 4 5) 2.0d0)) 0.4d0)))
  (dolist (red '(:mean :sum :none))
    (loss-grad-check (format nil "bce/~a" red) (make-bce-loss :reduction red)
                     (vt-copy p) (vt-copy tgt))))

(let ((logits (shape> (list 4 5)))
      (labels (vt-from-sequence '(0 3 1 4) :dtype :int64)))
  (dolist (red '(:mean :sum))
    (loss-grad-check (format nil "ce/~a" red) (make-ce-loss :reduction red)
                     logits labels))
  (loss-grad-check "ce/smooth" (make-ce-loss :label-smoothing 0.1d0)
                   logits labels))

(let ((logq (shape> (list 4 5)))
      (p (vt-scale (vt-abs (shape> (list 4 5))) 0.5d0)))
  (dolist (red '(:mean :sum :none))
    (loss-grad-check (format nil "kl/~a" red) (make-kl-divergence-loss :reduction red)
                     logq p)))

(format t "~%########## 完成: ~a 个失败 ##########~%" (length *fails*))
(dolist (f (reverse *fails*)) (format t " * ~a~%" f))
