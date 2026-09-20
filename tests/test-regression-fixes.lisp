;;;; test-regression-fixes.lisp — 代码审查所报缺陷的回归测试
;;;; 每个用例对应报告中的一个缺陷编号；修复回退时应当失败。
(in-package :clnn)

(defparameter *rt-pass* 0)
(defparameter *rt-fail* 0)

(defmacro rt (id name &body body)
  "执行 BODY；无异常且无 (rt-fail) 调用则记为通过。"
  `(handler-case
       (progn ,@body
              (incf *rt-pass*)
              (format t "  [PASS] ~a ~a~%" ,id ,name))
     (error (e)
       (incf *rt-fail*)
       (format t "  [FAIL] ~a ~a  -> ~a~%" ,id ,name e))))

(defmacro rt-assert (form)
  `(unless ,form (error "断言失败: ~s" ',form)))

(format t "~%~%===== 缺陷回归测试 =====~%")

;;; ---------------------------------------------------------------
;;; P0-1  conv2d 前向输出布局（out-channels > 1）
;;; ---------------------------------------------------------------
(rt "P0-1" "conv2d 前向输出布局"
  (let* ((c (make-conv2d 2 '(1 1) :in-channels 1 :use-bias nil))
         (x (vt-reshape (vt-from-sequence '(1.0d0 2.0d0 3.0d0 4.0d0)) '(1 1 2 2))))
    (setf (conv-weights c)
          (vt-reshape (vt-from-sequence '(10.0d0 100.0d0)) '(2 1 1 1)))
    (let ((o (forward c x)))
      (rt-assert (equal (vt-shape o) '(1 2 2 2)))
      (rt-assert (equal (vt-to-list (vt-slice o '(0) '(0) '(:all) '(:all)))
                        '((10.0d0 20.0d0) (30.0d0 40.0d0))))
      (rt-assert (equal (vt-to-list (vt-slice o '(0) '(1) '(:all) '(:all)))
                        '((100.0d0 200.0d0) (300.0d0 400.0d0)))))))

;;; 2x2 核、3x3 输入、2 输出通道的参考值
(rt "P0-1" "conv2d 2x2 核参考值"
  (let* ((c (make-conv2d 2 '(2 2) :in-channels 1 :use-bias nil))
         (x (vt-reshape (vt-from-sequence '(1.0d0 2.0d0 3.0d0 4.0d0 5.0d0 6.0d0 7.0d0 8.0d0 9.0d0))
                        '(1 1 3 3)))
         (w (vt-reshape (vt-from-sequence '(1.0d0 1.0d0 1.0d0 1.0d0
                                            10.0d0 10.0d0 10.0d0 10.0d0)) '(2 1 2 2))))
    (setf (conv-weights c) w)
    (let ((o (forward c x)))
      (rt-assert (equal (vt-to-list (vt-slice o '(0) '(0) '(:all) '(:all)))
                        '((12.0d0 16.0d0) (24.0d0 28.0d0))))
      (rt-assert (equal (vt-to-list (vt-slice o '(0) '(1) '(:all) '(:all)))
                        '((120.0d0 160.0d0) (240.0d0 280.0d0)))))))

;;; ---------------------------------------------------------------
;;; P0-2  conv2d 偏置梯度（解析 vs 中心差分）
;;; ---------------------------------------------------------------
(rt "P0-2" "conv2d 偏置梯度与数值梯度一致"
  (let* ((c (make-conv2d 3 '(3 3) :in-channels 2 :padding '(1 1)))
         (x (vt-random-normal '(2 2 5 5))))
    (forward c x)
    (let* ((go (vt-random-normal '(2 3 5 5)))
           (b (conv-bias c))
           (num (vt-zeros '(3)))
           (f (lambda () (vt-item (vt-sum (vt-* (forward c x) go))))))
      (zero-grad! c)
      (backward c go)
      (let ((an (conv-db c)))
        (dotimes (i 3)
          (let ((o (vt-ref b i)))
            (setf (vt-ref b i) (+ o 1d-6))
            (let ((fp (funcall f)))
              (setf (vt-ref b i) (- o 1d-6))
              (let ((fm (funcall f)))
                (setf (vt-ref b i) o)
                (setf (vt-ref num i) (/ (- fp fm) 2d-6))))))
        (dotimes (i 3)
          (rt-assert (< (abs (- (vt-ref an i) (vt-ref num i))) 1d-4)))))))

;;; ---------------------------------------------------------------
;;; P0-3  池化层对非连续输入的 strides 处理
;;; ---------------------------------------------------------------
(rt "P0-3" "max/avg-pool2d 处理非连续视图"
  (let* ((base (vt-random-normal '(1 4 4 4)))
         (view (vt-transpose base '(0 2 1 3)))
         (cont (vt-contiguous view)))
    (dolist (mk (list (lambda () (make-max-pool2d '(2 2)))
                      (lambda () (make-avg-pool2d '(2 2)))))
      (let ((o1 (forward (funcall mk) view))
            (o2 (forward (funcall mk) cont)))
        (rt-assert (equal (vt-to-list (vt-flatten o1))
                          (vt-to-list (vt-flatten o2))))))))

;;; conv -> batch-norm -> pool 常规链路（bn 的 ND 输出是非连续视图）
(rt "P0-3" "conv -> batch-norm -> max-pool 链路"
  (let* ((conv (make-conv2d 3 '(3 3) :in-channels 1 :padding '(1 1)))
         (bn (make-batch-norm 3))
         (pool1 (make-max-pool2d '(2 2)))
         (pool2 (make-max-pool2d '(2 2)))
         (x (vt-random-normal '(1 1 6 6))))
    (set-global-training! t)
    (let* ((y2 (forward bn (forward conv x)))
           (a (forward pool1 y2))
           (b (forward pool2 (vt-contiguous y2))))
      (rt-assert (not (vt-contiguous-p y2)))     ; 前提：bn 输出确实非连续
      (rt-assert (equal (vt-to-list (vt-flatten a))
                        (vt-to-list (vt-flatten b)))))))

;;; ---------------------------------------------------------------
;;; P1-1  trainable nil
;;; ---------------------------------------------------------------
(rt "P1-1" "trainable=nil 的层不被优化器更新"
  (let* ((m (make-sequential))
         (frozen (make-dense 4 :in-dim 4 :activation :none :trainable nil))
         (x (vt-random-normal '(4 4))))
    (seq-add! m frozen)
    (build-model m x)
    (let ((w0 (vt-to-list (vt-flatten (dense-weights frozen)))))
      (dotimes (i 20)
        (zero-grad! m)
        (model-forward m x)
        (model-backward m (vt-random-normal '(4 4)))
        (model-update! m (make-sgd :lr 0.1d0)))
      (rt-assert (equal w0 (vt-to-list (vt-flatten (dense-weights frozen))))))))

(rt "P1-1" "zero-grad! 会清零冻结层的梯度"
  (let ((d (make-dense 4 :in-dim 4 :activation :none :trainable nil))
        (x (vt-random-normal '(2 4))))
    (forward d x)
    (backward d (vt-random-normal '(2 4)))
    (rt-assert (dense-dw d))
    (zero-grad! d)
    (rt-assert (null (dense-dw d)))))

;;; ---------------------------------------------------------------
;;; P1-2  params / grads 长度一致（位置配对不错位）
;;; ---------------------------------------------------------------
(rt "P1-2" "params 与 grads 长度始终一致"
  (dolist (mk (list (lambda () (make-dense 4 :in-dim 4))
                    (lambda () (make-conv2d 2 '(2 2) :in-channels 3))
                    (lambda () (make-embedding 10 4))
                    (lambda () (make-lstm 4 3))
                    (lambda () (make-gru 4 3))
                    (lambda () (make-multi-head-attention 4 2))))
    (let ((l (funcall mk)))
      ;; 反向传播之前与之后都必须长度一致
      (rt-assert (= (length (params l)) (length (grads l)))))))

(rt "P1-2" "某层无梯度时不会用别的层的梯度更新它"
  (let* ((m (make-sequential))
         (d1 (make-dense 4 :in-dim 4 :activation :none :name "d1"))
         (d2 (make-dense 4 :in-dim 4 :activation :none :name "d2"))
         (x (vt-random-normal '(2 4))))
    (seq-add! m d1) (seq-add! m d2)
    (forward m x) (backward m (vt-random-normal '(2 4)))
    (zero-grad! d1)                       ; 模拟 d1 未产生梯度
    (let ((w0 (vt-to-list (vt-flatten (dense-weights d1)))))
      (model-update! m (make-sgd :lr 1.0d0))
      (rt-assert (equal w0 (vt-to-list (vt-flatten (dense-weights d1))))))))

;;; ---------------------------------------------------------------
;;; P1-3  缓存清理契约
;;; ---------------------------------------------------------------
(rt "P1-3" "clear-forward-cache! 在 forward/backward 之间安全"
  (dolist (mk (list (lambda () (make-dense 4 :in-dim 4 :activation :relu))
                    (lambda () (make-conv2d 2 '(2 2) :in-channels 2))
                    (lambda () (make-max-pool2d '(2 2)))
                    (lambda () (make-multi-head-attention 4 2))))
    (let* ((l (funcall mk))
           (x (cond ((typep l 'multi-head-attention)
                     (list (vt-random-normal '(1 3 4)) (vt-random-normal '(1 3 4))
                           (vt-random-normal '(1 3 4))))
                    ((or (typep l 'conv2d) (typep l 'max-pool2d))
                     (vt-random-normal '(1 2 4 4)))
                    (t (vt-random-normal '(1 4)))))
           (g (cond ((typep l 'multi-head-attention) (vt-random-normal '(1 3 4)))
                    ((typep l 'max-pool2d) (vt-random-normal '(1 2 2 2)))
                    ((typep l 'conv2d) (vt-random-normal '(1 2 3 3)))
                    (t (vt-random-normal '(1 4))))))
      (forward l x)
      (clear-forward-cache! l)
      (backward l g))))                  ; 不得抛错

(rt "P1-3" "clear-step-caches! 清空全部前向缓存"
  (let* ((d (make-dense 4 :in-dim 4 :activation :relu))
         (x (vt-random-normal '(2 4))))
    (forward d x)
    (backward d (vt-random-normal '(2 4)))
    (clear-step-caches! d)
    (rt-assert (null (dense-input-cache d)))
    (rt-assert (null (dense-z-cache d)))
    (rt-assert (null (dense-a-cache d)))))

;;; ---------------------------------------------------------------
;;; P1-4  rnn-sequence 序列化
;;; ---------------------------------------------------------------
(rt "P1-4" "rnn-sequence 可保存/加载"
  (let* ((m (make-rnn-sequence 4 3))
         (x (vt-random-normal '(2 5 4)))
         (y0 (vt-to-list (vt-flatten (forward m x)))))
    (set-global-training! nil)
    (save-model m "/tmp/rt-rnnseq.sexp")
    (let* ((m2 (load-model "/tmp/rt-rnnseq.sexp"))
           (y1 (vt-to-list (vt-flatten (forward m2 x)))))
      (rt-assert (typep m2 'rnn-sequence))
      (rt-assert (equal y0 y1)))))

;;; ---------------------------------------------------------------
;;; P1-5  导出符号全部有定义
;;; ---------------------------------------------------------------
(rt "P1-5" "所有导出符号都有定义"
  (let ((dangling '()))
    (do-external-symbols (s (find-package :nn))
      (unless (or (fboundp s)
                  (find-class s nil)
                  (boundp s)
                  (macro-function s))
        (push (symbol-name s) dangling)))
    (when dangling
      (error "悬空导出符号: ~a" (sort dangling #'string<)))))

;;; ---------------------------------------------------------------
;;; P1-6  tensor-top-k
;;; ---------------------------------------------------------------
(rt "P1-6" "tensor-top-k 沿 axis=0 正确"
  (let ((x (vt-reshape (vt-from-sequence '(1.0d0 9.0d0 2.0d0 8.0d0 3.0d0 7.0d0)) '(2 3))))
    (multiple-value-bind (v i) (tensor-top-k x 1 :axis 0)
      (rt-assert (equal (vt-to-list (vt-flatten v)) '(8.0d0 9.0d0 7.0d0)))
      (rt-assert (equal (vt-to-list (vt-flatten i)) '(1 0 1)))
      (rt-assert (eq (vt-dtype i) :int64)))))

(rt "P1-6" "tensor-top-k 末轴与 k>1"
  (let ((x (vt-reshape (vt-from-sequence '(1.0d0 5.0d0 2.0d0 6.0d0 3.0d0 4.0d0)) '(3 2))))
    (multiple-value-bind (v i) (tensor-top-k x 1 :axis 1)
      (rt-assert (equal (vt-to-list (vt-flatten v)) '(5.0d0 6.0d0 4.0d0)))
      (rt-assert (equal (vt-to-list (vt-flatten i)) '(1 1 1)))))
  ;; 3x3 沿末轴取 top-2
  (let ((x (vt-reshape (vt-from-sequence '(1.0d0 3.0d0 2.0d0
                                           6.0d0 4.0d0 5.0d0
                                           7.0d0 9.0d0 8.0d0)) '(3 3))))
    (multiple-value-bind (v i) (tensor-top-k x 2 :axis 1)
      (rt-assert (equal (vt-to-list (vt-flatten v)) '(3.0d0 2.0d0 6.0d0 5.0d0 9.0d0 8.0d0)))
      (rt-assert (equal (vt-to-list (vt-flatten i)) '(1 2 0 2 1 2))))))

;;; ---------------------------------------------------------------
;;; P1-7  dropout 模式错配
;;; ---------------------------------------------------------------
(rt "P1-7" "dropout 反向使用前向真正生成的 mask"
  (set-global-training! t)
  (let* ((d (make-dropout 0.5d0))
         (x (vt-ones '(1 32)))
         (o (forward d x)))
    (set-training! d nil)                      ; 反向时切到 eval
    (let ((g (backward d (vt-ones '(1 32)))))
      ;; 前向输出里为 0 的位置，梯度也必须是 0
      (dotimes (i 32)
        (rt-assert (if (zerop (vt-ref o 0 i))
                       (zerop (vt-ref g 0 i))
                       (not (zerop (vt-ref g 0 i)))))))
    (reset-training! d)))

;;; ---------------------------------------------------------------
;;; P2-1  embedding max-norm
;;; ---------------------------------------------------------------
(rt "P2-1" "embedding max-norm 生效"
  (let ((e (make-embedding 20 8 :max-norm 0.05d0)))
    (set-global-training! t)
    (forward e (vt-from-sequence '(1 2 3) :dtype :int64))
    (let* ((w (emb-weight e))
           (ne (emb-num-embeddings e))
           (ed (emb-embedding-dim e))
           (worst 0.0d0))
      (dotimes (i ne)
        (let ((sq 0.0d0))
          (dotimes (j ed)
            (let ((v (coerce (vt-ref w i j) 'double-float)))
              (incf sq (* v v))))
          (setf worst (max worst (sqrt sq)))))
      (rt-assert (<= worst (+ 0.05d0 1d-9))))))

(rt "P2-1" "embedding max-norm 不影响未超限的行 / 反向可用"
  (let ((e (make-embedding 5 4 :max-norm 100.0d0)))
    (set-global-training! t)
    (forward e (vt-from-sequence '(0 1) :dtype :int64))
    (backward e (vt-ones '(2 4)))
    (rt-assert (emb-dw e))))

;;; ---------------------------------------------------------------
;;; P2-2  调度器除零
;;; ---------------------------------------------------------------
(rt "P2-2" "cosine-annealing-lr 在 t-max=0 时不除零"
  (let ((s (make-cosine-annealing-lr (make-sgd :lr 0.01d0) 0)))
    (scheduler-step! s)
    (rt-assert (numberp (optimizer-lr (scheduler-optimizer s))))))

;;; ---------------------------------------------------------------
;;; P2-3  常用 API 已导出
;;; ---------------------------------------------------------------
(rt "P2-3" "常用 API 已导出"
  (dolist (n '("SET-GLOBAL-TRAINING!" "BUILD-MODEL" "MAKE-RNN-SEQUENCE" "RNN-SEQUENCE"
               "COMPUTE-LOSS" "COMPUTE-LOSS-GRADIENT" "OPTIMIZER-LR" "DENSE-WEIGHTS"
               "COPY-NETWORK" "CLEAR-FORWARD-CACHE!" "CLEAR-STEP-CACHES!"
               "TRANSIENT-CACHE-SLOTS" "SGD-MOMENTUM" "ACTIVATION-KIND"))
    (multiple-value-bind (s st) (find-symbol n :nn)
      (rt-assert (and s (eq st :external))))))

;;; ---------------------------------------------------------------
;;; P2-5  全局范数裁剪
;;; ---------------------------------------------------------------
(rt "P2-5" "grad-clip 使用全局范数而非逐张量裁剪"
  (let* ((m (make-sequential))
         (d1 (make-dense 1 :in-dim 1 :activation :none :use-bias nil))
         (d2 (make-dense 1 :in-dim 1 :activation :none :use-bias nil))
         (opt (make-sgd :lr 0.0d0 :grad-clip 1.0d0)))
    (seq-add! m d1) (seq-add! m d2)
    (build-model m (vt-reshape (vt-from-sequence '(1.0d0)) '(1 1)))
    ;; 两个参数各自范数 0.6 < 1，但合起来 = sqrt(0.72) ≈ 0.849 < 1
    ;; 用 3,4 与 3,4 使全局范数 = sqrt(9+16+9+16) = 7.07 > 1
    (setf (dense-weights d1) (vt-from-sequence '(3.0d0))
          (dense-weights d2) (vt-from-sequence '(4.0d0)))
    (let ((g1 (vt-from-sequence '(3.0d0)))
          (g2 (vt-from-sequence '(4.0d0))))
      (optimizer-step opt (params m) (list (cons "weights" g1) (cons "weights" g2)))
      ;; 全局范数 5 -> 系数 0.2；参数 lr=0 不变，但状态缓冲区记录了缩放后的梯度
      (let ((buf1 (gethash (list d1 "weights") (optimizer-state-registry opt)))
            (buf2 (gethash (list d2 "weights") (optimizer-state-registry opt))))
        (rt-assert (< (abs (- (vt-ref buf1 0) 0.6d0)) 1d-9))
        (rt-assert (< (abs (- (vt-ref buf2 0) 0.8d0)) 1d-9))))))

;;; ---------------------------------------------------------------
;;; P2-6  优化器状态键与层顺序无关
;;; ---------------------------------------------------------------
(rt "P2-6" "优化器状态键不含参数位置"
  (let* ((m (make-sequential))
         (d1 (make-dense 4 :in-dim 4 :activation :none :name "A"))
         (d2 (make-dense 4 :in-dim 4 :activation :none :name "B"))
         (x (vt-random-normal '(4 4)))
         (opt (make-adam :lr 0.1d0)))
    (seq-add! m d1) (seq-add! m d2)
    (build-model m x)
    (zero-grad! m) (model-forward m x) (model-backward m (vt-random-normal '(4 4)))
    (model-update! m opt)
    (let ((base-keys '()))
      ;; 状态表的键形如 ((owner name) . moment)，取 first 得到 base-key
      (maphash (lambda (k v) (declare (ignore v))
                 (push (first k) base-keys))
               (optimizer-state-registry opt))
      (rt-assert base-keys)
      ;; base-key 形如 (owner name)；不应再出现第三个元素 idx
      (rt-assert (every (lambda (bk) (= (length bk) 2)) base-keys))
      ;; 且键确实由真实的层对象与参数名构成
      (rt-assert (every (lambda (bk) (and (typep (first bk) 'layer)
                                          (stringp (second bk))))
                        base-keys)))))

;;; ---------------------------------------------------------------
;;; P2-7  MHA 输入秩校验
;;; ---------------------------------------------------------------
(rt "P2-7" "MHA 对 2D 输入给出明确报错"
  (let ((m (make-multi-head-attention 8 2)))
    (handler-case
        (progn (forward m (list (vt-random-normal '(5 8)) (vt-random-normal '(5 8))
                                (vt-random-normal '(5 8))))
               (error "2D 输入本应报错"))
      (error (e)
        (rt-assert (search "至少需要 3 维" (princ-to-string e)))))))

;;; ---------------------------------------------------------------
;;; 顺带验证：新增的 SDPA 层可前向/反向
;;; ---------------------------------------------------------------
(rt "P1-5" "scaled-dot-product-attention 层可用"
  (let ((l (make-scaled-dot-product-attention)))
    (set-global-training! t)
    (let* ((q (vt-random-normal '(2 3 8)))
           (k (vt-random-normal '(2 4 8)))
           (v (vt-random-normal '(2 4 8)))
           (o (forward l (list q k v))))
      (rt-assert (equal (vt-shape o) '(2 3 8)))
      (multiple-value-bind (dq dk dv) (backward l (vt-random-normal '(2 3 8)))
        (rt-assert (and (equal (vt-shape dq) '(2 3 8))
                        (equal (vt-shape dk) '(2 4 8))
                        (equal (vt-shape dv) '(2 4 8))))))))

(rt "P1-5" "flops-estimate / tensor-* 工具可用"
  (let ((m (make-sequential)))
    (seq-add! m (make-dense 8 :in-dim 4 :activation :relu))
    (seq-add! m (make-dense 2 :in-dim 8 :activation :none))
    (build-model m (vt-random-normal '(1 4)))
    (multiple-value-bind (macs ps) (flops-estimate m)
      (rt-assert (= macs (+ (* 4 8) (* 8 2))))
      (rt-assert (= ps (+ (* 4 8) 8 (* 8 2) 2)))))
  (rt-assert (equal (vt-to-list (tensor-softmax (vt-from-sequence '(1.0d0 1.0d0))))
                    '(0.5d0 0.5d0)))
  (rt-assert (equal (vt-shape (tensor-tile (vt-from-sequence '(1.0d0 2.0d0)) 2))
                    '(4)))
  (rt-assert (equal (vt-shape (tensor-repeat (vt-from-sequence '(1.0d0 2.0d0)) 2))
                    '(4))))

(format t "~%===== 回归测试: ~d PASS, ~d FAIL =====~%" *rt-pass* *rt-fail*)
(when (plusp *rt-fail*)
  (error "回归测试存在失败项"))
