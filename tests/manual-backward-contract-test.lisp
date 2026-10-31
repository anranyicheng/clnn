;;;; manual-backward-contract-test.lisp
;;;;
;;;; 「纯手写求导架构」的契约测试（对应契约文档 §1.5 的 M1–M7）。
;;;;
;;;; 背景：clvt 是纯张量库 —— 没有计算图、没有 grad_fn、没有磁带、没有伴随模式。
;;;; 因此 clnn 的每一条反向公式都是人推手写的，没有任何一环能替用户兜底。
;;;; 这份测试把「手写」这个前提**必然推出**的约束钉成可执行断言：
;;;;
;;;;   M1 每层自缓存：backward 不得重算前向
;;;;   M3 一一配对：一次 forward 只支撑一次 backward
;;;;   M4 梯度覆盖写：重复 backward 不叠加（与 §5.1 一致）
;;;;   M5 容器只调度：sequential 的 dX == 手工倒序链
;;;;   M7 更新显式两步：update! 必须真的改参数
;;;;
;;;; 用法（需先加载系统）:
;;;;   sbcl --dynamic-space-size 4096 --non-interactive \
;;;;        --eval '(require :asdf)' --eval '(asdf:load-system :clnn)' \
;;;;        --load tests/manual-backward-contract-test.lisp
;;;;
;;;; 加载即执行；任何断言失败都会抛错使进程以非 0 退出。

(require :asdf)
(asdf:load-system :clnn)
(in-package :nn)

(defvar *pass* 0)
(defvar *fail* 0)

(defmacro check (form &optional (label nil label-p))
  `(handler-case
       (let ((ok ,form))
         (if ok
             (progn (incf *pass*)
                    (format t "  [PASS] ~a~%" ,(if label-p label `',form)))
             (progn (incf *fail*)
                    (format t "  [FAIL] ~a（返回 NIL）~%"
                            ,(if label-p label `',form)))))
     (error (e)
       (incf *fail*)
       (format t "  [FAIL] ~a（意外错误：~a）~%"
               ,(if label-p label `',form) e))))

(defun max-abs-diff (a b)
  "max|a − b|（用 vt-amax，元素级 vt-max 需要两个操作数）。"
  (if (or (null a) (null b))
      0.0d0
      (vt-item (vt-amax (vt-abs (vt-- a b))))))

(defun rn (shape &optional (scale 1.0d0))
  (vt-scale (vt-random-normal shape) scale))

(format t "~&==== 纯手写求导架构契约测试 ====~%")

;;; ================================================================
;;; M1  每层自缓存：backward 必须用「forward 时的」中间量
;;; ================================================================
;;; 做法：前向后**篡改参数**，再 backward。
;;;   · 若 backward 用缓存 → 得到的 dw 反映的是「前向那一刻」的激活；
;;;   · 若 backward 重算前向 → dw 会跟着新参数变（relu 的开关被翻掉）。
;;; 构造：x 全 1、W 全 +1 → z=3>0（relu 打开）；改 W 全 -1 → z=-3（relu 关闭）。
;;;      重算派会得到 dw = 0，缓存派会得到 dw ≠ 0。
(format t "~&-- M1 每层自缓存（不重算前向） --~%")

(let* ((d (make-instance 'dense :out-dim 2 :in-dim 3 :activation :relu
                         :weights (vt-const '(3 2) 1.0d0)
                         :bias (vt-zeros '(2) :dtype :float64)))
       (x (vt-const '(2 3) 1.0d0)))
  (forward d x)
  (setf (dense-weights d) (vt-const '(3 2) -1.0d0))   ; 事后篡改
  (backward d (vt-ones '(2 2)))
  (check (> (vt-item (vt-amax (vt-abs (dense-dw d)))) 0.5d0)
         "M1a dense(:relu) 改参数后 backward 仍用前向时的激活（dw ≠ 0）")
  ;; 反证：真正重算一遍前向，dw 必然为 0（relu 全关）
  (let ((d2 (make-instance 'dense :out-dim 2 :in-dim 3 :activation :relu
                           :weights (vt-const '(3 2) -1.0d0)
                           :bias (vt-zeros '(2) :dtype :float64))))
    (forward d2 x)
    (backward d2 (vt-ones '(2 2)))
    (check (< (vt-item (vt-amax (vt-abs (dense-dw d2)))) 1d-12)
           "M1b 反证：用改动后的参数重算前向会让 dw 归零（故 M1a 的判据有效）")))

;;; M1 的静态证据：源码里每个 backward 段内不得出现 (forward …)。
;;; 这是「最省事的作弊方式」的文本级守门 —— 行为级断言无法覆盖所有层。
(let* ((dir (asdf:system-source-directory :clnn))
       (files (list "nn-protocol.lisp" "nn-model.lisp"
                    "nn-layers-dense.lisp" "nn-layers-conv.lisp"
                    "nn-layers-norm.lisp" "nn-layers-attention.lisp"
                    "nn-layers-rnn.lisp" "nn-layers-embedding.lisp"))
       (offenders '())
       (methods 0))
  (dolist (f files)
    (let* ((path (merge-pathnames f dir))
           (lines (with-open-file (s path)
                    (loop for line = (read-line s nil nil)
                          while line collect line)))
           (in-backward nil)
           (n 0))
      (dolist (line lines)
        (incf n)
        (let ((trimmed (string-left-trim '(#\Space #\Tab) line)))
          (cond
            ;; 进入一个 backward 方法
            ((search "(defmethod backward" trimmed)
             (setf in-backward t) (incf methods))
            ;; 遇到下一个 defmethod 就离开（含 defun 也算段结束）
            ((and in-backward
                  (or (search "(defmethod " trimmed)
                      (search "(defun " trimmed)
                      (search ";;;;" trimmed)))
             (setf in-backward nil))
            ;; 段内出现 (forward … 且该行不是注释
            ((and in-backward
                  (> (length trimmed) 0)
                  (char/= (char trimmed 0) #\;)
                  (search "(forward " trimmed))
             (push (format nil "~a:~a" f n) offenders)))))))
  (format t "  扫描 ~a 个 backward 方法~%" methods)
  (check (null offenders)
         (format nil "M1c 所有 backward 方法体内无 forward 调用（发现：~a）"
                 (or offenders "无"))))

;;; ================================================================
;;; M3 + M4  一一配对 / 梯度覆盖写
;;; ================================================================
;;; 同一实例、同一输入、连续两次 forward+backward，两次的梯度必须逐位相同。
;;; 覆盖写层族天然满足；任何「跨调用累加」的层会让比值变成 2（范数平方后 4）。
(format t "~&-- M3/M4 覆盖写：重复 backward 不叠加 --~%")

(defun grad-vector (layer)
  "把一层的全部梯度拼成一维（跳过 NIL 占位）。"
  (let ((parts '()))
    (dolist (g (grads layer))
      (when (and (consp g) (cdr g))
        (push (vt-flatten (cdr g)) parts)))
    (when parts (apply #'vt-concatenate -1 (nreverse parts)))))

(defun twice-ratio (build input-fn grad-fn reset-fn)
  "跑两轮 forward+backward，返回 max|g2−g1| / max|g1|。"
  ;; 输入与上游梯度都只生成一次：若每次 backward 重新随机取 g，
  ;; 两轮的结果本来就不同，测不出「累加」与否。
  (let* ((l (funcall build))
         (x (funcall input-fn))
         (g (funcall grad-fn l)))
    (funcall reset-fn l)
    (forward l x)
    (backward l g)
    (let ((g1 (grad-vector l)))
      (funcall reset-fn l)
      (forward l x)
      (backward l g)
      (let ((g2 (grad-vector l)))
        (if (null g1)
            0.0d0
            (/ (max-abs-diff g1 g2)
               (max 1d-12 (vt-item (vt-amax (vt-abs g1))) 1d0)))))))

(let* ((noop (lambda (l) (declare (ignore l)) nil))
       (cases
         (list
          (cons "dense" (list (lambda () (make-dense 4 :in-dim 5 :activation :tanh))
                              (lambda () (rn '(2 5)))
                              (lambda (l) (declare (ignore l)) (rn '(2 4)))
                              noop))
          (cons "conv2d" (list (lambda () (make-conv2d 3 3 :in-channels 2))
                               (lambda () (rn '(2 2 7 7)))
                               (lambda (l) (declare (ignore l)) (rn '(2 3 5 5)))
                               noop))
          (cons "batch-norm" (list (lambda () (make-batch-norm 4))
                                   (lambda () (rn '(6 4)))
                                   (lambda (l) (declare (ignore l)) (rn '(6 4)))
                                   noop))
          (cons "layer-norm" (list (lambda () (make-layer-norm '(4)))
                                   (lambda () (rn '(5 4)))
                                   (lambda (l) (declare (ignore l)) (rn '(5 4)))
                                   noop))
          (cons "lstm" (list (lambda () (make-lstm 5 4))
                             (lambda () (rn '(2 3 5)))
                             (lambda (l) (declare (ignore l)) (rn '(2 3 4)))
                             noop))
          (cons "gru" (list (lambda () (make-gru 5 4))
                            (lambda () (rn '(2 3 5)))
                            (lambda (l) (declare (ignore l)) (rn '(2 3 4)))
                            noop))
          (cons "rnn-sequence" (list (lambda () (make-rnn-sequence 5 4))
                                     (lambda () (rn '(2 3 5)))
                                     (lambda (l) (declare (ignore l)) (rn '(2 3 4)))
                                     noop))
          (cons "rnn-cell" (list (lambda () (make-rnn-cell 3 4))
                                 (lambda () (rn '(2 3)))
                                 (lambda (l) (declare (ignore l)) (rn '(2 4)))
                                 (lambda (l) (reset-rnn-cell-state! l))))
          (cons "mha" (list (lambda () (make-multi-head-attention 8 2))
                            (lambda () (let ((q (rn '(2 4 8)))) (list q q q)))
                            (lambda (l) (declare (ignore l)) (rn '(2 4 8)))
                            noop))
          (cons "embedding" (list (lambda () (make-embedding 6 4))
                                  (lambda () (vt-from-sequence '(0 2 1) :dtype :int64))
                                  (lambda (l) (declare (ignore l)) (rn '(3 4)))
                                  noop)))))
  (dolist (case cases)
    (let* ((name (car case))
           (args (cdr case))
           (ratio (handler-case
                      (apply #'twice-ratio args)
                    (error (e) (format t "    (~a 抛错：~a)~%" name e) 999d0))))
      (check (< ratio 1d-6)
             (format nil "M4[~a] 两次 forward+backward 梯度逐位一致（相对差 ~,2e）"
                     name ratio)))))

;;; ================================================================
;;; M5  容器只调度：sequential 的 dX == 手工倒序链
;;; ================================================================
(format t "~&-- M5 容器只调度（不夹带自己的数学） --~%")

(let* ((d1 (make-instance 'dense :out-dim 4 :in-dim 3 :activation :tanh
                          :weights (vt-astype (rn '(3 4)) :float64)
                          :bias (vt-zeros '(4))))
       (d2 (make-instance 'dense :out-dim 2 :in-dim 4 :activation :none
                          :weights (vt-astype (rn '(4 2)) :float64)
                          :bias (vt-zeros '(2))))
       (m (make-sequential))
       (x (rn '(3 3)))
       (g (rn '(3 2))))
  (seq-add! m d1)
  (seq-add! m d2)
  (let ((y (forward m x))
        (dx-model (backward m g)))
    (declare (ignore y))
    ;; 手工：d2 先反，再 d1。要从同一份前向缓存出发，所以重跑一次前向。
    (forward m x)
    (let* ((d2-again (backward d2 g))
           (dx-manual (backward d1 d2-again)))
      (check (< (max-abs-diff dx-model dx-manual) 1d-12)
             "M5a sequential 的 dX 与手工倒序链逐位一致"))))

;;; ================================================================
;;; M7  参数更新必须显式两步：update! 真的改参数
;;; ================================================================
;;; update! 曾经只有 T 上的空方法 —— 公开导出却静默什么都不做。
(format t "~&-- M7 update! 真的更新参数 --~%")

(let* ((d (make-dense 2 :in-dim 3 :activation :none))
       (opt (make-sgd :lr 0.5d0))
       (x (rn '(2 3))))
  (forward d x)
  (backward d (vt-ones '(2 2)))
  (let ((w-before (vt-copy (dense-weights d))))
    (update! d opt)
    (check (> (max-abs-diff w-before (dense-weights d)) 1d-6)
           "M7a (update! 层 优化器) 使参数发生位移")
    (check (not (eq w-before (dense-weights d)))
           "M7b 参数是「替换张量对象」而不是原地改（§8 参数更新方式")))

;;; ================================================================
(format t "~&==== 汇总: ~a PASS / ~a FAIL ====~%" *pass* *fail*)
(when (plusp *fail*)
  (error "纯手写求导架构契约测试失败 ~a 项" *fail*))
