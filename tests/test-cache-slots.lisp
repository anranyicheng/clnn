
(in-package :clnn)

;;; ================================================================
;;; 缓存清理契约（修复后）
;;;
;;; 旧契约：clear-forward-cache! 清空全部前向缓存。
;;;         问题：dense / conv2d / max-pool2d / mha / ... 的前向缓存
;;;         全都是 backward 必需的，在 forward 与 backward 之间
;;;         调用它必然让 backward 崩溃。
;;;
;;; 新契约：
;;;   clear-forward-cache!  —— 只清 transient-cache-slots（默认 '()），
;;;                            可在任意时刻安全调用；
;;;   clear-step-caches!    —— 清空 cache-slots 列出的全部前向缓存，
;;;                            只能在一次完整的 forward+backward 之后调用。
;;; ================================================================

;; 1. clear-forward-cache! 在 forward 与 backward 之间必须安全
(let* ((d (make-dense 4 :in-dim 4 :activation :relu))
       (x (vt-random-normal (list 2 4))))
  (set-global-training! t)
  (forward d x)
  (clear-forward-cache! d)                      ; 旧实现会在这里埋雷
  (let ((g (backward d (vt-random-normal (list 2 4)))))
    (assert (vt-p g)))
  (format t "  clear-forward-cache! 在 forward/backward 之间安全: OK~%"))

;; 2. clear-step-caches! 在一次完整步骤后清空全部前向缓存
(let* ((d (make-dense 4 :in-dim 4 :activation :relu))
       (x (vt-random-normal (list 2 4))))
  (set-global-training! t)
  (forward d x)
  (backward d (vt-random-normal (list 2 4)))
  (clear-step-caches! d)
  (assert (null (dense-input-cache d)))
  (assert (null (dense-z-cache d)))
  (assert (null (dense-a-cache d)))
  (format t "  clear-step-caches! 清空 dense 前向缓存: OK~%"))

;; 3. batch-norm：缓存被清，反向依赖的整型/统计状态保留
(let* ((bn (make-batch-norm 4))
       (x (vt-random-normal (list 8 4))))
  (set-global-training! t)
  (forward bn x)
  (let ((bs-before (bn-batch-size bn))
        (rm-before (bn-running-mean bn)))
    (clear-step-caches! bn)
    (assert (null (bn-input-cache bn)))
    (assert (null (bn-xhat-cache bn)))
    (assert (null (bn-std-inv-cache bn)))
    (assert (= (bn-batch-size bn) bs-before))     ; 状态未被清
    (assert (eq (bn-running-mean bn) rm-before))  ; 统计量未被清
    (format t "  batch-norm 缓存清理 + 状态保留: OK~%")))

;; 4. 自定义层：实现 transient-cache-slots 才会被 clear-forward-cache! 清理
(defclass my-layer (layer)
  ((my-cache :initform nil :accessor my-cache)
   (my-state :initform 42 :accessor my-state)))

(defmethod transient-cache-slots ((l my-layer)) '(my-cache))
(defmethod cache-slots ((l my-layer)) '(my-cache))

(let ((l (make-instance 'my-layer)))
  (setf (my-cache l) (vt-zeros (list 3))
        (my-state l) 99)
  (clear-forward-cache! l)
  (assert (null (my-cache l)))
  (assert (= (my-state l) 99))
  (format t "  自定义层 transient-cache-slots 生效: OK~%"))

;; 5. 容器层递归：clear-step-caches! 会清到子层
(let* ((m (make-sequential))
       (d (make-dense 4 :in-dim 4 :activation :relu))
       (x (vt-random-normal (list 2 4))))
  (seq-add! m d)
  (set-global-training! t)
  (forward m x)
  (backward m (vt-random-normal (list 2 4)))
  (clear-step-caches! m)
  (assert (null (dense-input-cache d)))
  (format t "  容器层 clear-step-caches! 递归子层: OK~%"))
