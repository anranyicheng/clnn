
(in-package :clnn)

;; 1. 常规层：前向+反向+清理后缓存应全为 nil
(let* ((d (make-dense 4 :in-dim 4 :activation :relu))
       (x (vt-random-normal (list 2 4))))
  (set-global-training! t)
  (forward d x)
  (backward d (vt-random-normal (list 2 4)))
  (clear-forward-cache! d)
  (assert (null (dense-input-cache d)))
  (assert (null (dense-z-cache d)))
  (assert (null (dense-a-cache d)))
  ;; 反向依赖的整型状态不受影响（dense 无此类 slot）
  (format t "  dense 缓存清理: OK~%"))

;; 2. batch-norm：缓存被清，反向依赖状态保留
(let* ((bn (make-batch-norm 4))
       (x (vt-random-normal (list 8 4))))
  (set-global-training! t)
  (forward bn x)
  (let ((bs-before (bn-batch-size bn))
        (rm-before (bn-running-mean bn)))
    (clear-forward-cache! bn)
    (assert (null (bn-input-cache bn)))
    (assert (null (bn-xhat-cache bn)))
    (assert (null (bn-std-inv-cache bn)))
    (assert (= (bn-batch-size bn) bs-before))     ; ★ 未被清
    (assert (eq (bn-running-mean bn) rm-before))  ; ★ 未被清
    (format t "  batch-norm 缓存清理 + 状态保留: OK~%")))

;; 3. 自定义层：只需实现 cache-slots 即可被清理
(defclass my-layer (layer)
  ((my-cache :initform nil :accessor my-cache)
   (my-state :initform 42 :accessor my-state)))

(defmethod cache-slots ((l my-layer)) '(my-cache))

(let ((l (make-instance 'my-layer)))
  (setf (my-cache l) (vt-zeros (list 3))
        (my-state l) 99)
  (clear-forward-cache! l)
  (assert (null (my-cache l)))
  (assert (= (my-state l) 99))
  (format t "  自定义层 cache-slots 生效: OK~%"))
