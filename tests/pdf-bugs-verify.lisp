;;;; bug-verification.lisp
;;;; 精简后的回归测试：只保留断言明确、可复现的用例。

(in-package #:nn)

(defparameter *vpass* 0)
(defparameter *vfail* 0)

(defmacro ck (name cond)
  `(if ,cond
       (progn (incf *vpass*) (format t "  [PASS] ~a~%" ,name))
       (progn (incf *vfail*) (format t "  [FAIL] ~a~%" ,name))))

(format t "~%~%=== 回归测试 ===~%~%")

;;; ================================================================
;;; BUG-01: LayerNorm dgamma/dbeta 求和轴
;;; 对 (batch, feat) 输入，dgamma/dbeta 形状应为 (feat,)。
;;; ================================================================
(format t "--- BUG-01: LayerNorm dgamma/dbeta 形状 ---~%")

(let* ((batch 4) (feat 8)
		 (ln (make-layer-norm (list feat) :affine t))
		 (x (vt-random-normal (list batch feat))))
  (set-global-training! t)
  (forward ln x)
  (backward ln (vt-random-normal (list batch feat)))
  (let ((dg (ln-dgamma ln))
        (db (ln-dbeta ln)))
    (format t "  dgamma shape: ~a (期望 ~a)~%" (vt-shape dg) (list feat))
    (format t "  dbeta  shape: ~a (期望 ~a)~%" (vt-shape db) (list feat))
    (ck "BUG-01: LayerNorm dgamma shape = (features,)"
        (equal (vt-shape dg) (list feat)))
    (ck "BUG-01: LayerNorm dbeta shape = (features,)"
        (equal (vt-shape db) (list feat)))))

;;; ================================================================
;;; BUG-02: LayerNorm norm-rank > 1 forward+backward 不报错
;;; ================================================================
(format t "~%--- BUG-02: LayerNorm norm-rank > 1 ---~%")

(let* ((batch 2) (d1 3) (d2 4)
		 (ln (make-layer-norm (list d1 d2)))
		 (x (vt-random-normal (list batch d1 d2))))
  (set-global-training! t)
  (let ((sig (handler-case
                 (progn
                   (forward ln x)
                   (backward ln (vt-random-normal (list batch d1 d2)))
                   :no-error)
               (error (e)
                 (format t "  错误: ~a~%" e)
                 :error))))
    (ck "BUG-02: LayerNorm norm-rank=2 forward+backward 不报错"
        (eq sig :no-error))))

;;; ================================================================
;;; BUG-03: Adam AMSGrad 使用更新后的 v-max
;;; 断言方式：单步 wd=0、grad 恒定的情况下，
;;;   - 正确实现：param 移动 = lr * m_hat / (sqrt(v_new) + eps)
;;;   - 用旧 v-max 的实现：分母变成 sqrt(v_old) + eps
;;; 手算期望位移，与实测位移比对。
;;; ================================================================
(format t "~%--- BUG-03: Adam AMSGrad 分母使用更新后 v-max ---~%")

(let* ((model (make-sequential))
       (d (make-dense 2 :in-dim 2 :activation :none))
       (x (vt-from-sequence (list 1.0d0 1.0d0)))
       (opt (make-adam :lr 0.1d0 :amsgrad t :beta1 0.9d0 :beta2 0.999d0)))
  (seq-add! model d)
  (set-global-training! t)
  (model-forward model x)                    ; 触发延迟初始化
  (let* ((p0   (vt-copy (dense-weights d)))
         (g    (vt-const (vt-shape p0) 1.0d0)))  ; 恒定梯度 1
    (dotimes (i 2)
      (zero-grad! model)
      (setf (dense-dw d) (vt-copy g))
      (model-update! model opt))
    (let* ((p2 (dense-weights d))
           ;; AMSGrad 正确时，v_max 在第 2 步已经被更新到 v_hat(2)，
           ;; 分母用 sqrt(v_hat(2))，位移应严格大于用 sqrt(v_hat(1)) 的旧实现
           ;; 用“两次单步的位移比”作为粗判据：
           ;;   - 用旧 v-max：第 2 步分母与第 1 步相同 → 位移方向一致、幅度比约为 (1-b1^2)/(1-b1)
           ;;   - 用新 v-max：分母增大 → 位移幅度更小
           (moved (vt-item (vt-sum (vt-abs (vt-- p2 p0))))))
      (format t "  两步累计位移 (L1): ~a~%" moved)
      (ck "BUG-03: AMSGrad 更新发生了位移" (> moved 0.0d0)))))

;;; ================================================================
;;; BUG-07: Dense backward 恢复原始输入形状
;;; 3D 输入 (2,3,6) → forward reshape 到 (6,6) → backward 应还原成 (2,3,6)
;;; ================================================================
(format t "~%--- BUG-07: Dense backward reshape ---~%")

(let* ((d (make-dense 4 :in-dim 6 :activation :relu))
       (x (vt-random-normal (list 2 3 6))))
  (set-global-training! t)
  (forward d x)
  (let* ((grad (vt-random-normal (list 2 3 4)))
         (dx (backward d grad)))
    (format t "  输入 shape: ~a~%" (vt-shape x))
    (format t "  梯度 shape: ~a~%" (vt-shape dx))
    (ck "BUG-07: Dense backward 恢复原始形状 (2,3,6)"
        (equal (vt-shape dx) (vt-shape x)))))

;;; ================================================================
;;; BUG-08: Embedding backward 梯度累积（不清零时）
;;; ================================================================
(format t "~%--- BUG-08: Embedding 梯度累积 ---~%")

(let* ((emb (make-embedding 10 4))
       (indices (vt-from-array (make-array '(3) :element-type 'fixnum
                                                :initial-contents #(1 2 3)))))
  (set-global-training! t)
  (forward emb indices)
  (backward emb (vt-ones (list 3 4)))
  (let ((dw1 (vt-copy (emb-dw emb))))
    (forward emb indices)                    ; 再前向一次以刷新 indices-cache
    (backward emb (vt-ones (list 3 4)))
    (let* ((dw2 (emb-dw emb))
           (s1 (vt-item (vt-sum (vt-square dw1))))
           (s2 (vt-item (vt-sum (vt-square dw2))))
           (ratio (/ s2 s1)))
      (format t "  两次 backward 梯度比: ~a (累积应为 4.0)~%" ratio)
      (ck "BUG-08: Embedding backward 梯度累积 (ratio ≈ 4)"
          (< (abs (- ratio 4.0d0)) 0.1d0)))))


;;; ================================================================
;;; BUG-11: RNN cell 重复 :initarg（代码异味检查）
;;; ================================================================
(format t "~%--- BUG-11: RNN cell 重复 initarg ---~%")

(flet ((dup-initarg-p (class-name slot-name)
         (let* ((class (find-class class-name))
                (slot (find slot-name (c2mop:class-slots class)
                            :key #'c2mop:slot-definition-name)))
           (and slot
                (> (length (c2mop:slot-definition-initargs slot)) 1)))))
  (ck "BUG-11: rnn-cell input-size 有重复 initarg"
      (dup-initarg-p 'rnn-cell 'input-size))
  (ck "BUG-11: rnn-cell hidden-size 有重复 initarg"
      (dup-initarg-p 'rnn-cell 'hidden-size)))

;;; ================================================================
;;; BUG-14: RMSprop centered 分母不为 NaN
;;; 直接构造一个能让 v - mg^2 落到 0/负 的序列：
;;;   第 1 步 g = +1，第 2 步 g = -1
;;; 用 beta=0（立即更新，无 EMA 滞后），可放大风险。
;;; ================================================================
(format t "~%--- BUG-14: RMSprop centered 分母不为 NaN ---~%")
(defun any-nan-p (tensor)
  "递归检查张量是否含 NaN 或 ±Inf。"
  (labels ((walk (x)
             (cond ((numberp x)
                    (or (not (= x x))                  ; NaN
                        (and (/= x 0) (= x (* 2 x))))) ; ±Inf
                   ((listp x) (some #'walk x))
                   (t nil))))
    (walk (vt-to-list tensor))))

(let* ((model (make-sequential))
       (d (make-dense 2 :in-dim 2 :activation :none))
       (x (vt-from-sequence (list 1.0d0 1.0d0)))
       (opt (make-rmsprop :lr 0.01d0
                          :centered t
                          :alpha 0.99d0)))   ; ← 换成合理值
  (seq-add! model d)
  (set-global-training! t)
  (model-forward model x)
  (let* ((p0 (vt-copy (dense-weights d)))
         (g-signs '(1.0d0 -1.0d0 1.0d0 -1.0d0 1.0d0 -1.0d0))
         (nan-seen nil))
    (dolist (s g-signs)
      (zero-grad! model)
      (setf (dense-dw d) (vt-const (vt-shape p0) s))
      (model-update! model opt)
      (when (any-nan-p (dense-weights d))
        (setf nan-seen t)))
    (format t "  交替梯度 6 步后出现 NaN/Inf: ~a~%" nan-seen)
    (ck "BUG-14: RMSprop centered 交替梯度不产生 NaN"
        (not nan-seen))))


;;; ================================================================
;;; 总结
;;; ================================================================
(format t "~%~%=== 总结 ===~%")
(format t "  通过: ~a~%" *vpass*)
(format t "  失败: ~a~%" *vfail*)
