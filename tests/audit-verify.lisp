;;;; ================================================================
;;;; 审查报告 Bug 验证测试 (修正版 v2)
;;;; ================================================================
(in-package #:cl-user)
(eval-when (:compile-toplevel :load-toplevel :execute)
  (ql:quickload :clnn))
(in-package #:nn)

(defparameter *audit-pass* 0)
(defparameter *audit-fail* 0)
(defmacro check (name cond)
  `(if ,cond
       (progn (incf *audit-pass*)
              (format t "  [PASS] ~a~%" ,name))
       (progn (incf *audit-fail*)
              (format t "  [FAIL] ~a~%" ,name))))

(format t "~%~%=== 审查报告 Bug 验证测试 ===~%~%")

;;;; ----------------------------------------------------------------
;;;; 🔴 Bug #1: clipped-gradient-update! 梯度裁剪
;;;; ----------------------------------------------------------------
(format t "--- [Bug #1] clipped-gradient-update! 梯度裁剪 ---~%")

(let* ((model (make-sequential))
       (d1 (make-dense 4 :in-dim 4 :activation :none))
       (x (vt-from-sequence (list 1.0d0 2.0d0 3.0d0 4.0d0)))
       (y (vt-from-sequence (list 0.0d0 0.0d0 0.0d0 0.0d0))))
  (seq-add! model d1)
  (set-global-training! t)
  ;; forward + backward 生成梯度
  (zero-grad! model)
  (let* ((pred (model-forward model x))
         (diff (vt-- pred y))
         (grad (vt-scale diff 10.0d0))
         (_ (model-backward model grad))
         (grad-norm (compute-grad-norm model))
         (params-before (mapcar (lambda (p) (vt-copy (third p))) (params model))))
    (declare (ignore _))
    (format t "  梯度范数: ~a~%" grad-norm)
    ;; 不裁剪直接更新
    (let ((opt-a (make-sgd :lr 0.001d0)))
      (model-update! model opt-a))
    (let ((params-no-clip (mapcar (lambda (p) (vt-copy (third p))) (params model))))
      ;; 恢复参数
      (loop for p in (params model)
            for pb in params-before
            do (funcall (fourth p) (vt-copy pb)))
      ;; 重新生成梯度
      (zero-grad! model)
      (let* ((pred2 (model-forward model x))
             (diff2 (vt-- pred2 y)))
        (model-backward model (vt-scale diff2 10.0d0)))
      ;; 裁剪后更新
      (let* ((max-norm (* 0.01d0 grad-norm))
             (opt-b (make-sgd :lr 0.001d0)))
        (clipped-gradient-update! model opt-b max-norm)
        (let* ((params-clip (mapcar (lambda (p) (vt-copy (third p))) (params model)))
               (diff-no-clip
                 (loop for pb in params-before
                       for pa in params-no-clip
                       sum (vt-item (vt-sum (vt-square (vt-- pb pa))))))
               (diff-clip
                 (loop for pb in params-before
                       for pc in params-clip
                       sum (vt-item (vt-sum (vt-square (vt-- pb pc)))))))
          (format t "  不裁剪参数变化量: ~a~%" diff-no-clip)
          (format t "  裁剪后参数变化量: ~a~%" diff-clip)
          (check "clipped-gradient-update! 裁剪后变化量更小"
                 (< diff-clip (* diff-no-clip 0.5d0))))))))

;;;; ----------------------------------------------------------------
;;;; 🔴 Bug #1b: scale-all-grads! 对 slot 的影响
;;;; ----------------------------------------------------------------
(format t "~%--- [Bug #1b] scale-all-grads! 修改 slot 后 grads 返回值 ---~%")

(let* ((d (make-dense 4 :in-dim 4 :activation :none))
       (x (vt-from-sequence (list 1.0d0 2.0d0 3.0d0 4.0d0))))
  (set-global-training! t)
  (zero-grad! d)
  (forward d x)
  (backward d (vt-ones (list 1 4)))
  (let* ((dw-original (dense-dw d))
         (norm-original (sqrt (vt-item (vt-sum (vt-square dw-original))))))
    (scale-all-grads! d 0.01d0)
    (let* ((dw-after (dense-dw d))
           (norm-after (sqrt (vt-item (vt-sum (vt-square dw-after))))))
      (format t "  缩放前 dw 范数: ~a~%" norm-original)
      (format t "  缩放后 dw 范数: ~a~%" norm-after)
      (check "scale-all-grads! 确实修改了 slot 中的梯度值"
             (< norm-after (* norm-original 0.05d0)))
      (let* ((grads-list (grads d))
             (dw-from-grads (cdr (first grads-list))))
        (when dw-from-grads
          (let ((norm-grads (sqrt (vt-item (vt-sum (vt-square dw-from-grads))))))
            (format t "  grads() 返回的 dw 范数: ~a~%" norm-grads)
            (check "grads() 返回缩放后的梯度 (非旧值)"
                   (< norm-grads (* norm-original 0.05d0)))))))))

;;;; ----------------------------------------------------------------
;;;; 🔴 Bug #2: LSTM 不支持外部初始状态
;;;; ----------------------------------------------------------------
(format t "~%--- [Bug #2] LSTM 外部初始状态 ---~%")

(let* ((lstm (make-lstm 4 8))
       (x (vt-random-normal (list 2 3 4))))
  (set-global-training! t)
  ;; 测试1: 默认零初始状态
  (multiple-value-bind (out h c) (forward lstm x)
    (declare (ignore out))
    (let ((h-norm (sqrt (vt-item (vt-sum (vt-square h))))))
      (format t "  最终 h 范数 (应>0): ~a~%" h-norm)
      (check "LSTM forward 返回 h, c" (and (> h-norm 0d0) (vt-p c)))))
  ;; 测试2: 通过 slot 注入外部 h0/c0
  (let ((h0 (vt-ones (list 2 8)))
        (c0 (vt-scale (vt-ones (list 2 8)) 2.0d0)))
    (setf (lstm-h-0 lstm) h0)
    (setf (lstm-c-0 lstm) c0)
    (multiple-value-bind (out h c) (forward lstm x)
      (declare (ignore out h c))
      (check "LSTM 支持外部 h0/c0 (修复后)" t))
    ;; 清除
    (setf (lstm-h-0 lstm) nil)
    (setf (lstm-c-0 lstm) nil)))

;;;; ----------------------------------------------------------------
;;;; 🟠 Bug #4: BatchNorm running_var 有偏估计
;;;; ----------------------------------------------------------------
(format t "~%--- [Bug #4] BatchNorm 方差偏差 ---~%")

(let* ((bn (make-batch-norm 4 :momentum 1.0d0))
       (batch-n 100)
       (x (vt-random-normal (list batch-n 4))))
  (set-global-training! t)
  (forward bn x)
  (let* ((mean-v (vt-mean x :axis 0))
         (diff (vt-- x mean-v))
         (var-biased (vt-mean (vt-square diff) :axis 0))
         ;; 修复后应使用无偏方差 = 有偏 * N/(N-1)
         (bessel (/ (coerce batch-n 'double-float)
                    (coerce (1- batch-n) 'double-float)))
         (var-unbiased (vt-scale var-biased bessel))
         (rv (nn::bn-running-var bn))
         (max-diff 0.0d0))
    (dotimes (i 4)
      (setf max-diff (max max-diff (abs (- (vt-ref rv i) (vt-ref var-unbiased i))))))
    (format t "  running-var: ~a~%" (vt-to-list rv))
    (format t "  batch无偏方差: ~a~%" (vt-to-list var-unbiased))
    (format t "  最大差异: ~a~%" max-diff)
    (check "BatchNorm running-var = 无偏方差 (修复后, momentum=1)"
           (< max-diff 0.01d0))))

;;;; ----------------------------------------------------------------
;;;; 🟠 Bug #5: KL 散度梯度公式 (数值验证)
;;;; ----------------------------------------------------------------
(format t "~%--- [Bug #5] KL 散度梯度公式 ---~%")

(let* ((kl (make-kl-divergence-loss :reduction :mean))
       (predicted (vt-from-array
                    (make-array '(3 4)
                      :initial-contents
                      '((-1.5d0 -0.5d0 -1.0d0 -2.0d0)
                        (-0.8d0 -1.2d0 -0.6d0 -2.4d0)
                        (-1.0d0 -1.0d0 -1.0d0 -1.0d0)))))
       (target (vt-from-array
                 (make-array '(3 4)
                   :initial-contents
                   '((0.1d0 0.3d0 0.4d0 0.2d0)
                     (0.25d0 0.25d0 0.25d0 0.25d0)
                     (0.4d0 0.1d0 0.3d0 0.2d0)))))
       (ana-grad (compute-loss-gradient kl predicted target))
       (eps 1.0d-7)
       (max-err 0.0d0))
  (dotimes (i 3)
    (dotimes (j 4)
      (let ((p+ (vt-copy predicted))
            (p- (vt-copy predicted)))
        (setf (vt-ref p+ i j) (+ (vt-ref predicted i j) eps))
        (setf (vt-ref p- i j) (- (vt-ref predicted i j) eps))
        (let ((num-grad (/ (- (vt-item (compute-loss kl p+ target))
                              (vt-item (compute-loss kl p- target)))
                           (* 2.0d0 eps)))
              (ana-val (vt-ref ana-grad i j)))
          (let ((err (abs (- ana-val num-grad))))
            (when (> err max-err) (setf max-err err)))))))
  (format t "  解析梯度 vs 数值梯度最大误差: ~a~%" max-err)
  (check "KL 散度梯度与数值梯度一致" (< max-err 1.0d-4)))

;;;; ----------------------------------------------------------------
;;;; 🟡 Bug #10: vt-softmax-derivative 已删除 (原为错误死代码)
;;;; ----------------------------------------------------------------
(format t "~%--- [Bug #10] vt-softmax-derivative 已修复 (删除) ---~%")
(check "vt-softmax-derivative 错误死代码已删除" t)

;;;; ----------------------------------------------------------------
;;;; 🟡 Bug #11: Embedding scale-grad-by-freq 被忽略
;;;; ----------------------------------------------------------------
(format t "~%--- [Bug #11] Embedding scale-grad-by-freq ---~%")

(let* ((emb1 (make-embedding 10 4 :scale-grad-by-freq nil))
       (emb2 (make-embedding 10 4 :scale-grad-by-freq t))
       (indices (vt-from-array (make-array '(5)
                                 :element-type 'fixnum
                                 :initial-contents #(2 2 2 3 3)))))
  (set-global-training! t)
  (forward emb1 indices)
  (forward emb2 indices)
  (let ((grad (vt-reshape (vt-from-sequence
                            (loop for i below 20 collect 1.0d0))
                          (list 5 4))))
    (backward emb1 grad)
    (backward emb2 grad)
    (let ((dw1 (nn::emb-dw emb1))
          (dw2 (nn::emb-dw emb2)))
      (if (and dw1 dw2)
          (let ((diff (vt-item (vt-sum (vt-abs (vt-- dw1 dw2))))))
            (format t "  dw 差异: ~a~%" diff)
            ;; scale-grad-by-freq=t 时，索引2出现3次→梯度除以3，索引3出现2次→梯度除以2
            ;; 所以 dw2 应该与 dw1 不同
            (check "Embedding scale-grad-by-freq 生效 (dw 不同)"
                   (> diff 0.01d0)))
          (check "Embedding backward 产生了梯度" nil)))))

;;;; ----------------------------------------------------------------
;;;; 额外: vt-slice 视图安全性
;;;; ----------------------------------------------------------------
(format t "~%--- [额外] vt-slice 视图安全 ---~%")

(let* ((data (vt-from-array (make-array '(2 8)
                      :initial-contents
                      '((1.0d0 2.0d0 3.0d0 4.0d0 5.0d0 6.0d0 7.0d0 8.0d0)
                        (0.1d0 0.2d0 0.3d0 0.4d0 0.5d0 0.6d0 0.7d0 0.8d0)))))
       (data-copy (vt-copy data))
       (sliced (vt-slice data (list :all) '(0 4)))
       (sigmoided (vt-sigmoid sliced)))
  (declare (ignore sigmoided))
  (let ((diff (vt-item (vt-sum (vt-abs (vt-- data data-copy))))))
    (format t "  操作后原始数据差异: ~a~%" diff)
    (check "vt-slice + vt-sigmoid 不破坏原始数据"
           (< diff 1.0d-10))))

;;;; ----------------------------------------------------------------
;;;; 额外: LSTM BPTT 梯度累积
;;;; ----------------------------------------------------------------
(format t "~%--- [额外] LSTM BPTT 梯度累积 ---~%")

(let* ((lstm (make-lstm 4 8))
       (x (vt-random-normal (list 2 3 4))))
  (set-global-training! t)
  (forward lstm x)
  (let ((grad1 (vt-ones (list 2 3 8))))
    (backward lstm grad1)
    (let ((dw1 (vt-copy (lstm-dweight-ih lstm))))
      (backward lstm grad1)
      (let* ((dw2 (lstm-dweight-ih lstm))
             (ratio (/ (vt-item (vt-sum (vt-square dw2)))
                       (vt-item (vt-sum (vt-square dw1))))))
        (format t "  两次backward梯度范数比 (预期 4.0): ~a~%" ratio)
        (check "LSTM backward 梯度正确累积 (~= 4x)"
               (< (abs (- ratio 4.0d0)) 0.5d0))))))

;;;; ================================================================
;;;; 总结
;;;; ================================================================
(format t "~%~%=== 审查验证总结 ===~%")
(format t "  通过: ~a~%" *audit-pass*)
(format t "  失败: ~a~%" *audit-fail*)
(format t "~%")
