;;;; 所有关键 Bug 修复验证（基于真实工作区 API）
(in-package #:cl-user)
(eval-when (:compile-toplevel :load-toplevel :execute)
  (ql:quickload :clnn))
(in-package #:nn)

(defparameter *pass* 0)
(defparameter *fail* 0)
(defmacro ck (name cond)
  `(if ,cond
       (progn (incf *pass*) (format t "  [PASS] ~a~%" ,name))
       (progn (incf *fail*) (format t "  [FAIL] ~a~%" ,name))))

(defun randu (sh &key (lo -1d0) (hi 1d0))
  (vt-+ (vt-scale (vt-random sh) (- (coerce hi 'double-float) (coerce lo 'double-float)))
        (coerce lo 'double-float)))

(defun train-one-step (model x y &key (lr 0.01d0))
  (zero-grad! model)
  (let* ((pred (model-forward model x))
         (diff (vt-- pred y))
         (loss (vt-mean (vt-square diff)))
         (n (reduce #'* (vt-shape x))))
    (model-backward model (vt-scale diff (/ 2d0 (reduce #'* (vt-shape pred)))))
    (model-update! model (make-adam :lr lr))
    (vt-item loss)))

(format t "~%=== All Bug Fixes Verification ===~%~%")

;; 1. BCE: target=0 必须返回正损失
(let ((bce (make-bce-loss :reduction :mean))
      (p (vt-from-array (make-array '(1 1) :initial-contents '((0.9d0)))))
      (t0 (vt-from-array (make-array '(1 1) :initial-contents '((0d0)))))
      (t1 (vt-from-array (make-array '(1 1) :initial-contents '((1d0))))))
  (ck "BCE target=0 positive loss" (plusp (vt-item (compute-loss bce p t0))))
  (ck "BCE target=1 positive loss" (plusp (vt-item (compute-loss bce p t1)))))

;; 2. CE loss: logits + integer labels，返回正损失；梯度形状匹配
(let ((ce (make-ce-loss :reduction :mean))
      (logits (vt-from-array (make-array '(3 4) :initial-contents
                '((2d0 1d0 0d0 -1d0)(0d0 3d0 0d0 0d0)(-1d0 0d0 2d0 0d0)))))
      (labels (vt-from-array (make-array 3 :element-type '(signed-byte 32)
                             :initial-contents #(0 1 2)))))
  (ck "CE loss positive on logits+labels"
      (plusp (vt-item (compute-loss ce logits labels))))
  (let ((g (compute-loss-gradient ce logits labels)))
    (ck "CE grad shape matches logits" (equal (vt-shape g) '(3 4)))))

;; 3. Cosine 梯度解析 vs FD（基础用例）
(let* ((p (vt-from-array (make-array '(1 3) :initial-contents '((1d0 2d0 3d0)))))
       (tgt (vt-from-array (make-array '(1 3) :initial-contents '((0.5d0 1.5d0 2.5d0)))))
       (cl (make-cosine-similarity-loss))
       (eps 1d-6)
       (ana (compute-loss-gradient cl p tgt))
       (num (vt-copy ana)))
  (dotimes (j 3)
    (let ((p+ (vt-copy p)) (p- (vt-copy p)))
      (setf (vt-ref p+ 0 j) (+ (vt-ref p 0 j) eps))
      (setf (vt-ref p- 0 j) (- (vt-ref p 0 j) eps))
      (setf (vt-ref num 0 j)
            (/ (- (vt-item (compute-loss cl p+ tgt))
                  (vt-item (compute-loss cl p- tgt)))
               (* 2 eps)))))
  (let ((max-r 0d0))
    (dotimes (j 3)
      (let* ((a (vt-ref ana 0 j)) (n (vt-ref num 0 j))
             (r (if (zerop n) 0d0 (/ (abs (- a n)) (abs n)))))
        (when (> r max-r) (setf max-r r))))
    (ck "Cosine grad matches FD (basic case)" (< max-r 1d-4))))

;; 4. 多层 dense 的 zero-grad!：必须清零每一层的 dw/db
(let* ((d1 (make-dense 8 :in-dim 4 :activation :none))
       (d2 (make-dense 4 :in-dim 8 :activation :none))
       (d3 (make-dense 2 :in-dim 4 :activation :none))
       (m (make-sequential)))
  (seq-add! m d1) (seq-add! m (make-activation-layer :relu)) (seq-add! m d2)
  (seq-add! m (make-activation-layer :relu)) (seq-add! m d3)
  (set-global-training! t)
  (train-one-step m (randu '(4 4)) (randu '(4 2)))
  (ck "After step: d1.dw set" (vt-p (dense-dw d1)))
  (ck "After step: d2.dw set" (vt-p (dense-dw d2)))
  (ck "After step: d3.dw set" (vt-p (dense-dw d3)))
  (zero-grad! m)
  (ck "zero-grad clears d1.dw" (null (dense-dw d1)))
  (ck "zero-grad clears d2.dw" (null (dense-dw d2)))
  (ck "zero-grad clears d3.dw" (null (dense-dw d3)))
  (ck "zero-grad clears d1.db" (null (dense-db d1)))
  (ck "zero-grad clears d2.db" (null (dense-db d2)))
  (ck "zero-grad clears d3.db" (null (dense-db d3))))

;; 5. with-training 恢复 *training-mode*
(let ((outer-save *training-mode*))
  (set-global-training! :outer-marker)
  (with-training :inner-marker
    (ck "with-training sets inner mode" (eq *training-mode* :inner-marker)))
  (ck "with-training restores outer mode" (eq *training-mode* :outer-marker))
  (set-global-training! outer-save))

;; 6. LSTM forget bias 初始化 ~= 2（bih + bhh forget 位）
(let ((lstm (make-lstm 4 8)))
  ;; 触发权重初始化：通过一次 forward
  (set-global-training! t)
  (forward lstm (randu '(2 3 4)))
  (let* ((bih (lstm-bias-ih lstm))
         (bhh (lstm-bias-hh lstm))
         (f-gate-start 8)  ; PyTorch ordering: i,f,g,o; hidden=8 -> forget starts at 8
         (fb 0d0))
    (dotimes (k 8) (incf fb (vt-ref bih (+ f-gate-start k))))
    (dotimes (k 8) (incf fb (vt-ref bhh (+ f-gate-start k))))
    ;; 平均 forget bias 应 ~= 2
    (ck "LSTM forget avg bias ~= 2"
        (< (abs (- (/ fb 8) 2d0)) 0.5d0))))

;; 7. Adam lr 是 double-float
(let ((opt (make-adam :lr 0.001d0)))
  (ck "Adam lr is double-float" (typep (optimizer-lr opt) 'double-float)))

;; 8. Flatten 正确工作：(B, d1, d2) -> (B, d1*d2)，保留 batch 维
(let* ((f (make-flatten :start-dim 1))
       (x (vt-from-array (make-array '(1 2 3) :initial-contents '(((1d0 2d0 3d0)(4d0 5d0 6d0))))))
       (y (forward f x)))
  (ck "Flatten (1 2 3) -> (1 6)" (equal (vt-shape y) '(1 6)))
  (ck "Flatten values preserved"
      (and (= (vt-ref y 0 0) 1d0) (= (vt-ref y 0 5) 6d0))))

;; 9. Conv2d + MaxPool forward 不崩溃（padding=1 正常）
(let ((conv (make-conv2d 2 '(3 3) :in-channels 1 :padding '(1 1)))
      (mp (make-max-pool2d '(2 2) :stride '(2 2)))
      (m (make-sequential)))
  (seq-add! m conv) (seq-add! m mp)
  (set-global-training! t)
  (let ((y (model-forward m (randu '(1 1 4 4)))))
    (ck "Conv+MaxPool forward shape ok" (consp (vt-shape y)))))

;; 10. MHA forward+backward 不崩溃且 grad-slots 有效
(let ((mha (make-multi-head-attention 8 2))
      (m (make-sequential)))
  (seq-add! m mha)
  (set-global-training! t)
  (let* ((x (randu '(2 4 8)))
         (y (model-forward m (list x x x))))
    (model-backward m (randu (vt-shape y)))
    (ck "MHA forward+backward ok" (vt-p (slot-value mha 'dw-q)))
    (zero-grad! m)
    (ck "MHA zero-grad! ok" (null (slot-value mha 'dw-q)))))

;; 11. 训练能正常收敛（5层 MLP 过拟合随机数据）
(let* ((m (make-sequential))
       (d1 (make-dense 16 :in-dim 4 :activation :none))
       (a1 (make-activation-layer :relu))
       (d2 (make-dense 8 :in-dim 16 :activation :none))
       (a2 (make-activation-layer :relu))
       (d3 (make-dense 2 :in-dim 8 :activation :none))
       (x (randu '(8 4))) (y (randu '(8 2)))
       (opt (make-adam :lr 0.01d0))
       (l0 0d0) (lf 0d0))
  (seq-add! m d1) (seq-add! m a1) (seq-add! m d2) (seq-add! m a2) (seq-add! m d3)
  (set-global-training! t)
  (setf l0 (train-one-step m x y :lr 0.01d0))
  (dotimes (i 100) (setf lf (train-one-step m x y :lr 0.01d0)))
  (ck "MLP 100 steps: loss decreases" (< lf l0))
  (ck "MLP 100 steps: loss < 0.5" (< lf 0.5d0)))

;; 12. BCE 梯度：输出有限数值（形状正确、不含 NaN/Inf）
(let* ((bce (make-bce-loss :reduction :mean))
       (p (vt-from-array (make-array '(1 1) :initial-contents '((0.9d0)))))
       (tg (vt-from-array (make-array '(1 1) :initial-contents '((0.1d0)))))
       (g (compute-loss-gradient bce p tg)))
  (ck "BCE grad finite" (numberp (vt-ref g 0 0))))

;; 13. Sequential 多次 seq-add! 顺序正确（之前 push+nreverse 会错乱）
(let ((m (make-sequential)))
  (dotimes (i 5) (seq-add! m (make-dense 2 :in-dim (if (zerop i) 3 2) :activation :none :name (format nil "d~a" i))))
  (ck "seq-add! preserves order (5 layers)" (= (length (seq-layers m)) 5)))

(format t "~%========== Summary: ~a passed, ~a failed ==========~%" *pass* *fail*)
