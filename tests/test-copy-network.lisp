(in-package :clnn)

(defun test-copy-network ()
  (set-global-training! t)

  (let ((pass 0) (fail 0))
    (flet ((ck (name cond)
             (if cond
                 (progn (incf pass) (format t "  [PASS] ~a~%" name))
                 (progn (incf fail) (format t "  [FAIL] ~a~%" name)))))

      ;; 1. activation-layer：kind 保真
      (let* ((a (make-activation-layer :gelu :leaky-alpha 0.05d0))
             (b (copy-network a)))
        (ck "activation-layer kind 保真"
            (eq (activation-kind b) :gelu))
        (ck "activation-layer leaky-alpha 保真"
            (= (act-leaky-alpha b) 0.05d0)))

      ;; 2. dropout：p 保真
      (let* ((d (make-dropout 0.3d0))
             (c (copy-network d)))
        (ck "dropout p 保真" (= (dropout-p c) 0.3d0)))

      ;; 3. flatten：start-dim 保真
      (let* ((f (make-flatten :start-dim 2))
             (g (copy-network f)))
        (ck "flatten start-dim 保真"
            (= (flatten-start-dim g) 2)))

      ;; 4. max-pool2d：kernel/stride/padding 保真，且前向不崩
      (let* ((p (make-max-pool2d '(2 2) :stride '(2 2) :padding '(1 1)))
             (q (copy-network p))
             (x (vt-random-normal (list 1 3 8 8))))
        (ck "max-pool2d kernel-size 保真"
            (equal (pool-kernel-size q) '(2 2)))
        (ck "max-pool2d 前向不崩"
            (handler-case (progn (forward q x) t)
              (error () nil))))

      ;; 5. dense：权重深拷贝（不同底层数组）
      (let* ((d (make-dense 4 :in-dim 4 :activation :relu))
             (_ (forward d (vt-random-normal (list 2 4))))
             (c (copy-network d)))
        (ck "dense weights 深拷贝（非同一对象）"
            (not (eq (dense-weights d) (dense-weights c))))
        (ck "dense weights 数值相等"
            (< (abs (- (vt-item (vt-sum (dense-weights d)))
                       (vt-item (vt-sum (dense-weights c)))))
               1.0d-12)))

      ;; 6. batch-norm：running-mean / running-var 深拷贝
      (let* ((bn (make-batch-norm 4))
             (_ (forward bn (vt-random-normal (list 8 4))))
             (c (copy-network bn)))
        (ck "batch-norm running-mean 深拷贝"
            (and (bn-running-mean c)
                 (not (eq (bn-running-mean bn)
                          (bn-running-mean c)))))
        (ck "batch-norm running-var 深拷贝"
            (and (bn-running-var c)
                 (not (eq (bn-running-var bn)
                          (bn-running-var c))))))

      ;; 7. multi-head-attention：配置保真，前向不崩
      (let* ((m (make-multi-head-attention 8 2 :use-bias t))
             (c (copy-network m))
             (x (vt-random-normal (list 2 3 8))))
        (ck "mha embed-dim 保真" (= (mha-embed-dim c) 8))
        (ck "mha num-heads 保真" (= (mha-num-heads c) 2))
        (ck "mha 前向不崩"
            (handler-case
                (progn (forward c (list x x x)) t)
              (error () nil))))

      ;; 8. residual：block 递归复制
      (let* ((r (make-residual (make-dense 4 :in-dim 4)))
             (_ (forward r (vt-random-normal (list 2 4))))
             (c (copy-network r)))
        (ck "residual block 非 nil" (not (null (residual-block c))))
        (ck "residual 前向不崩"
            (handler-case
                (progn (forward c (vt-random-normal (list 2 4))) t)
              (error () nil))))

      ;; 9. transformer-block：内部子层齐全
      (let* ((tb (make-transformer-block 8 2 :ffn-dim 16))
             (c (copy-network tb))
             (x (vt-random-normal (list 2 3 8))))
        (ck "transformer-block mha 非 nil" (not (null (tb-mha c))))
        (ck "transformer-block ffn1 非 nil" (not (null (tb-ffn1 c))))
        (ck "transformer-block ln1 非 nil" (not (null (tb-ln1 c))))
        (ck "transformer-block 前向不崩"
            (handler-case (progn (forward c x) t)
              (error () nil))))

      ;; 10. sequential：递归子层完整
      (let* ((m (make-sequential))
             (_ (seq-add! m (make-dense 8 :in-dim 4 :activation :relu)))
             (_ (seq-add! m (make-dropout 0.2d0)))
             (_ (seq-add! m (make-dense 4 :activation :none)))
             (_ (forward m (vt-random-normal (list 2 4))))
             (c (copy-network m)))
        (ck "sequential 层数保真"
            (= (length (seq-layers c)) 3))
        (ck "sequential dropout p 保真"
            (= (dropout-p (second (seq-layers c))) 0.2d0)))

      (format t "~%copy-network 测试: ~d PASS, ~d FAIL~%" pass fail)
      (values pass fail))))

(test-copy-network)
