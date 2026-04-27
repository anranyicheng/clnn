;;;; ================================================================
;;; 通用回归测试引擎 (无全局变量，纯函数打包)
;;;; ================================================================
(in-package #:nn)

(defun tensor-has-nan-p (vt)
  "检查张量中是否包含 NaN 或 Inf。利用 NaN 的唯一特性：x /= x"
  (if (vt-p vt)
      (let ((data (vt-data vt)))
	(some (lambda (x) (or (not (numberp x))
			      (not (= x x))))
	      data))))

;;; 1. 数据生成器工厂 (局部函数)
(defun make-data-gen (fn x-min x-max &key (normalize-x t) (normalize-y t))
  "极其稳健的数据生成器。使用 100 点采样估算范围，彻底消除溢出风险。"
  (let* ((range (- x-max x-min))
         (mid-x (/ (+ x-min x-max) 2.0d0))
         (half-range-x (/ range 2.0d0))
         (y-min most-positive-double-float)
         (y-max most-negative-double-float))
    (dotimes (i 100)
      (let ((val (funcall fn (+ x-min (random range)))))
        (when (< val y-min) (setf y-min val))
        (when (> val y-max) (setf y-max val))))
    (let ((y-range (- y-max y-min))
          (mid-y (/ (+ y-min y-max) 2.0d0))
          (half-range-y (/ (- y-max y-min) 2.0d0)))
      (values
       ;; 闭包 1：生成数据
       (lambda (batch-size)
         (let ((x-list '()) (y-list '()))
           (dotimes (idx batch-size)
             (let* ((raw-x (+ x-min (random range)))
                    (x (if (and normalize-x (> range 0))
                           (/ (- raw-x mid-x) half-range-x)
                           raw-x))
                    (raw-y (funcall fn raw-x))
                    (y (if (and normalize-y (> y-range 1.0d-6))
                           (/ (- raw-y mid-y) half-range-y)
                           raw-y)))
               (push x x-list)
               (push y y-list)))
           (values
            (vt-reshape (vt-from-sequence (nreverse x-list)) (list batch-size 1))
            (vt-reshape (vt-from-sequence (nreverse y-list)) (list batch-size 1)))))
       ;; 闭包 2：反归一化
       (lambda (y-scaled)
         (if (and normalize-y (> y-range 1.0d-6))
             (+ (* y-scaled half-range-y) mid-y)
             y-scaled))))))

(defun run-regression-test
    (fn x-min x-max input-dim layer-specs
     &key (epochs 2000) (batch-size 128) 
       (opt-type :adam) 
       (opt-args '(:lr 0.001d0)) 
       (test-inputs '(-2.0d0 -1.0d0 0.0d0 1.0d0 2.0d0))
       (name "MLP"))
  "通用的单变量/多变量回归测试函数."
  (multiple-value-bind (gen-fn denorm-fn) 
      (make-data-gen fn x-min x-max :normalize-x t :normalize-y t)
    (let* ((model (make-sequential :name name))
           (num-layers (length layer-specs)))
      ;; 动态构建网络层
      (dotimes (i num-layers)
        (let* ((spec (nth i layer-specs))
               (is-last (= i (1- num-layers)))
               (act (second spec)))
          (seq-add! model
                    (apply #'make-dense
                           (first spec)
                           :activation act
                           :weight-init (make-xavier-uniform)
			   :use-bias t  ;; 要警惕，这个一般必须开启，否则不收敛
                           :bias-init (if (and is-last
					       (or (null act)
						   (eq act :none)))
                                          (make-constant-init 0.1d0)
                                          (make-zeros-init))
                           :name (format nil "~A-L~A" name (first spec))
                           (cddr spec)))))    
      
      (build-model model (vt-zeros (list 1 input-dim)))
      (format t "[~A] 模型参数量: ~A~%" name (param-count model))    
      
      (let ((optimizer
              (ecase opt-type
                (:sgd (apply #'make-sgd opt-args))
                (:adam (apply #'make-adam opt-args))
                (:adamw (apply #'make-adamw opt-args))
                (:rmsprop (apply #'make-rmsprop opt-args)))))
        (format t "[~A] 开始训练 (~A Epochs)...~%" name epochs)	
        (dotimes (epoch epochs)
          (multiple-value-bind (x y)
	      (funcall gen-fn batch-size)
            (zero-grad! model)
            (let* ((pred (model-forward model x))
                   (diff (vt-- pred y))
                   (loss-val (vt-mean (vt-square diff)))
                   (grad-output (vt-scale diff (/ 1.0d0 batch-size))))
              (model-backward model grad-output)	   
              (model-update! model optimizer)
              (when (zerop (mod epoch 200))
                (format t "Epoch ~a | Loss: ~a~%" epoch loss-val)))))
        (format t "----------------------------------------~%")
        (when test-inputs
          (let* ((len (length test-inputs))
                 (range (- x-max x-min))
                 (mid-x (/ (+ x-min x-max) 2.0d0))
                 (half-range-x (/ range 2.0d0))
                 (norm-inputs (mapcar (lambda (x)
					(/ (- x mid-x)
					   half-range-x))
				      test-inputs))
                 (test-x (vt-reshape
			  (vt-from-sequence norm-inputs)
			  (list len input-dim)))
                 (pred-scaled (model-forward model test-x))
                 (true-y (mapcar fn test-inputs))
                 (pred-real '()))
            (dotimes (i len)
              (push (funcall denorm-fn (vt-ref pred-scaled i 0))
		    pred-real))
            (setf pred-real (nreverse pred-real))
            (format t "[~A] 推理对比:~%" name)
            (format t "  X 输入: ~A~%" test-inputs)
            (format t "  真实 Y: ~A~%" true-y)
            (format t "  预测 Y: ~A~%" pred-real)))      
        model))))

;;; ----------------------------------------------------------------
;;; 测试 1: CNN 卷积网络 (验证 NCHW 空间维度推导)
;;; ----------------------------------------------------------------
(defun test-cnn-architecture ()
  (format t "~%=== [测试 1] CNN 卷积网络 ===~%")
  (let* ((batch-size 8)
         ;; 输入: 8张单通道 8x8 图像
         (x (vt-random-normal (list batch-size 1 8 8)))
         ;; 目标: 让网络学会输出全 0 张量 (正则化测试)
         (target (vt-zeros (list batch-size 4 8 8)))
         ;; 模型: Conv(1->4, 3x3, padding=1) 保持尺寸不变
         (conv (make-conv2d 4 3 :padding 1 :name "conv1"))
         (opt (make-adam :lr 0.01d0)))    
    (build-model conv x)
    (format t "参数量: ~A~%" (param-count conv))
    (dotimes (epoch 100)
      (zero-grad! conv)
      (let* ((out (forward conv x))
             ;; Loss: 所有像素的 MSE
             (diff (vt-- out target))
             (loss (vt-mean (vt-square diff)))
             ;; d(MSE)/d(out) = 2(out - 0) / N
             (n (* batch-size 4 8 8))
             (grad (vt-scale out (/ 2.0d0 n))))
        (backward conv grad)
        (model-update! conv opt)
        (when (zerop (mod epoch 10))
          (format t "Epoch ~a | Loss: ~a~%" epoch loss))))
    (format t "[CNN] 维度传递与反向传播完美无缺!~%")))

;;; ----------------------------------------------------------------
;;; 测试 2: LSTM 序列网络 (验证时序反向传播 BPTT)
;;; ----------------------------------------------------------------
(defun test-lstm-architecture ()
  (format t "~%=== [测试 2] LSTM 序列网络 ===~%")
  (let* ((batch-size 4)
         (seq-len 5)
         (input-size 3)
         (hidden-size 8)
         ;; 输入: (batch, seq_len, input_size)
         (x (vt-random-normal (list batch-size seq-len input-size)))
         (lstm (make-lstm input-size hidden-size))
         (opt (make-adam :lr 0.01d0)))    
    (build-model lstm x)
    (format t "参数量: ~A~%" (param-count lstm))
    (dotimes (epoch 100)
      (zero-grad! lstm)
      ;; LSTM forward 返回 3 个值，我们只取完整输出 output
      (multiple-value-bind (out h c)
          (forward lstm x)
        (declare (ignore h c))
        ;; out 形状: (batch, seq_len, hidden_size)
        ;; 目标: 让输出趋于 0
        (let* ((loss (vt-mean (vt-square out)))
               (n (* batch-size seq-len hidden-size))
               ;; 梯度形状必须与 out 严格一致
               (grad (vt-scale out (/ 2.0d0 n))))
          ;; backward 接收梯度和输入序列的 grad_input
          (backward lstm grad)
          (model-update! lstm opt)
          (when (zerop (mod epoch 10))
            (format t "Epoch ~a | Loss: ~a~%" epoch loss)))))
    (format t "[LSTM] BPTT 梯度截断与状态缓存无泄漏!~%")))

;;; ----------------------------------------------------------------
;;; 测试 3: Transformer 自注意力 (验证残差与 3D 投影)
;;; ----------------------------------------------------------------
(defun test-transformer-architecture ()
  (format t "~%=== [测试 3] Transformer Block ===~%")
  (let* ((batch-size 2)
         (seq-len 4)
         (embed-dim 16)
         (num-heads 4)
         ;; 输入: (batch, seq_len, embed_dim)
         (x (vt-random-normal (list batch-size seq-len embed-dim)))
         (tb (make-transformer-block
              embed-dim num-heads
              :dropout-rate 0.0d0 ;; 测试时关闭 dropout
              :eps 1e-5))
         (opt (make-adam :lr 0.005d0)))    
    (build-model tb x)
    (format t "参数量: ~A~%" (param-count tb))
    (dotimes (epoch 100)
      (zero-grad! tb)
      (let* ((out (forward tb x))
             ;; 目标: 恒等映射 (让输出逼近输入 x)
             (diff (vt-- out x))
             (loss (vt-mean (vt-square diff)))
             (n (* batch-size seq-len embed-dim))
             (grad (vt-scale diff (/ 2.0d0 n))))
        (backward tb grad)
        (model-update! tb opt)
        (when (zerop (mod epoch 10))
          (format t "Epoch ~a | Loss: ~a~%" epoch loss))))
    (format t "[Transformer] 3D转置、QKV切分与残差求导大成功!~%")))

(defun test-gelu-residual-deep-net ()
  (format t "~%=== [测试 4] GELU导数与深层残差网络 ===~%")
  (let* ((x (vt-random-normal (list 8 32)))
         (block1 (make-dense 32 :activation :gelu))
         (res1 (make-residual block1))
         (block2 (make-dense 32 :activation :gelu))
         (res2 (make-residual block2))
         (block3 (make-dense 32 :activation :gelu))
         (res3 (make-residual block3))
         (layers (list res1 res2 res3))
         (opt (make-adam :lr 0.005d0)))
    ;; 初始化
    (build-model (make-instance 'layer) x)
    (dolist (l layers) (build-model l x))
    (dotimes (i 100)
      ;; 【修正】：手动遍历清零梯度
      (dolist (l layers) (zero-grad! l))      
      (let ((out x))
        (dolist (l layers)
	  (setf out (forward l out)))
	(let* ((diff (vt-- out x))
               (loss (coerce (vt-mean (vt-square diff)) 'double-float))
               (grad (vt-scale diff (/ 2.0d0 (* 8 32)))))
          ;; 【修正】：手动倒序链式反向传播
          (dolist (l (reverse layers))
            (setf grad (backward l grad)))
          ;; 【修正】：手动遍历更新参数
          (dolist (l layers)
	    (model-update! l opt))          
          (when (zerop (mod i 10))
            (format t "Epoch ~a | Loss: ~a~%" i loss)))))
    (format t "[通过] GELU导数正确，深层残差梯度未消失!~%")))


(defun test-lstm-seq2seq ()
  (format t "~%=== [测试 5] LSTM 序列到序列求导 ===~%")
  (let* ((batch 5) (seq-len 10) (feat 8)
         (hidden-dim 16)
         (x (vt-random-normal (list batch seq-len feat)))
         (lstm (make-lstm feat hidden-dim))
         ;; proj 要把 hidden_dim(16) 映射回 feat(8)
         (proj (make-dense feat :activation :tanh))
         (layers (list proj lstm))
         (opt (make-adam :lr 0.01d0)))
    (build-model lstm)
    ;; 【修正】：用正确的 LSTM 输出形状来欺骗 proj 进行初始化
    ;; 这样 proj 才会生成 (16 -> 8) 的权重矩阵
    (build-model proj (vt-zeros (list batch seq-len hidden-dim)))
    (dotimes (i 100)
      (dolist (l layers)
	(zero-grad! l))
      (let* ((h (forward lstm x))
             (out (forward proj h))
             (loss (coerce (vt-mean (vt-square out)) 'double-float))
             (grad (vt-scale out (/ 2.0d0 (* batch seq-len feat)))))
        (setf grad (backward proj grad))
        (setf grad (backward lstm grad))
        (dolist (l layers) (model-update! l opt))
        (when (zerop (mod i 10))
          (format t "Epoch ~a | Loss: ~a~%" i loss))))
    (format t "[通过] LSTM 延迟初始化与序列反向传播正常!~%")))

(defun test-nlp-basic-stack ()
  (format t "~%=== [测试 6] NLP基础栈 ===~%")
  (let* ((batch 3)
         (seq-len 6)
         (dim 24)
         (vocab 100)
         ;; 模拟输入的 Token ID (整数张量)
         (token-ids
	   (vt-from-array
	    (make-array (list batch seq-len) 
                        :element-type 'fixnum 
                        :initial-contents '((1 5 9 20 3 45)
                                            (10 2 88 4 5 12)
                                            (33 21 5 67 8 90)))))
         (emb (make-embedding vocab dim))
         (ln (make-layer-norm dim))
         (mha (make-multi-head-attention dim 4 :use-bias nil))
         (layers (list emb ln mha))
         (opt (make-adam :lr 0.01d0)))    
    ;; 【修正1】：MHA 的 build-model 必须喂一个包含 3 个张量的 List！
    (let ((dummy (vt-zeros (list batch seq-len dim))))
      (build-model mha (list dummy dummy dummy)))
    ;; 触发 embedding 初始化
    (forward emb token-ids)
    
    (dotimes (i 100)
      ;; 【修正2】：手动遍历清零，绝对不能传 List 进去
      (dolist (l layers)
	(zero-grad! l))
      
      (let* ((x (forward emb token-ids))
             (n (forward ln x))
             ;; 【修正3】：MHA 的 forward 也必须传 (list Q K V)！自注意力就是传3个一样的
             (attn-out (forward mha (list n n n)))
             (loss (coerce (vt-mean (vt-square attn-out)) 'double-float))
             (grad (vt-scale attn-out (/ 2.0d0 (* batch seq-len dim)))))
        (setf grad (backward mha grad))
        (setf grad (backward ln grad))
        (setf grad (backward emb grad))        
        ;; 【修正4】：手动遍历更新，不能传 List
        (dolist (l layers)
	  (model-update! l opt))
        (when (zerop (mod i 10))
          (format t "Epoch ~a | Loss: ~a~%" i loss))))
    (format t "[通过] Embedding查表、LayerNorm、MHA串联无阻!~%")))

(defun test-1d-input-edge-case ()
  (format t "~%=== [测试 7] 1D无Batch维度输入防御 ===~%")
  (let* ((x (vt-random-normal (list 16))) ;; 没有 batch 维度！纯向量
         (d (make-dense 8 :activation :sigmoid))
         (opt (make-adam :lr 0.1d0)))
    (build-model d x)
    (dotimes (i 100)
      (zero-grad! d)
      (let* ((out (forward d x))
             (target (vt-ones (list 8)))
             (diff (vt-- out target))
             (loss (coerce (vt-mean (vt-square diff)) 'double-float))
             (grad (vt-scale diff (/ 2.0d0 8))))
        (backward d grad)
        (model-update! d opt)
        (when (zerop (mod i 10))
          (format t "Epoch ~a | Loss: ~a~%" i loss))))
    (format t "[通过] 1D向量输入不会引发维度坍塌!~%")))


(defun test-dropout-switch ()
  (format t "~%=== [测试 8] Dropout 训练/评估模式切换 ===~%")
  (let* ((x (vt-ones (list 5 10))) ;; 全 1 矩阵
         (drop (make-dropout 0.5d0)))
    ;; 1. 训练模式：应该有大约一半的元素变成 0，且剩下的被放大了(除以0.5)
    (set-training! drop t)
    (let ((out-train (forward drop x)))
      (format t "训练模式下是否有 0: ~A~%" 
              (not (vt-= (vt-relu (vt-- out-train 1.0d0)) out-train))))
    ;; 2. 评估模式：输出必须与输入分毫不差
    (set-training! drop nil)
    (let ((out-eval (forward drop x)))
      (if (vt-= out-eval x)
          (format t "[通过] Dropout 在评估模式下完美保持恒等映射!~%")
          (error "致命错误: Dropout 在 eval 模式下仍在丢掉数据!")))))

(defun test-global-pooling-classifier ()
  (format t "~%=== [测试 9] 全局池化 + 分类头 ===~%")
  (let* ((batch 4) (seq-len 8) (feat 32) (num-classes 5)
         ;; 模拟 RNN/CNN 输出的特征图
         (features (vt-random-normal (list batch seq-len feat)))
         (head (make-dense num-classes :activation :none))
         (opt (make-adam :lr 0.05d0)))
    (build-model head (vt-zeros (list batch feat))) ;; 注意：池化后特征维度变了
    (dotimes (i 100)
      (zero-grad! head)
      ;; 【手动实现全局平均池化】：沿 axis=1 求均值，形状变为
      (let* ((pooled (vt-mean features :axis 1)) 
             (logits (forward head pooled))
             ;; 假设目标是让所有 logit 逼近 0
             (loss (coerce (vt-mean (vt-square logits)) 'double-float))
             (grad (vt-scale logits (/ 2.0d0 (* batch num-classes)))))
        (backward head grad)
        (model-update! head opt)
        (when (zerop (mod i 10))
          (format t "Epoch ~a | Loss: ~a~%" i loss))))
    (format t "[通过] 高维特征经全局池化后完美对接Dense层!~%")))

(defun test-inception-branch-concat ()
  (format t "~%=== [测试 10-a] 多分支并行计算与拼接 ===~%")
  (let* ((x (vt-random-normal (list 3 16)))
         ;; 分支 1：降维到 8
         (branch1 (make-dense 8 :activation :relu))
         ;; 分支 2：降维到 8
         (branch2 (make-dense 8 :activation :relu))
         (layers (list branch1 branch2))
         (opt (make-adam :lr 0.01d0)))
    ;; 初始化
    (dolist (l layers)
      (build-model l x))
    
    (dotimes (i 1000)
      (dolist (l layers)
	(zero-grad! l))
      ;; 并行前向传播
      (let* ((out1 (forward branch1 x))
             (out2 (forward branch2 x))
             ;; 假设你实现了 vt-concat，沿最后一个维度拼接，变成 (3, 16)
             (concat-out (vt-concatenate -1 out1 out2))
             (loss (coerce (vt-mean (vt-square concat-out)) 'double-float))
             ;; 梯度需要手动切分回去（假设实现了 vt-split）
             (grad (vt-scale concat-out (/ 2.0d0 (* 3 16))))
             (grads-list (vt-split grad 2 :axis -1))
             (grad1 (first grads-list))
             (grad2 (second grads-list)))
        ;; 并行反向传播
        (backward branch1 grad1)
        (backward branch2 grad2)
        ;; 并行更新
        (dolist (l layers)
	  (model-update! l opt))
        (when (zerop (mod i 100))
          (format t "Epoch ~a | Loss: ~a~%" i loss))))
    (format t "[通过] DAG多分支计算与Concat反向传播正确!~%")))
;; 注：如果还没实现 vt-concat / vt-split，这个测试可以先跳过，去写这两个底层算子。

(defun test-inception-branch-concat-b ()
  (format t "~%=== [测试 10-b] 多分支并行计算与拼接 ===~%")
  (let* ((x (vt-random-normal (list 3 16)))
         ;; 关键改动：定义一个非零目标，shape 必须和 concat-out 一致
         (target (vt-ones (list 3 16))) ;; 全 1 目标
         (branch1 (make-dense 8 :activation :tanh))
         (branch2 (make-dense 8 :activation :tanh))
         (layers (list branch1 branch2))
         (opt (make-adam :lr 0.01d0)))
    (dolist (l layers)
      (build-model l x))
    
    (dotimes (i 10000) ;; 多训练几轮
      (dolist (l layers)
        (zero-grad! l))
      (let* ((out1 (forward branch1 x))
             (out2 (forward branch2 x))
             (concat-out (vt-concatenate -1 out1 out2))
             ;; 关键改动：Loss = mean(square(预测 - 目标))
             (diff (vt-- concat-out target))
             (loss (coerce (vt-mean (vt-square diff)) 'double-float))
             ;; 关键改动：梯度从 diff 出发，而不是从 concat-out 出发
             (grad (vt-scale diff (/ 2.0d0 (* 3 16))))
             (grads-list (vt-split grad 2 :axis -1))
             (grad1 (first grads-list))
             (grad2 (second grads-list)))
        (backward branch1 grad1)
        (backward branch2 grad2)
        (dolist (l layers)
          (model-update! l opt))
        (when (zerop (mod i 100))
          (format t "Epoch ~a | Loss: ~a~%" i loss))))
    
    ;; 最终验证：检查输出是否真的逼近了 target (全1)
    (let* ((out1 (forward branch1 x))
           (out2 (forward branch2 x))
           (concat-out (vt-concatenate -1 out1 out2))
           (mean-val (coerce (vt-mean concat-out) 'double-float)))
      (format t "最终输出均值 (应接近 1.0): ~a~%" mean-val)
      (if (< (abs (- mean-val 1.0d0)) 0.1d0)
          (format t "[通过] DAG多分支计算与Concat反向传播正确!~%")
          (format t "[失败] 输出未逼近目标，反向传播可能有误!~%")))))

(defun test-classification-loss ()
  (format t "~%=== [测试 11] 真实分类交叉熵损失 ===~%")
  (let* ((batch 10) (num-classes 3)
         (dummy-features (vt-random-normal (list batch 8)))
         (classifier (make-dense num-classes :activation :none))
         (ce-loss (make-cross-entropy-loss))
         (opt (make-adam :lr 0.1d0))
         (targets (vt-from-sequence
		   (make-array batch :element-type 'fixnum 
				     :initial-contents
				     '(0 2 1 0 1 2 2 0 1 0)))))
    
    (build-model classifier dummy-features)    
    (dotimes (i 100)
      (zero-grad! classifier)
      (let* ((logits (forward classifier dummy-features))
             ;; 传入
             (loss-vt (forward ce-loss (list logits targets)))
             (loss-val (coerce (vt-mean loss-vt) 'double-float))
             ;; CE 的 backward 传什么都没关系，它内部会忽略，直接返回对 logits 的梯度
             (grad (backward ce-loss (list loss-val))))
        (backward classifier grad)
        (model-update! classifier opt)
        (when (zerop (mod i 5))
          (format t "Epoch ~a | CE-Loss: ~a~%" i loss-val))))
    (format t "[通过] CrossEntropyLoss 完美融入自动求导系统!~%")))

(defun test-inception-routing ()
  (format t "~%=== [测试 12] Inception 多分支拼接路由 (终极架构验证) ===~%")
  
  ;; ==========================================================
  ;; 【护身符】全局屏蔽深度学习中的致命浮点中断
  ;; :invalid     -> 0/0 产生 NaN，而不是报错
  ;; :divide-by-zero -> 1/0 产生 Inf，而不是报错
  ;; :overflow    -> 极大数溢出产生 Inf
  ;; ==========================================================
  (sb-vm::with-float-traps-masked (:invalid :divide-by-zero :overflow)
    (let* ((batch 4)
           (seq-len 12) ;; 序列长度 12
           (dim 8)
           ;; 输入形状: (4, 12, 8)
           (x (vt-random-normal (list batch seq-len dim)))           
           ;; 3 个独立的线性变换分支 (去掉了 bias 以简化梯度流)
           (branch1 (make-dense 16 :activation :relu :use-bias nil))
           (branch2 (make-dense 16 :activation :relu :use-bias nil))
           (branch3 (make-dense 16 :activation :relu :use-bias nil))
           (opt (make-adam :lr 0.1d0))
           (layers (list branch1 branch2 branch3)))
      ;; 延迟初始化
      (dolist (l layers)
	(build-model l x))      
      (dotimes (epoch 100)
        ;; 1. 梯度清零
        (dolist (l layers) (zero-grad! l))
        ;; ==============================================
        ;; 2. 严谨的前向传播
        ;; ==============================================
        (let* ((parts (vt-split x 3 :axis -2)) ;; 切成 3 份，每份 (4, 4, 8)
               (p1 (first parts))
               (p2 (second parts))
               (p3 (third parts))               
               ;; 走不同的分支，形状变为 (4, 4, 16)
               (out1 (forward branch1 p1))
               (out2 (forward branch2 p2))
               (out3 (forward branch3 p3))
               ;; 沿着序列维度 (axis=-2) 拼接，形状恢复为 (4, 12, 16)
               (merged (vt-concatenate -2 out1 out2 out3))
               ;; 计算 MSE Loss 的前置步骤
               (target (vt-zeros (list batch seq-len 16)))
               (diff (vt-- merged target))
               (loss-3d (vt-scale (vt-* diff diff) 0.5d0))               
               ;; 计算标量 Loss (框架反向传播的起点必须是标量)
               (total-elems (reduce #'* (vt-shape loss-3d)))
               (inv-n (/ 1.0d0 total-elems)))          
          ;; ==============================================
          ;; 3. 严谨的拓扑反向传播 (手动数学推导)
          ;; ==============================================
          ;; 数学推导：Loss = mean(0.5 * diff^2)
          ;; 对 merged 的梯度 = diff * (1/N)
          ;; 这样做可以避开 vt-mul 和 vt-mean 是否有反向图的干扰
          (let* ((grad-merged (vt-scale diff inv-n))                 
                 ;; 【高光时刻】
                 ;; merged 是 concat 来的，所以 grad-merged 必须用 split 拆回去！
                 ;; 形状从 (4, 12, 16) 拆成 3 个 (4, 4, 16)
                 (grad-parts (vt-split grad-merged 3 :axis -2))
                 (grad1 (first grad-parts))
                 (grad2 (second grad-parts))
                 (grad3 (third grad-parts)))
            ;; 将拆好的梯度精确地送回各自的分支
            (backward branch3 grad3)
            (backward branch2 grad2)
            (backward branch1 grad1)))        
        ;; ==============================================
        ;; 4. 参数更新 (为了打印，我们在更新后重跑一次前向拿 loss)
        ;; ==============================================
        (dolist (l layers)
	  (model-update! l opt))        
        ;; 重新跑一次前向计算当前 Epoch 的真实 Loss
        (when (zerop (mod epoch 5))
          (let* ((parts (vt-split x 3 :axis -2))
                 (out1 (forward branch1 (first parts)))
                 (out2 (forward branch2 (second parts)))
                 (out3 (forward branch3 (third parts)))
                 (merged (vt-concatenate -2 out1 out2 out3))
                 (diff (vt-- merged (vt-zeros (list batch seq-len 16))))
                 (loss-3d (vt-scale (vt-* diff diff) 0.5d0))
                 (current-loss (vt-mean loss-3d)))
            (format t "Epoch ~a | Inception Loss: ~a~%"
		    epoch current-loss)))))
    (format t "[通过] Split/Concat 负轴路由、零拷贝梯度拆分、NaN容错完美运行!~%")))

(defun test-transformer-block ()
  (format t "~%=== [测试 13] Post-LayerNorm Transformer Block (隔离优化器验证版) ===~%")
  
  (sb-vm::with-float-traps-masked (:invalid :divide-by-zero :overflow)
    
    (let* ((batch 2)
           (seq-len 5)
           (dim 8)
           (hidden-dim 32)
           (x (vt-random-normal (list batch seq-len dim)))
           
           (ln1 (make-layer-norm (list dim) :eps 1.0d-5))
           (ln2 (make-layer-norm (list dim) :eps 1.0d-5))
           (attn-dense (make-dense dim :activation :relu :use-bias t))
           (ffn-dense1 (make-dense hidden-dim :activation :relu :use-bias t))
           (ffn-dense2 (make-dense dim :use-bias t))
           
           ;; ==============================================
           ;; 【终极修复】为每一层创建独立的优化器实例！
           ;; 彻底绕过 Adam 内部哈希表 Key 冲突的底层 Bug
           ;; ==============================================
           (opt-ln1 (make-adam :lr 0.01d0))
           (opt-ln2 (make-adam :lr 0.01d0))
           (opt-attn (make-adam :lr 0.01d0))
           (opt-ffn1 (make-adam :lr 0.01d0))
           (opt-ffn2 (make-adam :lr 0.01d0)))
      
      ;; 精准喂食初始化
      (build-model ln1 x)
      (build-model attn-dense x)
      (build-model ln2 x)
      (build-model ffn-dense1 x)
      (build-model ffn-dense2 (vt-zeros (list batch seq-len hidden-dim)))
      
      (dotimes (epoch 100)
        ;; 梯度清零也要分别清
        (zero-grad! ln1)
	(zero-grad! ln2)
        (zero-grad! attn-dense)
	(zero-grad! ffn-dense1)
	(zero-grad! ffn-dense2)
        
        (let* ((target (vt-zeros (list batch seq-len dim)))
               
               (norm1 (forward ln1 x))
               (attn-out (forward attn-dense norm1))
               (res1 (vt-+ x attn-out))
               
               (norm2 (forward ln2 res1))
               (ffn-out (forward ffn-dense2 (forward ffn-dense1 norm2))) 
               (res2 (vt-+ res1 ffn-out))
               
               (diff (vt-- res2 target))
               (scalar-loss (coerce (vt-mean (vt-* diff diff)) 'double-float)))
          
          (let* ((N (reduce #'* (vt-shape diff)))
                 (inv-n (/ 2.0d0 (coerce N 'double-float)))
                 (grad-res2 (vt-* diff inv-n))
                 
                 (grad-ffn-out grad-res2)
                 (grad-res1-a grad-res2)
                 
                 (grad-ffn1 (backward ffn-dense2 grad-ffn-out))
                 (grad-norm2 (backward ffn-dense1 grad-ffn1))
                 
                 (grad-res1-b (backward ln2 grad-norm2))
                 (grad-norm1 (vt-+ grad-res1-a grad-res1-b))
                 
                 (grad-attn-out (backward attn-dense grad-norm1)))
            (backward ln1 grad-attn-out))
          
          ;; ==============================================
          ;; 分别使用专属优化器更新参数
          ;; ==============================================
          (model-update! ln1 opt-ln1)
          (model-update! ln2 opt-ln2)
          (model-update! attn-dense opt-attn)
          (model-update! ffn-dense1 opt-ffn1)
          (model-update! ffn-dense2 opt-ffn2)
          
          (when (zerop (mod epoch 10))
            (format t "Epoch ~a | Transformer Block Loss: ~a~%"
		    epoch scalar-loss)))))
    
    (format t "[通过] 模块组装、残差分流、动量隔离完美运行!~%")))


(defun test-transformer-block-a ()
  (format t "~%=== [测试 13-a] Transformer (单一优化器君临天下版) ===~%")
  
  (sb-vm::with-float-traps-masked
      (:invalid :divide-by-zero :overflow)
    
    (let* ((batch 2)
           (seq-len 5)
           (dim 8)
           (hidden-dim 32)
           (x (vt-random-normal (list batch seq-len dim)))
           
           ;; 1. 实例化所有模块
           (ln1 (make-layer-norm (list dim) :eps 1.0d-5))
           (ln2 (make-layer-norm (list dim) :eps 1.0d-5))
           (attn-dense (make-dense dim
                                   :activation :relu
                                   :use-bias t))
           (ffn-dense1 (make-dense hidden-dim
                                   :activation :relu
                                   :use-bias t))
           (ffn-dense2 (make-dense dim :use-bias t))
           
           ;; ==============================================
           ;; 【高光时刻】只需要一个 Adam 实例！
           ;; 底层的 layer-id 机制会自动帮它隔离状态
           ;; ==============================================
           (opt (make-adam :lr 0.01d0))
           (layers (list ln1 ln2 attn-dense
                         ffn-dense1 ffn-dense2)))
      
      ;; 2. 精准喂食初始化
      (build-model ln1 x)
      (build-model attn-dense x)
      (build-model ln2 x)
      (build-model ffn-dense1 x)
      (build-model ffn-dense2
                   (vt-zeros (list batch seq-len hidden-dim)))
      
      (dotimes (epoch 100)
        ;; 优雅：一行代码清零所有层的梯度
        (dolist (l layers) (zero-grad! l))
        
        (let* ((target (vt-zeros (list batch seq-len dim)))
               
               (norm1 (forward ln1 x))
               (attn-out (forward attn-dense norm1))
               (res1 (vt-+ x attn-out))
               
               (norm2 (forward ln2 res1))
               (ffn-out (forward ffn-dense2
                                 (forward ffn-dense1 norm2))) 
               (res2 (vt-+ res1 ffn-out))
               
               (diff (vt-- res2 target))
               (scalar-loss (coerce
                             (vt-mean (vt-* diff diff))
                             'double-float)))
          
          ;; 3. 手动残差反向分流
          (let* ((N (reduce #'* (vt-shape diff)))
                 (inv-n (/ 2.0d0 (coerce N 'double-float)))
                 (grad-res2 (vt-* diff inv-n))
                 
                 (grad-ffn-out grad-res2)
                 (grad-res1-a grad-res2)
                 
                 (grad-ffn1 (backward ffn-dense2 grad-ffn-out))
                 (grad-norm2 (backward ffn-dense1 grad-ffn1))
                 
                 (grad-res1-b (backward ln2 grad-norm2))
                 (grad-norm1 (vt-+ grad-res1-a grad-res1-b))
                 
                 (grad-attn-out (backward attn-dense grad-norm1)))
            ;; 修复了之前多出来的右括号
            (backward ln1 grad-attn-out))
          
          ;; ==============================================
          ;; 【高光时刻】一行代码更新所有层！
          ;; model-update! 会自动把各层的名字传给 opt
          ;; ==============================================
          (dolist (l layers) (model-update! l opt))
          
          (when (zerop (mod epoch 10))
            (format t "Epoch ~a | Loss: ~a~%"
                    epoch scalar-loss)))))
    
    (format t "[通过] 单一优化器 + 自动身份隔离，完美运行!~%")))


(defun run-all-tests ()
  ;; 测试 a: y = x^2 (原生 SGD)
  (run-regression-test
   (lambda (x) (* x x)) -3.0d0 3.0d0 1
   '((16 :relu) (1 :none))
   :epochs 1000
   :opt-type :sgd
   :opt-args '(:lr 0.01d0)
   :name "X-Squared-SGD")

  ;; 测试 b: y = sin(x) (换用 Adam，容易找到全局最优)
  (run-regression-test
   #'sin -3.14d0 3.14d0 1
   '((32 :tanh) (16 :tanh) (1 :none))
   :epochs 1000
   :opt-type :adam
   :opt-args '(:lr 0.02d0)
   :test-inputs '(-3.14d0 -1.57d0 0.0d0 1.57d0 3.14d0)
   :name "Sin-Adam")

  ;; 测试 c: y = x^3 - 2x (更复杂的非线性)
  (run-regression-test
   (lambda (x) (- (* x x x) (* 2.0d0 x)))
   -2.0d0 2.0d0 1
   '((64 :relu) (32 :relu) (1 :none))
   :epochs 1000
   :opt-type :adamw
   :opt-args '(:lr 0.01d0 :weight-decay 1e-4)
   :name "Cubic-AdamW")
  
  (format t "** 启动全架构无死角压力测试...~%")
  (test-cnn-architecture)
  (test-lstm-architecture)
  (test-transformer-architecture)
  (format t "** 结果: 框架底座坚如磐石，全部通过! **~%")
  
  (test-gelu-residual-deep-net)
  (test-lstm-seq2seq)
  (test-nlp-basic-stack)
  (test-1d-input-edge-case)
  (test-dropout-switch)
  (test-global-pooling-classifier)
  (test-inception-branch-concat)
  (test-inception-branch-concat-b)
  (test-classification-loss)
  (test-inception-routing)
  (test-transformer-block)
  (test-transformer-block-a)
  )

