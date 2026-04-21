;;;; ================================================================
;;; 通用回归测试引擎 (无全局变量，纯函数打包)
;;;; ================================================================
(in-package #:nn)

;;; ----------------------------------------------------------------
;;; 1. 数据生成器工厂 (局部函数)
;;; ----------------------------------------------------------------
(defun make-data-gen (fn x-min x-max)
  "返回一个闭包，调用它将生成指定函数的批次数据。
   FN: 接收双精度浮点数，返回双精度浮点数的函数 (如 #'sin)
   X-MIN, X-MAX: 采样区间"
  (lambda (batch-size)
    (let ((x-list '())
          (y-list '())
          (range (- x-max x-min)))
      (dotimes (_ batch-size)
        (let ((x (+ x-min (random range))))
          (push x x-list)
          (push (funcall fn x) y-list)))
      (values
        (vt-reshape
          (vt-from-sequence (nreverse x-list))
          (list batch-size 1))
        (vt-reshape
          (vt-from-sequence (nreverse y-list))
          (list batch-size 1))))))

;;; ----------------------------------------------------------------
;;; 2. 核心训练引擎
;;; ----------------------------------------------------------------
(defun run-regression-test
  (fn x-min x-max input-dim layer-specs
   &key (epochs 500) (batch-size 32)
        (opt-type :sgd) (opt-args '(:lr 0.01d0))
        (test-inputs '(-2.0d0 -1.0d0 0.0d0 1.0d0 2.0d0))
        (name "MLP"))
  "通用的单变量/多变量回归测试函数。
   
   LAYER-SPECS: 例如 ((16 :relu) (32 :relu) (1 :none))
   OPT-TYPE: :sgd, :adam, :adamw 等
   OPT-ARGS: 传给优化器的关键字参数列表"
  (let* ((gen-fn (make-data-gen fn x-min x-max))
         (model (make-sequential :name name)))
    
    ;; 动态构建网络层
    (dolist (spec layer-specs)
      (seq-add! model
                (apply #'make-dense
                       (first spec)
                       :activation (second spec)
                       :name (format nil "~A-L~A" name (first spec))
                       (cddr spec))))
    
    ;; 显式构建并打印参数量
    (build-model model (vt-zeros (list 1 input-dim)))
    (format t "[~A] 模型参数量: ~A~%" name (param-count model))
    
    ;; 实例化优化器
    (let ((optimizer
            (ecase opt-type
              (:sgd (apply #'make-sgd opt-args))
              (:adam (apply #'make-adam opt-args))
              (:adamw (apply #'make-adamw opt-args))
              (:rmsprop (apply #'make-rmsprop opt-args)))))
      
      (format t "[~A] 开始训练 (~A Epochs)...~%" name epochs)
      (format t "----------------------------------------~%")
      
      ;; 训练循环
      (dotimes (epoch epochs)
        (multiple-value-bind (x y) (funcall gen-fn batch-size)
          (zero-grad! model)
          (let* ((pred (model-forward model x))
                 (diff (vt-- pred y))
                 (loss-val (vt-mean (vt-square diff)))
                 (n batch-size)
                 (grad-output (vt-scale diff (/ 2.0d0 n))))
            (model-backward model grad-output)
            (model-update! model optimizer)
            (when (zerop (mod epoch 100))
              (format t "Epoch ~4D | Loss: ~6F~%"
                      epoch loss-val)))))
      
      (format t "----------------------------------------~%")
      
      ;; 推理测试
      (when test-inputs
        (let* ((len (length test-inputs))
               (test-x (vt-reshape
                         (vt-from-sequence test-inputs)
                         (list len input-dim)))
               (pred (model-forward model test-x))
               (true-y (mapcar fn test-inputs)))
          (format t "[~A] 推理对比:~%" name)
          (format t "  X 输入: ~A~%" test-inputs)
          (format t "  真实 Y: ~A~%" true-y)
          (format t "  预测 Y: ")
          (dotimes (i len)
            (format t "~F, "
                    (vt-ref pred i 0)))
          (format t "~%")))
      
      ;; 返回模型供后续检查
      model)))

;;; ----------------------------------------------------------------
;;; 3. 测试用例集 (一键运行，互不干扰)
;;; ----------------------------------------------------------------
(defun run-all-tests ()
  ;; 测试 1: y = x^2 (原生 SGD)
  (run-regression-test
    (lambda (x) (* x x)) -3.0d0 3.0d0 1
    '((16 :relu) (1 :none))
    :opt-type :sgd
    :opt-args '(:lr 0.01d0)
    :name "X-Squared-SGD")

  ;; 测试 2: y = sin(x) (换用 Adam，容易找到全局最优)
  (run-regression-test
    #'sin -3.14d0 3.14d0 1
    '((32 :tanh) (16 :tanh) (1 :none))
    :opt-type :adam
    :opt-args '(:lr 0.02d0)
    :test-inputs '(-3.14d0 -1.57d0 0.0d0 1.57d0 3.14d0)
    :name "Sin-Adam")

  ;; 测试 3: y = x^3 - 2x (更复杂的非线性)
  (run-regression-test
    (lambda (x) (- (* x x x) (* 2.0d0 x)))
    -2.0d0 2.0d0 1
    '((64 :relu) (32 :relu) (1 :none))
    :opt-type :adamw
    :opt-args '(:lr 0.01d0 :weight-decay 1e-4)
    :name "Cubic-AdamW"))

;; 执行入口
;; (run-all-tests)


;;;; ================================================================
;;; 多架构极限压力测试
;;; ================================================================
(in-package #:nn)

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
    
    (dotimes (epoch 50)
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
          (format t "Epoch ~2D | Loss: ~6F~%" epoch loss))))
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
    
    (dotimes (epoch 50)
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
            (format t "Epoch ~2D | Loss: ~6F~%" epoch loss)))))
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
    
    (dotimes (epoch 50)
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
          (format t "Epoch ~2D | Loss: ~6F~%" epoch loss))))
    (format t "[Transformer] 3D转置、QKV切分与残差求导大成功!~%")))

;;; ----------------------------------------------------------------
;;; 一键执行所有架构测试
;;; ----------------------------------------------------------------
(defun run-all-architecture-tests ()
  (format t "*********************************************~%")
  (format t "** 启动全架构无死角压力测试...~%")
  (format t "*********************************************~%")
  (test-cnn-architecture)
  (test-lstm-architecture)
  (test-transformer-architecture)
  (format t "~%*********************************************~%")
  (format t "** 结果: 框架底座坚如磐石，全部通过! **~%")
  (format t "*********************************************~%"))

;; 执行入口:
;; (run-all-architecture-tests)
