(in-package :clnn)

(defun test-xor ()
  (format t "~%=== [例1] XOR 最小分类 ===~%")
  (let* ((x (vt-from-array
             (make-array '(4 2) :element-type 'double-float
				:initial-contents '((0.0d0 0.0d0)
                                                    (0.0d0 1.0d0)
                                                    (1.0d0 0.0d0)
                                                    (1.0d0 1.0d0)))))
         (y (vt-reshape (vt-from-array
                         (make-array '(4 1) :element-type 'double-float
                                            :initial-contents '((0.0d0) (1.0d0)
								(1.0d0) (0.0d0))))
                        '(4 1)))
         (m (make-sequential :name "xor"))
         (opt (make-adam :lr 0.01d0))
         (loss-fn (make-bce-loss)))
    (seq-add! m (make-dense 8 :in-dim 2 :activation :tanh))
    (seq-add! m (make-dense 1 :in-dim 8 :activation :sigmoid))
    (build-model m x)
    (dotimes (epoch 2000)
      (let* ((pred (forward m x))
             (loss (compute-loss loss-fn pred y))
             (g (compute-loss-gradient loss-fn pred y)))
        (zero-grad! m)
        (backward m g)
        (model-update! m opt)
        (when (zerop (mod epoch 500))
          (format t "  epoch ~4d: loss = ~,6F~%" epoch (vt-item loss)))))
    (let* ((pred (forward m x))
           (pred-class (vt-map (lambda (v) (if (> v 0.5d0) 1.0d0 0.0d0)) pred))
           (acc (vt-item
                 (vt-mean
                  (vt-map (lambda (a b) (if (< (abs (- a b)) 1e-6) 1.0d0 0.0d0))
                          pred-class y)))))
      (format t "  最终准确率: ~a~%" acc)
      (if (> acc 0.999d0)
          (format t "  [PASS] XOR 收敛到 100%~%")
          (format t "  [FAIL] XOR 未完全收敛~%")))))


(defun make-spiral-data (n-per-class n-classes)
  "确定性生成螺旋数据：(样本数, 2) 输入，(样本数, 1) 整数标签。"
  (let* ((total (* n-per-class n-classes))
         (x (make-array (list total 2) :element-type 'double-float))
         (y (make-array total :element-type 'fixnum)))
    (dotimes (c n-classes)
      (dotimes (i n-per-class)
        (let* ((idx (+ (* c n-per-class) i))
               (r (/ (coerce i 'double-float) n-per-class))
               (theta (* 4.0d0 pi (+ r (/ c n-classes))))
               (noise (* 0.05d0 (- (random 1.0d0) 0.5d0))))
          (setf (aref x idx 0) (+ (* r (cos theta)) noise))
          (setf (aref x idx 1) (+ (* r (sin theta)) noise))
          (setf (aref y idx) c))))
    (values (vt-from-array x)
            (vt-from-array y))))

(defun test-spiral ()
  (format t "~%=== [例2] 三分类螺旋数据 ===~%")
  (multiple-value-bind (x y) (make-spiral-data 100 3)
    (let* ((m (make-sequential :name "spiral"))
           (opt (make-adam :lr 0.005d0))
           (loss-fn (make-ce-loss)))
      (seq-add! m (make-dense 32 :in-dim 2 :activation :relu))
      (seq-add! m (make-dense 32 :in-dim 32 :activation :relu))
      (seq-add! m (make-dense 3 :in-dim 32 :activation :linear)) ; logits
      (build-model m x)
      (dotimes (epoch 3000)
        (let* ((logits (forward m x))
               (loss (compute-loss loss-fn logits y))
               (g (compute-loss-gradient loss-fn logits y)))
          (zero-grad! m)
          (backward m g)
          (model-update! m opt)
          (when (zerop (mod epoch 500))
            (format t "  epoch ~4d: loss = ~,6F~%" epoch (vt-item loss)))))
      ;; 计算准确率
      (let* ((logits (forward m x))
             (pred (vt-argmax logits :axis 1))
             (correct (vt-item
                       (vt-mean
                        (vt-map (lambda (a b) (if (= a b) 1.0d0 0.0d0))
                                pred (vt-map #'identity y))))))
        (format t "  最终训练准确率: ~a~%" correct)
        (if (> correct 0.95d0)
            (format t "  [PASS] 螺旋三分类 > 95%~%")
            (format t "  [FAIL] 螺旋三分类未达标~%"))))))


(defun finite-diff-grad (f x &key (eps 1.0d-5))
  "对 F: R^n -> R（标量输出）在 X 处做中心差分求梯度。

   返回与 X 同形状的张量。
   要求 X 可以 copy 出连续副本（vt-copy 生成的即是）；对副本按
   线性索引扰动，原张量不受影响。"
  (let* ((shape (vt-shape x))
         (n (reduce #'* shape))
         (x-data (vt-data x))
         (x-off (vt-offset x))
         (grad-data (make-array n :element-type 'double-float
                                  :initial-element 0.0d0)))
    (dotimes (i n)
      (let* ((xp (vt-copy x))
             (xm (vt-copy x))
             (xp-data (vt-data xp))
             (xm-data (vt-data xm))
             (orig (aref x-data (+ x-off i))))
        ;; vt-copy 生成的副本 offset=0 且连续，直接线性索引即可
        (setf (aref xp-data i) (+ orig eps))
        (setf (aref xm-data i) (- orig eps))
        (let ((fp (vt-item (funcall f xp)))
              (fm (vt-item (funcall f xm))))
          (setf (aref grad-data i) (/ (- fp fm) (* 2.0d0 eps))))))
    (vt-reshape (vt-from-array grad-data) shape)))
(defun max-relative-error (a b)
  "同形状张量 A、B 的逐元素最大相对误差。
   分母用 max(|a|, |b|, 1e-8) 防止除零。"
  (let* ((fa (vt-flatten a))
         (fb (vt-flatten b))
         (a-data (vt-data fa))
         (b-data (vt-data fb))
         (a-off (vt-offset fa))
         (b-off (vt-offset fb))
         (n (reduce #'* (vt-shape fa)))
         (max-err 0.0d0))
    (dotimes (i n max-err)
      (let* ((av (aref a-data (+ a-off i)))
             (bv (aref b-data (+ b-off i)))
             (denom (max 1.0d-8 (max (abs av) (abs bv))))
             (rel (/ (abs (- av bv)) denom)))
        (when (> rel max-err) (setf max-err rel))))))

(defun check-layer-gradient-squared-loss (name layer x0
                                          &key (tol 1.0d-4) (eps 1.0d-5))
  "对 L(x) = sum(forward(layer, x)²) 做梯度检查。
   LAYER 是一个已构造的层对象；X0 是输入张量。
   EPS 是中心差分步长。

   通过比较：
     数值梯度 = ∂L/∂x 的有限差分近似
     解析梯度 = backward(layer, 2 * forward(layer, x0))
   返回 T/NIL。"
  ;; 触发延迟初始化
  (forward layer x0)
  ;; 数值梯度
  (let* ((num-grad
           (finite-diff-grad
            (lambda (x) (vt-item (vt-sum (vt-square (forward layer x)))))
            x0 :eps eps))
         ;; 解析梯度：注意 forward 会覆盖 cache，但没关系，
         ;; 最后一次 (forward layer x0) 会把 cache 修正回 x0 对应的状态
         (y (forward layer x0))
         (dy (vt-scale y 2.0d0))
         (dx (backward layer dy))
         (err (max-relative-error dx num-grad)))
    (format t "  ~a: max rel-err = ~,6E~%" name err)
    (if (< err tol)
        (progn (format t "  [PASS] ~a 梯度正确~%" name) t)
        (progn (format t "  [FAIL] ~a 梯度偏差 ~a~%" name err) nil))))

(defun test-gradient-dense ()
  "Dense 层梯度检查：L = sum(forward(layer, x)²)。"
  (format t "~%=== [例3a] Dense 梯度检查 ===~%")
  (let* ((d (make-dense 3 :in-dim 4 :activation :tanh))
         (x0 (vt-from-array
              (make-array '(1 4) :element-type 'double-float
                                 :initial-contents '((0.5d0 -0.3d0 0.8d0 0.1d0))))))
    (check-layer-gradient-squared-loss "Dense(tanh)" d x0)))

(defun test-gradient-layernorm ()
  "LayerNorm 梯度检查：L = sum(forward(layer, x)²)。"
  (format t "~%=== [例3b] LayerNorm 梯度检查 ===~%")
  (let* ((ln (make-layer-norm '(4) :affine t))
         (x0 (vt-from-array
              (make-array '(1 4) :element-type 'double-float
                                 :initial-contents '((0.5d0 -0.3d0 0.8d0 0.1d0))))))
    (check-layer-gradient-squared-loss "LayerNorm" ln x0)))


(defun test-gradient-dense-1 ()
  "线性层梯度检查：L = sum(forward(l, x)^2)，dL/dx 应与解析公式一致。"
  (format t "~%=== [例3a] Dense 梯度检查 ===~%")
  (let* ((d (make-dense 3 :in-dim 4 :activation :tanh))
         (x0 (vt-from-array
              (make-array '(1 4) :element-type 'double-float
                                 :initial-contents '((0.5d0 -0.3d0 0.8d0 0.1d0))))))
    ;; 先把权重固定
    (forward d x0)
    ;; 数值梯度：对 x 每个分量做中心差分
    (let* ((eps 1e-5)
           (num-grad (make-array 4 :element-type 'double-float)))
      (dotimes (i 4)
        (let ((xp (vt-copy x0)) (xm (vt-copy x0)))
          (setf (row-major-aref (vt-data xp) i)
                (+ (row-major-aref (vt-data x0) i) eps))
          (setf (row-major-aref (vt-data xm) i)
                (- (row-major-aref (vt-data x0) i) eps))
          (let ((fp (vt-item (vt-sum (vt-square (forward d xp)))))
                (fm (vt-item (vt-sum (vt-square (forward d xm))))))
            (setf (aref num-grad i) (/ (- fp fm) (* 2.0d0 eps))))))
      ;; 解析梯度
      (let* ((y (forward d x0))
             (dy (vt-scale y 2.0d0))    ; d(sum(y^2))/dy = 2y
             (dx (backward d dy)))
        (format t "  解析 dx: ~a~%" (vt-to-list dx))
        (format t "  数值 dx: ~a~%" (coerce num-grad 'list))
        (let ((max-rel-err 0.0d0))
          (dotimes (i 4)
            (let* ((a (abs (row-major-aref (vt-data dx) i)))
                   (n (abs (aref num-grad i)))
                   (denom (max 1e-8 (max a n)))
                   (rel (/ (abs (- a n)) denom)))
              (when (> rel max-rel-err) (setf max-rel-err rel))))
          (format t "  最大相对误差: ~a~%" max-rel-err)
          (if (< max-rel-err 1e-4)
              (format t "  [PASS] Dense 梯度正确~%")
              (format t "  [FAIL] Dense 梯度偏差 ~a~%" max-rel-err)))))))

(defun test-gradient-layernorm-1 ()
  "LayerNorm 梯度检查：对输入做有限差分。"
  (format t "~%=== [例3b] LayerNorm 梯度检查 ===~%")
  (let* ((ln (make-layer-norm '(4) :affine t))
         (x0 (vt-from-array
              (make-array '(1 4) :element-type 'double-float
                                 :initial-contents '((0.5d0 -0.3d0 0.8d0 0.1d0)))))
         (eps 1e-5)
         (num-grad (make-array 4 :element-type 'double-float)))
    (forward ln x0)                          ; 初始化 gamma/beta
    (dotimes (i 4)
      (let ((xp (vt-copy x0)) (xm (vt-copy x0)))
        (setf (row-major-aref (vt-data xp) i)
              (+ (row-major-aref (vt-data x0) i) eps))
        (setf (row-major-aref (vt-data xm) i)
              (- (row-major-aref (vt-data x0) i) eps))
        (let ((fp (vt-item (vt-sum (vt-square (forward ln xp)))))
              (fm (vt-item (vt-sum (vt-square (forward ln xm))))))
          (setf (aref num-grad i) (/ (- fp fm) (* 2.0d0 eps))))))
    (let* ((y (forward ln x0))
           (dy (vt-scale y 2.0d0))
           (dx (backward ln dy)))
      (format t "  解析 dx: ~a~%" (vt-to-list dx))
      (format t "  数值 dx: ~a~%" (coerce num-grad 'list))
      (let ((max-rel-err 0.0d0))
        (dotimes (i 4)
          (let* ((a (abs (row-major-aref (vt-data dx) i)))
                 (n (abs (aref num-grad i)))
                 (denom (max 1e-8 (max a n)))
                 (rel (/ (abs (- a n)) denom)))
            (when (> rel max-rel-err) (setf max-rel-err rel))))
        (format t "  最大相对误差: ~a~%" max-rel-err)
        (if (< max-rel-err 1e-4)
            (format t "  [PASS] LayerNorm 梯度正确~%")
            (format t "  [FAIL] LayerNorm 梯度偏差 ~a~%" max-rel-err))))))

(defun make-bar-data (n-samples img-size)
  "每张图 img-size × img-size，横条 vs 竖条。返回 (x, y)。
   横条（label=1）：第 pos 行整行 1
   竖条（label=0）：第 pos 列整列 1"
  (let ((x (make-array (list n-samples 1 img-size img-size)
                       :element-type 'double-float
                       :initial-element 0.0d0))
        (y (make-array n-samples :element-type 'fixnum)))
    (dotimes (i n-samples)
      (let ((horizontal (if (evenp i) 1 0))
            (pos (random img-size)))
        (if (= horizontal 1)
            ;; 横条：只写第 pos 行
            (dotimes (c img-size)
              (setf (aref x i 0 pos c) 1.0d0))
            ;; 竖条：只写第 pos 列
            (dotimes (r img-size)
              (setf (aref x i 0 r pos) 1.0d0)))
        (setf (aref y i) horizontal)))
    (values (vt-from-array x) (vt-from-array y))))


(defun test-cnn-bars ()
  (format t "~%=== [例4] CNN 横竖条分类 ===~%")
  (multiple-value-bind (x y) (make-bar-data 200 8)
    ;; 数据自检
    (let ((s (vt-item (vt-sum (vt-slice x (list 0) (list 0))))))
      (format t "  数据自检：sample 0 sum = ~a（应为 8）~%" s)
      (unless (= s 8.0d0)
        (format t "  [FAIL] 数据坏了，先修 make-bar-data~%")
        (return-from test-cnn-bars nil)))
    (let* ((m (make-sequential :name "cnn-bars"))
           (opt (make-adam :lr 0.01d0))
           (loss-fn (make-ce-loss)))
      (seq-add! m (make-conv2d 8 3
                               :in-channels 1
                               :stride 1 :padding 1 :use-bias t))
      (seq-add! m (make-activation-layer :relu))
      (seq-add! m (make-global-avg-pool2d))   ; (B, 8, 8, 8) → (B, 8)
      (seq-add! m (make-dense 2 :activation :linear))
      (build-model m x)
      (dotimes (epoch 800)
        (let* ((logits (forward m x))
               (loss (compute-loss loss-fn logits y))
               (g (compute-loss-gradient loss-fn logits y)))
          (zero-grad! m)
          (backward m g)
          (model-update! m opt)
          (when (zerop (mod epoch 100))
            (let* ((pred (vt-argmax logits :axis 1))
                   (acc (vt-item
                         (vt-mean
                          (vt-map (lambda (a b)
                                    (if (= (coerce a 'fixnum)
                                           (coerce b 'fixnum))
                                        1.0d0 0.0d0))
                                  pred (vt-map #'identity y))))))
              (format t "  epoch ~3d: loss = ~,6F  acc = ~,4F~%"
                      epoch (vt-item loss) acc)))))
      (let* ((logits (forward m x))
             (pred (vt-argmax logits :axis 1))
             (acc (vt-item
                   (vt-mean
                    (vt-map (lambda (a b)
                              (if (= (coerce a 'fixnum)
                                     (coerce b 'fixnum))
                                  1.0d0 0.0d0))
                            pred (vt-map #'identity y))))))
        (format t "  最终训练准确率: ~a~%" acc)
        (if (> acc 0.99d0)
            (format t "  [PASS] CNN 横竖条分类 100%~%")
            (format t "  [FAIL] CNN 未达标，acc = ~a~%" acc))))))

(defun test-conv-minimal ()
  "极简：一个 conv + flatten + dense，看能否学到 8x8 上 1 个像素的位置分类。"
  (format t "~%=== [测试] 极简 Conv ===~%")
  (let* ((x (make-array '(200 1 8 8) :element-type 'double-float
                                     :initial-element 0.0d0))
         (y (make-array 200 :element-type 'fixnum))
         (m (make-sequential))
         (opt (make-adam :lr 0.01d0))
         (loss-fn (make-ce-loss)))
    ;; 生成 4 分类数据：1 像素的位置在 (0,0)/(0,7)/(7,0)/(7,7)
    (dotimes (i 200)
      (let ((cls (mod i 4)))
        (let ((r (case cls (0 0) (1 0) (2 7) (t 7)))
              (c (case cls (0 0) (1 7) (2 0) (t 7))))
          (setf (aref x i 0 r c) 1.0d0))
        (setf (aref y i) cls)))
    (seq-add! m (make-conv2d 4 3 :in-channels 1
				 :stride 1 :padding 1 :use-bias t))
    (seq-add! m (make-activation-layer :relu))
    (seq-add! m (make-flatten :start-dim 1))
    (seq-add! m (make-dense 4 :activation :linear))
    (build-model m (vt-from-array x))
    (dotimes (epoch 1000)
      (let* ((xv (vt-from-array x))
             (yv (vt-from-array y))
             (logits (forward m xv))
             (loss (compute-loss loss-fn logits yv))
             (g (compute-loss-gradient loss-fn logits yv)))
        (zero-grad! m)
        (backward m g)
        (model-update! m opt)
        (when (zerop (mod epoch 200))
          (format t "  epoch ~3d: loss = ~,6F~%" epoch (vt-item loss)))))
    (let* ((xv (vt-from-array x))
           (logits (forward m xv))
           (pred (vt-argmax logits :axis 1))
           (acc (vt-item (vt-mean
                          (vt-map (lambda (a b) (if (= a b) 1.0d0 0.0d0))
                                  pred (vt-from-array y))))))
      (format t "  准确率: ~a~%" acc)
      (if (> acc 0.95d0)
          (format t "  [PASS] conv2d 基本可用~%")
          (format t "  [FAIL] conv2d 有问题~%")))))


(defun test-conv-maxpool-minimal ()
  "4 分类：1 个像素的位置在 (0,0)/(0,7)/(7,0)/(7,7)。
   结构：conv -> relu -> maxpool -> flatten -> dense
   和 test-conv-minimal 唯一区别是多了一个 max-pool2d。
   如果这个学不到，max-pool2d 就有 bug。"
  (format t "~%=== [测试] Conv + MaxPool 极简 ===~%")
  (let* ((x (make-array '(200 1 8 8) :element-type 'double-float
                                     :initial-element 0.0d0))
         (y (make-array 200 :element-type 'fixnum))
         (m (make-sequential))
         (opt (make-adam :lr 0.01d0))
         (loss-fn (make-ce-loss)))
    (dotimes (i 200)
      (let ((cls (mod i 4)))
        (let ((r (case cls (0 0) (1 0) (2 7) (t 7)))
              (c (case cls (0 0) (1 7) (2 0) (t 7))))
          (setf (aref x i 0 r c) 1.0d0))
        (setf (aref y i) cls)))
    (seq-add! m (make-conv2d 4 3 :in-channels 1
				 :stride 1 :padding 1 :use-bias t))
    (seq-add! m (make-activation-layer :relu))
    (seq-add! m (make-max-pool2d 2))       ; 8x8 -> 4x4
    (seq-add! m (make-flatten :start-dim 1))
    (seq-add! m (make-dense 4 :activation :linear))
    (build-model m (vt-from-array x))
    (dotimes (epoch 1000)
      (let* ((xv (vt-from-array x))
             (yv (vt-from-array y))
             (logits (forward m xv))
             (loss (compute-loss loss-fn logits yv))
             (g (compute-loss-gradient loss-fn logits yv)))
        (zero-grad! m)
        (backward m g)
        (model-update! m opt)
        (when (zerop (mod epoch 200))
          (format t "  epoch ~3d: loss = ~,6F~%" epoch (vt-item loss)))))
    (let* ((xv (vt-from-array x))
           (logits (forward m xv))
           (pred (vt-argmax logits :axis 1))
           (acc (vt-item (vt-mean
                          (vt-map (lambda (a b) (if (= a b) 1.0d0 0.0d0))
                                  pred (vt-from-array y))))))
      (format t "  准确率: ~a~%" acc)
      (cond ((> acc 0.95d0)
             (format t "  [PASS] max-pool2d 可用，问题在结构~%"))
            (t
             (format t "  [FAIL] max-pool2d 有问题~%"))))))

(defun make-sum-data (n-samples seq-len)
  "每个样本：seq_len 个 ±1 随机数，标签是 sum > 0 与否。"
  (let ((x (make-array (list n-samples seq-len 1)
                       :element-type 'double-float))
        (y (make-array n-samples :element-type 'fixnum)))
    (dotimes (i n-samples)
      (let ((s 0.0d0))
        (dotimes (j seq-len)
          (let ((v (if (evenp (random 2)) 1.0d0 -1.0d0)))
            (setf (aref x i j 0) v)
            (incf s v)))
        (setf (aref y i) (if (> s 0.0d0) 1 0))))
    (values (vt-from-array x) (vt-from-array y))))

(defun test-rnn-sequence-sum ()
  (format t "~%=== [例5b] rnn-sequence 累计和分类 ===~%")
  (multiple-value-bind (x y) (make-sum-data 200 8)
    (let* ((seq-len 8)
           (hidden 16)
           (seq-layer (make-rnn-sequence 1 hidden :activation :tanh))
           (head (make-dense 2 :in-dim hidden :activation :linear))
           (opt (make-adam :lr 0.005d0))
           (loss-fn (make-ce-loss)))
      ;; 初始化延迟参数
      (build-model seq-layer x)
      (build-model head (forward seq-layer x))
      (dotimes (epoch 500)
        ;; -------- 前向 --------
        (let* ((seq-out (forward seq-layer x))                    ; (200, 8, 16)
               (last-t  (vt-slice seq-out
                                  (list :all) (list (1- seq-len)) (list :all))) ; (200, 16)
               (logits  (forward head last-t))                    ; (200, 2)
               (g-logits (compute-loss-gradient loss-fn logits y))); (200, 2)

          ;; -------- 反向：head --------
          (zero-grad! head)
          (let ((g-last-t (backward head g-logits)))               ; (200, 16)
            ;; 把 (200,16) 的梯度“放回” (200,8,16) 的最后时间步，其余为 0
            (let ((g-seq (vt-zeros (vt-shape seq-out))))            ; (200, 8, 16)
              (setf (vt-slice g-seq (list :all) (list (1- seq-len)) (list :all))
                    g-last-t)
              ;; -------- 反向：rnn-sequence --------
              (zero-grad! seq-layer)
              (backward seq-layer g-seq)
              (model-update! seq-layer opt)
              (model-update! head opt))))

        (when (zerop (mod epoch 100))
          (let* ((seq-out (forward seq-layer x))
                 (last-t (vt-slice seq-out (list :all) (list (1- seq-len)) (list :all)))
                 (logits (forward head last-t))
                 (loss (compute-loss loss-fn logits y))
                 (pred (vt-argmax logits :axis 1))
                 (acc (vt-item
                       (vt-mean
                        (vt-map (lambda (a b)
                                  (if (= (coerce a 'fixnum) (coerce b 'fixnum))
                                      1.0d0 0.0d0))
                                pred (vt-map #'identity y))))))
            (format t "  epoch ~3d: loss = ~,4F  acc = ~,4F~%"
                    epoch (vt-item loss) acc))))

      ;; -------- 最终评估 --------
      (let* ((seq-out (forward seq-layer x))
             (last-t (vt-slice seq-out (list :all) (list (1- seq-len)) (list :all)))
             (logits (forward head last-t))
             (pred (vt-argmax logits :axis 1))
             (acc (vt-item
                   (vt-mean
                    (vt-map (lambda (a b)
                              (if (= (coerce a 'fixnum) (coerce b 'fixnum))
                                  1.0d0 0.0d0))
                            pred (vt-map #'identity y))))))
        (format t "  最终准确率: ~a~%" acc)
        (if (> acc 0.9d0)
            (format t "  [PASS] rnn-sequence 累计和分类 > 90%~%")
            (format t "  [FAIL] 未达标，acc = ~a~%" acc))))))

(defun test-transformer-converge ()
  (format t "~%=== [例6] Transformer Block 收敛 ===~%")
  (let* ((batch 2) (seq-len 4) (embed-dim 16) (num-heads 4)
         (x (vt-random-normal (list batch seq-len embed-dim)))
         (tb (make-transformer-block embed-dim num-heads
                                     :dropout-rate 0.0d0
                                     :eps 1e-5))
         (opt (make-adam :lr 0.005d0)))
    (build-model tb x)
    (let ((initial-loss nil))
      (dotimes (epoch 100)
        (zero-grad! tb)
        (let* ((out (forward tb x))
               (target x)                        ; 学恒等映射
               (diff (vt-- out target))
               (loss (vt-mean (vt-square diff)))
               (n (* batch seq-len embed-dim))
               (grad (vt-scale diff (/ 2.0d0 n))))
          (when (null initial-loss) (setf initial-loss (vt-item loss)))
          (backward tb grad)
          (model-update! tb opt)
          (when (zerop (mod epoch 20))
            (format t "  epoch ~3d: loss = ~,6F~%" epoch (vt-item loss)))))
      (zero-grad! tb)
      (let ((final-loss (vt-item (vt-mean (vt-square (vt-- (forward tb x) x))))))
        (format t "  初始 loss ~a, 最终 loss ~a~%" initial-loss final-loss)
        (if (< final-loss (* 0.5d0 initial-loss))
            (format t "  [PASS] Transformer Block 收敛~%")
            (format t "  [FAIL] Transformer Block 未收敛~%"))))))

(defun run-bn-experiment (use-bn? lr depth)
  "DEPTH 是中间隐藏层的数量，每层 16 维。"
  (let* ((m (make-sequential))
         (opt (make-adam :lr lr))       ; 用较大的 lr 放大差异
         (loss-fn (make-mse-loss))
         (x (vt-random-normal '(64 8)))
         (y (vt-random-normal '(64 1))))
    (seq-add! m (make-dense 16 :in-dim 8 :activation :relu))
    (when use-bn? (seq-add! m (make-batch-norm 16)))
    (dotimes (i (1- depth))
      (seq-add! m (make-dense 16 :activation :relu))
      (when use-bn? (seq-add! m (make-batch-norm 16))))
    (seq-add! m (make-dense 1 :activation :linear))
    (build-model m x)
    (let ((losses '()))
      (dotimes (epoch 100)
        (let* ((pred (forward m x))
               (g (compute-loss-gradient loss-fn pred y)))
          (zero-grad! m)
          (backward m g)
          (model-update! m opt)
          (when (zerop (mod epoch 20))
            (push (vt-item (compute-loss loss-fn pred y)) losses))))
      (nreverse losses))))

(defun test-batchnorm-accel ()
  (format t "~%=== [例7] BatchNorm 加速收敛（深网络 + 大 lr） ===~%")
  (set-global-training! t)
  (let ((no-bn (run-bn-experiment nil 0.5d0 5))
        (bn    (run-bn-experiment t   0.5d0 5)))
    (format t "  无 BN 的 loss 曲线 (每 20 epoch): ~a~%" no-bn)
    (format t "  有 BN 的 loss 曲线 (每 20 epoch): ~a~%" bn)
    (let ((final-no-bn (car (last no-bn)))
          (final-bn    (car (last bn))))
      (format t "  最终 loss: 无BN=~,4F  有BN=~,4F~%" final-no-bn final-bn)
      (cond ((or (not (= final-no-bn final-no-bn))) ; NaN 检查
             (format t "  [FAIL] 无 BN 训练发散 (NaN)~%"))
            ((< final-bn (* 0.5d0 final-no-bn))
             (format t "  [PASS] BN 显著改善~%"))
            (t
             (format t "  [INFO] 差异不显著，可能任务仍太简单~%"))))))

(defun run-optimizer (opt)
  (let* ((m (make-sequential))
         (x (vt-random-normal '(32 4)))
         (y (vt-random-normal '(32 1)))
         (loss-fn (make-mse-loss)))
    (seq-add! m (make-dense 8 :in-dim 4 :activation :relu))
    (seq-add! m (make-dense 1 :activation :linear))
    (build-model m x)
    (dotimes (epoch 500)
      (let* ((pred (forward m x))
             (g (compute-loss-gradient loss-fn pred y)))
        (zero-grad! m)
        (backward m g)
        (model-update! m opt)))
    (vt-item (compute-loss loss-fn (forward m x) y))))

(defun test-optimizers ()
  (format t "~%=== [例8] 优化器对比 ===~%")
  (dolist (spec (list (cons "SGD"     (lambda () (make-sgd :lr 0.1d0 :momentum 0.9d0)))
                      (cons "Adam"    (lambda () (make-adam :lr 0.01d0)))
                      (cons "AdamW"   (lambda () (make-adamw :lr 0.01d0 :weight-decay 1d-4)))
                      (cons "RMSprop" (lambda () (make-rmsprop :lr 0.01d0)))
                      (cons "Adagrad" (lambda () (make-adagrad :lr 0.1d0)))))
    (let ((loss (run-optimizer (funcall (cdr spec)))))
      (format t "  ~10a: final loss = ~,6F~%" (car spec) loss)))
  (format t "  [PASS] 所有优化器均执行了 500 步更新~%"))

(defun test-optimizers-fair ()
  (format t "~%=== [例8] 优化器对比（公平配置） ===~%")
  (let* ((x (vt-random-normal '(128 8)))
         ;; 有真实信号的任务：y = f(x) + 小噪声
         (w-true (vt-random-normal '(8 1)))
         (y-clean (vt-matmul x w-true))
         (y (vt-+ y-clean (vt-scale (vt-random-normal '(128 1)) 0.1d0)))
         (loss-fn (make-mse-loss))
         (specs (list
                 (list "SGD-0.1"        (lambda () (make-sgd :lr 0.1d0 :momentum 0.9d0)))
                 (list "SGD-0.01"       (lambda () (make-sgd :lr 0.01d0 :momentum 0.9d0)))
                 (list "Adam-0.1"       (lambda () (make-adam :lr 0.1d0)))
                 (list "Adam-0.01"      (lambda () (make-adam :lr 0.01d0)))
                 (list "Adam-0.001"     (lambda () (make-adam :lr 0.001d0)))
                 (list "RMSprop-0.01"   (lambda () (make-rmsprop :lr 0.01d0)))
                 (list "Adagrad-0.1"    (lambda () (make-adagrad :lr 0.1d0))))))
    (dolist (spec specs)
      (let* ((m (make-sequential))
             (opt (funcall (second spec))))
        (seq-add! m (make-dense 32 :in-dim 8 :activation :relu))
        (seq-add! m (make-dense 1 :activation :linear))
        (build-model m x)
        (dotimes (epoch 1000)
          (let* ((pred (forward m x))
                 (g (compute-loss-gradient loss-fn pred y)))
            (zero-grad! m)
            (backward m g)
            (model-update! m opt)))
        (let ((final-loss (vt-item (compute-loss loss-fn (forward m x) y))))
          (format t "  ~14a: final loss = ~,6F~%" (first spec) final-loss))))))


(defun test-schedulers ()
  (format t "~%=== [例9] 学习率调度器 ===~%")
  (flet ((make-and-probe (scheduler-ctor probe-steps)
           (let* ((opt (make-adam :lr 0.1d0))
                  (s (funcall scheduler-ctor opt))
                  (lrs '()))
             (dolist (step probe-steps)
               (loop while (< (scheduler-step-count s) step)
                     do (scheduler-step! s))
               (push (scheduler-get-lr s) lrs))
             (nreverse lrs))))
    (format t "  StepLR (每 10 步 ×0.1): ~a~%"
            (make-and-probe (lambda (o) (make-step-lr o 10 :gamma 0.1d0))
                            '(0 10 20 30)))
    (format t "  ExpLR (每步 ×0.95):    ~a~%"
            (make-and-probe (lambda (o) (make-exponential-lr o :gamma 0.95d0))
                            '(0 1 2 3)))
    (format t "  CosAnnealing (T=100):  ~a~%"
            (make-and-probe (lambda (o) (make-cosine-annealing-lr o 100))
                            '(0 25 50 75 100)))
    (format t "  OneCycle (T=100):      ~a~%"
            (make-and-probe (lambda (o) (make-one-cycle-lr o 0.1d0 100))
                            '(0 30 50 75 100)))
    (format t "  [PASS] 所有调度器均产生 lr 序列~%")))


(defun test-serialization ()
  (format t "~%=== [例10] 序列化往返 ===~%")
  (let* ((m (make-sequential))
         (x (vt-random-normal '(4 8))))
    (seq-add! m (make-dense 16 :in-dim 8 :activation :relu))
    (seq-add! m (make-dense 4 :in-dim 16 :activation :linear))
    (build-model m x)
    (let* ((y1 (forward m x))
           (path "/tmp/clnn-test-model.sexp"))
      (save-model m path)
      (let* ((m2 (load-model path))
             (y2 (forward m2 x))
             (max-diff (vt-item (vt-amax (vt-abs (vt-- y1 y2))))))
        (format t "  原始输出 vs 加载输出 最大差: ~a~%" max-diff)
        (if (< max-diff 1e-12)
            (format t "  [PASS] 序列化往返无损~%")
            (format t "  [FAIL] 序列化有损，最大差 ~a~%" max-diff))))))

(defun run-clnn-test-suite ()
  (format t "~%╔══════════════════════════════════════╗~%")
  (format t "║   clnn 框架分层验证套件                 ║~%")
  (format t "╚══════════════════════════════════════╝~%")
  (set-global-training! t)
  (handler-case (test-xor) (error (e) (format t "  [ERROR] ~a~%" e)))
  (handler-case (test-spiral) (error (e) (format t "  [ERROR] ~a~%" e)))
  (handler-case (test-gradient-dense) (error (e) (format t "  [ERROR] ~a~%" e)))
  (handler-case (test-gradient-layernorm) (error (e) (format t "  [ERROR] ~a~%" e)))
  (handler-case (test-cnn-bars) (error (e) (format t "  [ERROR] ~a~%" e)))
  (handler-case (test-transformer-converge) (error (e) (format t "  [ERROR] ~a~%" e)))
  (handler-case (test-batchnorm-accel) (error (e) (format t "  [ERROR] ~a~%" e)))
  (handler-case (test-optimizers) (error (e) (format t "  [ERROR] ~a~%" e)))
  (handler-case (test-schedulers) (error (e) (format t "  [ERROR] ~a~%" e)))
  (handler-case (test-serialization) (error (e) (format t "  [ERROR] ~a~%" e)))
  (format t "~%=== 套件运行完毕 ===~%"))

#|
NN> (run-clnn-test-suite)


╔══════════════════════════════════════╗
║   clnn 框架分层验证套件              ║
╚══════════════════════════════════════╝

=== [例1] XOR 最小分类 ===
  epoch    0: loss = 0.751829
  epoch  500: loss = 0.005875
  epoch 1000: loss = 0.001766
  epoch 1500: loss = 0.000862
  最终准确率: 1.0
  [PASS] XOR 收敛到 100%

=== [例2] 三分类螺旋数据 ===
  epoch    0: loss = 1.155560
  epoch  500: loss = 0.093513
  epoch 1000: loss = 0.019581
  epoch 1500: loss = 0.014020
  epoch 2000: loss = 0.011037
  epoch 2500: loss = 0.008052
  最终训练准确率: 0.9966666666666667
  [PASS] 螺旋三分类 > 95%

=== [例3a] Dense 梯度检查 ===
  Dense(tanh): max rel-err = 1.690075e-8
  [PASS] Dense(tanh) 梯度正确

=== [例3b] LayerNorm 梯度检查 ===
  LayerNorm: max rel-err = 7.085662e-7
  [PASS] LayerNorm 梯度正确

=== [例4] CNN 横竖条分类 ===
  数据自检：sample 0 sum = 8.0（应为 8）
  epoch   0: loss = 0.690758  acc = 0.8350
  epoch 100: loss = 0.123301  acc = 1.0000
  epoch 200: loss = 0.026439  acc = 1.0000
  epoch 300: loss = 0.012067  acc = 1.0000
  epoch 400: loss = 0.007005  acc = 1.0000
  epoch 500: loss = 0.004509  acc = 1.0000
  epoch 600: loss = 0.003128  acc = 1.0000
  epoch 700: loss = 0.002249  acc = 1.0000
  最终训练准确率: 1.0
  [PASS] CNN 横竖条分类 100%

=== [例6] Transformer Block 收敛 ===
  epoch   0: loss = 2.333935
  epoch  20: loss = 0.052036
  epoch  40: loss = 0.005124
  epoch  60: loss = 0.000780
  epoch  80: loss = 0.000086
  初始 loss 2.3339352537281792, 最终 loss 1.1745683938678534e-5
  [PASS] Transformer Block 收敛

=== [例7] BatchNorm 加速收敛（深网络 + 大 lr） ===
  无 BN 的 loss 曲线 (每 20 epoch): (1.2515957991706823 4.72524931287598
                                5.062305452404406 3.9157856111450586
                                2.7762432248990123)
  有 BN 的 loss 曲线 (每 20 epoch): (0.9420416745288402 1.2304088642823217
                                0.5499057505569237 0.16228827970669654
                                0.02759399801304805)
  最终 loss: 无BN=2.7762  有BN=0.0276
  [PASS] BN 显著改善

=== [例8] 优化器对比 ===
  SGD       : final loss = 0.822575
  Adam      : final loss = 0.144297
  AdamW     : final loss = 0.176637
  RMSprop   : final loss = 0.117280
  Adagrad   : final loss = 0.110690
  [PASS] 所有优化器均执行了 500 步更新

=== [例9] 学习率调度器 ===
  StepLR (每 10 步 ×0.1): (0.1 0.010000000000000002 0.0010000000000000002
                         1.0000000000000003e-4)
  ExpLR (每步 ×0.95):    (0.1 0.095 0.09025 0.0857375)
  CosAnnealing (T=100):  (0.1 0.08535533905932738 0.05 0.014644660940672627 0.0)
  OneCycle (T=100):      (0.004 0.1 0.0811763726439274 0.028312982462817687
                          1.0e-5)
  [PASS] 所有调度器均产生 lr 序列

=== [例10] 序列化往返 ===
  原始输出 vs 加载输出 最大差: 0.0
  [PASS] 序列化往返无损

=== 套件运行完毕 ===
|#
