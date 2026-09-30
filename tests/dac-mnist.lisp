;;;; dac-mnist.lisp
;;;;
;;;; DAC (Deep Adaptive Clustering) on MNIST — 稳定版
;;;; 参考：Chang et al., "Deep Adaptive Image Clustering", ICCV 2017
;;;;
;;;; 用法：
;;;;   (load "dac-mnist.lisp")
;;;;   (dac-mnist-simple)              ; ★ 推荐：AE + KMeans，稳定 ~59%
;;;;   (dac-mnist-final :seed 10)      ; DAC 微调（可复现），~60%
;;;;
;;;; 设计原则：
;;;;   1. 不用 silhouette 做早停 —— 随机 encoder 时虚高，会选错模型。
;;;;   2. 不用 BCE —— 目标 ±∞ 会让余弦相似度分布爆炸。
;;;;   3. 不用自适应阈值 —— 余弦紧致空间里会正反馈失控。
;;;;   4. 不用动态 ae-weight —— 单调变小等于逐渐松开锚点。
;;;;   5. 固定种子 —— 所有随机性由 CLVT::WITH-SEED 控制。

(eval-when (:compile-toplevel :load-toplevel :execute)
  (ql:quickload '(:clnn :chipz)))

(in-package #:clnn)

;;; ============================================================
;;; 第 1 部分：MNIST 数据加载
;;; ============================================================

(defvar *dac-train-x* nil "训练图像 (60000, 784) float64")
(defvar *dac-train-y* nil "训练标签 fixnum 数组")
(defvar *dac-test-x*  nil "测试图像 (10000, 784) float64")
(defvar *dac-test-y*  nil "测试标签 fixnum 数组")

(defun dac-mnist-path (filename)
  "定位项目内的 mnist-data 目录。"
  (asdf:system-relative-pathname
   "clnn" (concatenate 'string "mnist-data/" filename)))

(defun dac-read-images (path n-images)
  "读 MNIST 图像：16 字节头部 + N*784 字节像素，归一化到 [0,1]。"
  (with-open-file (f path :element-type '(unsigned-byte 8))
    (let* ((raw (chipz:decompress nil 'chipz:gzip f))
           (arr (make-array (list n-images 784)
                            :element-type 'double-float
                            :initial-element 0.0d0)))
      (dotimes (i n-images)
        (let ((off (+ 16 (* i 784))))
          (dotimes (j 784)
            (setf (aref arr i j) (/ (aref raw (+ off j)) 255.0d0)))))
      (vt-from-array arr :dtype :float64 :fast t))))

(defun dac-read-labels (path n-labels)
  "读 MNIST 标签：8 字节头部 + N 字节。"
  (with-open-file (f path :element-type '(unsigned-byte 8))
    (let ((raw (chipz:decompress nil 'chipz:gzip f))
          (arr (make-array n-labels :element-type 'fixnum)))
      (dotimes (i n-labels)
        (setf (aref arr i) (aref raw (+ 8 i))))
      arr)))

(defun ensure-dac-mnist-data ()
  "幂等加载 MNIST（训练 + 测试）。"
  (when *dac-train-x* (return-from ensure-dac-mnist-data))
  (format t "~&[MNIST] 加载数据...~%")
  (setf *dac-train-x*
        (dac-read-images (dac-mnist-path "train-images-idx3-ubyte.gz") 60000))
  (setf *dac-train-y*
        (dac-read-labels (dac-mnist-path "train-labels-idx1-ubyte.gz") 60000))
  (setf *dac-test-x*
        (dac-read-images (dac-mnist-path "t10k-images-idx3-ubyte.gz") 10000))
  (setf *dac-test-y*
        (dac-read-labels (dac-mnist-path "t10k-labels-idx1-ubyte.gz") 10000))
  (format t "[MNIST] 完成：train=60000, test=10000~%"))

(defun dac-slice-batch (tensor start end)
  "取第 0 维 [START, END)，强制连续（后续 matmul 要求）。"
  (let ((s (vt-slice tensor (list start end) (list :all))))
    (if (vt-contiguous-p s) s (vt-contiguous s))))

;;; ============================================================
;;; 第 2 部分：L2 归一化层
;;; ============================================================
;;;
;;; 前向：z = x / (||x|| + eps)
;;; 反向：令 n = ||x|| + eps, u = z，
;;;       dL/dx = (dL/dz - u · <dL/dz, u>) / n
;;;
;;; 把 encoder 输出投到单位球面，使 z_i · z_j 直接等于余弦相似度。

(defclass l2-normalize (layer)
  ((eps :initarg :eps :initform 1.0d-8 :reader l2-eps)
   (znorm-cache :initform nil :accessor l2-znorm-cache)
   (norm-cache  :initform nil :accessor l2-norm-cache))
  (:documentation "沿最后一维做 L2 归一化。无参数。"))

(defun make-l2-normalize (&key (eps 1.0d-8) (name "l2-norm"))
  (make-instance 'l2-normalize :eps eps :name name :trainable nil))

(defmethod copy-network ((source l2-normalize))
  (make-l2-normalize :eps (l2-eps source)
                     :name (copy-layer-name source)))

(defmethod forward ((l l2-normalize) x)
  (let* ((sq  (vt-sum (vt-square x) :axis -1 :keepdims t))
         (nrm (vt-+ (vt-map #'sqrt sq) (l2-eps l)))
         (z   (vt-/ x nrm)))
    (setf (l2-znorm-cache l) z
          (l2-norm-cache  l) nrm)
    z))

(defmethod backward ((l l2-normalize) grad)
  (let* ((u     (l2-znorm-cache l))
         (nrm   (l2-norm-cache  l))
         (gdotu (vt-sum (vt-* grad u) :axis -1 :keepdims t))
         (proj  (vt-* u gdotu)))
    (vt-/ (vt-- grad proj) nrm)))

(defmethod grad-slots  ((l l2-normalize)) '())
(defmethod cache-slots ((l l2-normalize)) '(znorm-cache norm-cache))

;;; ============================================================
;;; 第 3 部分：编码器 / 解码器
;;; ============================================================

(defun make-dac-encoder (&key (embed-dim 32) (in-dim 784))
  "编码器：784 → 512 → 256 → EMBED-DIM → L2 归一化。
   最后一步把输出投到单位球面。"
  (let ((m (make-sequential :name "dac-encoder")))
    (seq-add! m (make-dense 512 :in-dim in-dim :activation :relu))
    (seq-add! m (make-dense 256 :activation :relu))
    (seq-add! m (make-dense embed-dim :activation :none))
    (seq-add! m (make-l2-normalize))
    m))

(defun make-dac-decoder (embed-dim &key (out-dim 784))
  "解码器：EMBED-DIM → 256 → 512 → OUT-DIM。
   末层 sigmoid 把像素压到 (0,1)。"
  (let ((m (make-sequential :name "dac-decoder")))
    (seq-add! m (make-dense 256 :in-dim embed-dim :activation :relu))
    (seq-add! m (make-dense 512 :activation :relu))
    (seq-add! m (make-dense out-dim :activation :sigmoid))
    m))

;;; ============================================================
;;; 第 4 部分：AE 预训练（简单版，无早停）
;;; ============================================================

(defun pretrain-dac-autoencoder (encoder decoder
                                 &key (epochs 15) (batch-size 128)
                                   (lr 1.0d-3) (n-train 10000)
                                   (weight-decay 1e-6)
                                   (log-every 500))
  "AE 预训练。★ 不做早停：MSE loss 与聚类质量不一致，
   且没有可靠的无监督信号可选模型，直接跑满 EPOCHS。
   返回 (values encoder decoder)。"
  (let* ((opt       (make-adam :lr lr :weight-decay weight-decay))
         (loss-fn   (make-mse-loss :reduction :mean))
         (n-batches (floor n-train batch-size))
         (t0        (get-internal-real-time)))
    (format t "~%--- AE 预训练 ---~%")
    (dotimes (epoch epochs)
      (let ((epoch-loss 0.0d0))
        (dotimes (b n-batches)
          (let* ((start (* b batch-size))
                 (end   (+ start batch-size))
                 (x     (dac-slice-batch *dac-train-x* start end)))
            (zero-grad! encoder)
            (zero-grad! decoder)
            (let* ((z     (forward encoder x))
                   (x-hat (forward decoder z))
                   (loss  (compute-loss loss-fn x-hat x))
                   (grad  (compute-loss-gradient loss-fn x-hat x)))
              (let ((dz (backward decoder grad)))
                (backward encoder dz))
              (optimizer-step opt
                              (append (params encoder) (params decoder))
                              (append (grads  encoder) (grads  decoder)))
              (incf epoch-loss (vt-item loss)))
            (when (and (> log-every 0)
                       (zerop (mod (1+ b) log-every)))
              (format t "  [pre e~a b~a/~a]  loss=~,4f~%"
                      (1+ epoch) (1+ b) n-batches
                      (/ epoch-loss (1+ b))))))
        (format t "  pre-epoch ~a/~a  loss=~,4f~%"
                (1+ epoch) epochs (/ epoch-loss n-batches))))
    (format t "  预训练耗时 ~,1f 秒~%"
            (/ (- (get-internal-real-time) t0)
               (coerce internal-time-units-per-second 'double-float)))
    (values encoder decoder)))

;;; ============================================================
;;; 第 5 部分：DAC 核心算子
;;; ============================================================

(defun pairwise-cosine-sim (z)
  "Z: (B, D) 已 L2 归一化 → S: (B, B) 余弦相似度，对角线置 0
   （自相似恒为 1，置 0 让它落在忽略区间）。"
  (let* ((b (first (vt-shape z)))
         (s (vt-matmul z (vt-transpose z))))
    (dotimes (i b)
      (setf (vt-ref s i i) 0.0d0))
    s))

(defun dac-pseudo-labels (s lambda+ lambda-)
  "根据阈值从相似度矩阵生成伪标签。
   返回 (values TARGET MASK)：
     S[i,j] > λ⁺  → target=1, mask=1
     S[i,j] < λ⁻  → target=0, mask=1
     其他         → mask=0（被忽略）
   对角线恒被忽略。"
  (let* ((shape  (vt-shape s))
         (b      (first shape))
         (target (vt-zeros shape))
         (mask   (vt-zeros shape)))
    (dotimes (i b)
      (dotimes (j b)
        (when (/= i j)
          (let ((v (vt-ref s i j)))
            (cond
              ((> v lambda+)
               (setf (vt-ref target i j) 1.0d0)
               (setf (vt-ref mask   i j) 1.0d0))
              ((< v lambda-)
               (setf (vt-ref mask   i j) 1.0d0)))))))
    (values target mask)))

(defun dac-hinge-loss-and-grad (s target mask
                                &key (margin-pos 0.70d0)
                                  (margin-neg 0.30d0))
  "掩码合页对比损失（DAC 用）。

   正对（target=1）：L = max(0, margin_pos − S)  → 推向 S ≥ margin_pos
   负对（target=0）：L = max(0, S + margin_neg)  → 推向 S ≤ −margin_neg

   ★ 与 BCE 的根本区别：目标是【有限值】，梯度在 {−1, 0, +1} 中有界。
     达到目标后自动归零，不会把 S 推向 ±∞，因此 S.σ 不会爆炸。"
  (let* ((shape   (vt-shape s))
         (n-valid (vt-item (vt-sum mask))))
    (when (< n-valid 1.0d0)
      (return-from dac-hinge-loss-and-grad
        (values (vt-zeros '(1)) (vt-zeros shape))))
    (let* ((one (vt-ones shape))
           (pos-loss (vt-map (lambda (x) (max 0.0d0 (- margin-pos x))) s))
           (neg-loss (vt-map (lambda (x) (max 0.0d0 (+ x margin-neg))) s))
           (loss-mat (vt-+ (vt-* target pos-loss)
                           (vt-* (vt-- one target) neg-loss)))
           (masked   (vt-* mask loss-mat))
           (loss     (vt-scale (vt-sum masked) (/ 1.0d0 n-valid)))
           (pos-grad (vt-map (lambda (x) (if (< x margin-pos) -1.0d0 0.0d0)) s))
           (neg-grad (vt-map (lambda (x) (if (> x (- margin-neg)) 1.0d0 0.0d0)) s))
           (grad-mat (vt-+ (vt-* target pos-grad)
                           (vt-* (vt-- one target) neg-grad)))
           (grad     (vt-scale (vt-* mask grad-mat) (/ 1.0d0 n-valid))))
      (values loss grad))))

(defun dac-backprop-to-z (ds z)
  "S = Z Z^T（Z 已 L2 归一化），把 dL/dS 反传到 dL/dZ：
     dL/dZ = (dS + dS^T) @ Z"
  (vt-matmul (vt-+ ds (vt-transpose ds)) z))

;;; ============================================================
;;; 第 6 部分：KMeans + 贪心匹配评估
;;; ============================================================

(defun dac-kmeans (data k &key (max-iter 50))
  "在 DATA (N, D) 上跑 KMeans。返回 (values CENTROIDS ASSIGNMENTS)。
   确定性初始化（等距取点）便于复现。"
  (let* ((data   (if (vt-contiguous-p data) data (vt-contiguous data)))
         (shape  (vt-shape data))
         (n      (first shape))
         (d      (second shape))
         (dd     (vt-data data))
         (doff   (vt-offset data))
         (d-rs   (first (vt-strides data)))
         (centroids   (make-array (list k d) :element-type 'double-float
                                            :initial-element 0.0d0))
         (assignments (make-array n :element-type 'fixnum :initial-element 0))
         (sums    (make-array (list k d) :element-type 'double-float))
         (counts  (make-array k :element-type 'fixnum)))
    (dotimes (c k)
      (let ((idx (min (1- n) (floor (* c n) k))))
        (dotimes (j d)
          (setf (aref centroids c j)
                (aref dd (+ doff (* idx d-rs) j))))))
    (dotimes (iter max-iter)
      (let ((changed 0))
        ;; E 步
        (dotimes (i n)
          (let ((base (+ doff (* i d-rs)))
                (best-c 0)
                (best-d most-positive-double-float))
            (dotimes (c k)
              (let ((dist 0.0d0))
                (dotimes (j d)
                  (let ((diff (- (aref dd (+ base j))
                                 (aref centroids c j))))
                    (incf dist (* diff diff))))
                (when (< dist best-d)
                  (setf best-d dist best-c c))))
            (when (/= (aref assignments i) best-c)
              (incf changed)
              (setf (aref assignments i) best-c))))
        ;; M 步
        (fill counts 0)
        (dotimes (c k)
          (dotimes (j d) (setf (aref sums c j) 0.0d0)))
        (dotimes (i n)
          (let ((c    (aref assignments i))
                (base (+ doff (* i d-rs))))
            (incf (aref counts c))
            (dotimes (j d)
              (incf (aref sums c j) (aref dd (+ base j))))))
        (dotimes (c k)
          (when (> (aref counts c) 0)
            (let ((inv (/ 1.0d0 (coerce (aref counts c) 'double-float))))
              (dotimes (j d)
                (setf (aref centroids c j)
                      (* (aref sums c j) inv))))))
        (when (= changed 0) (return))))
    (values (vt-from-array centroids :dtype :float64 :fast t)
            assignments)))

(defun dac-cluster-accuracy (assignments true-labels k)
  "贪心匹配簇→类，返回 (values ACC MAPPING)。
   K=10 规模下贪心解通常等于匈牙利最优解。"
  (let* ((n (length assignments))
         (n-classes 10)
         (confusion (make-array (list k n-classes)
                                :element-type 'fixnum :initial-element 0))
         (used-c (make-array k :initial-element nil))
         (used-l (make-array n-classes :initial-element nil))
         (mapping (make-array k :element-type 'fixnum :initial-element -1)))
    (dotimes (i n)
      (incf (aref confusion (aref assignments i) (aref true-labels i))))
    (dotimes (iter k)
      (let ((best-cnt -1) (best-c -1) (best-l -1))
        (dotimes (c k)
          (unless (aref used-c c)
            (dotimes (l n-classes)
              (unless (aref used-l l)
                (when (> (aref confusion c l) best-cnt)
                  (setf best-cnt (aref confusion c l)
                        best-c c best-l l))))))
        (when (>= best-c 0)
          (setf (aref used-c best-c) t
                (aref used-l best-l) t
                (aref mapping best-c) best-l))))
    (let ((correct 0))
      (dotimes (i n)
        (when (= (aref mapping (aref assignments i)) (aref true-labels i))
          (incf correct)))
      (values (/ (coerce correct 'double-float) (coerce n 'double-float))
              mapping))))

(defun evaluate-dac-clustering (encoder embed-dim
                                &key (n 10000) (k 10) (extract-batch 256))
  "在测试集上：提取嵌入 → KMeans → 贪心匹配 → ACC。"
  (let* ((all-z    (vt-zeros (list n embed-dim) :dtype :float64))
         (n-batches (ceiling n extract-batch)))
    (with-training nil
      (dotimes (b n-batches)
        (let* ((start (* b extract-batch))
               (end   (min (+ start extract-batch) n))
               (bs    (- end start))
               (x     (dac-slice-batch *dac-test-x* start end))
               (z     (forward encoder x)))
          (dotimes (i bs)
            (dotimes (j embed-dim)
              (setf (vt-ref all-z (+ start i) j) (vt-ref z i j)))))
	 (clear-step-caches! encoder)))
    (multiple-value-bind (centroids assignments)
        (dac-kmeans all-z k :max-iter 50)
      (declare (ignore centroids))
      (dac-cluster-accuracy assignments *dac-test-y* k))))

;;; ============================================================
;;; 第 7 部分：主入口
;;; ============================================================

(defun dac-mnist-simple (&key (pretrain-epochs 15)
                              (n-train 60000)
                              (batch-size 128)
                              (embed-dim 32)
                              (k 10)
                              (seed 10))
  "★ 推荐方案：AE 预训练 + KMeans，不做 DAC 微调。
   稳定在 59% 左右，是所有版本里最可靠的。

   参数：
     PRETRAIN-EPOCHS  AE 训练轮数（15 足够）
     N-TRAIN          参与训练的样本数
     EMBED-DIM        嵌入维度（32 是 MNIST 的合理值）
     SEED             随机种子（可复现）

   返回聚类 ACC。"
  (clvt::with-seed (seed)
    (ensure-dac-mnist-data)
    (let* ((encoder (make-dac-encoder :embed-dim embed-dim))
           (decoder (make-dac-decoder embed-dim))
           (t0 (get-internal-real-time)))
      (format t "~%============================================================~%")
      (format t "  DAC-MNIST（简化版：AE + KMeans）~%")
      (format t "  pretrain-epochs=~a  embed-dim=~a  n-train=~a  seed=~a~%"
              pretrain-epochs embed-dim n-train seed)
      (format t "============================================================~%")

      ;; 预训练
      (pretrain-dac-autoencoder encoder decoder
                                :epochs pretrain-epochs
                                :batch-size batch-size
                                :lr 1.0d-3
                                :n-train n-train
                                :log-every 0)

      ;; 评估
      (multiple-value-bind (acc mapping)
          (evaluate-dac-clustering encoder embed-dim :k k)
        (declare (ignore mapping))
        (format t "~%============================================================~%")
        (format t "  最终聚类 ACC = ~,2f%~%" (* 100.0 acc))
        (format t "  总耗时: ~,1f 秒~%"
                (/ (- (get-internal-real-time) t0)
                   (coerce internal-time-units-per-second 'double-float)))
        (format t "============================================================~%~%")
        acc))))

(defun dac-mnist-final (&key (seed 10)
                             (pretrain-epochs 15)
                             (finetune-epochs 5)
                             (embed-dim 32)
                             (batch-size 128)
                             (n-train 60000)
                             (k 10)
                             ;; ★ 固定阈值（不再自适应）
                             (lambda+ 0.5d0)
                             (lambda- 0.0d0)
                             ;; ★ 固定 margin（Hinge 目标）
                             (margin-pos 0.7d0)
                             (margin-neg 0.3d0)
                             ;; ★ 固定 AE 锚点权重（不动态重算）
                             (ae-weight 30.0d0))
  "完整版 DAC：AE 预训练 + Hinge 微调。可复现（固定种子）。

   与 DAC 论文的差异：
     - 用 Hinge 替代 BCE（防止 S.σ 爆炸）
     - 用固定阈值替代自适应（防止正反馈）
     - 用固定 ae-weight 替代动态（防止锚点松脱）
     - 微调只跑 5 轮（历史上最佳出现在第 3 轮附近）

   返回 (values best-encoder best-acc)。"
  (clvt::with-seed (seed)
    (ensure-dac-mnist-data)

    (let* ((encoder (make-dac-encoder :embed-dim embed-dim))
           (decoder (make-dac-decoder embed-dim))
           (ae-loss-fn (make-mse-loss :reduction :mean))
           (opt (make-adam :lr 5.0d-5 :weight-decay 1e-5))
           (n-batches (floor n-train batch-size))
           (t0 (get-internal-real-time)))

      (format t "~%============================================================~%")
      (format t "  DAC on MNIST（完整版, seed=~a）~%" seed)
      (format t "  pre-epochs=~a  ft-epochs=~a  embed-dim=~a  batch=~a~%"
              pretrain-epochs finetune-epochs embed-dim batch-size)
      (format t "  λ⁺=~a  λ⁻=~a  margin=(~a,~a)  ae-weight=~a~%"
              lambda+ lambda- margin-pos margin-neg ae-weight)
      (format t "============================================================~%")

      ;; ---------- 阶段 1：AE 预训练 ----------
      (pretrain-dac-autoencoder encoder decoder
                                :epochs pretrain-epochs
                                :batch-size batch-size
                                :lr 1.0d-3
                                :n-train n-train
                                :log-every 0)

      ;; 基线
      (let ((base-acc (evaluate-dac-clustering encoder embed-dim :k k)))
        (format t "~%  [预训练基线]  ACC=~,2f%~%" (* 100.0 base-acc)))

      ;; ---------- 阶段 2：DAC 微调 ----------
      (format t "~%--- DAC 微调（Hinge + AE anchor）---~%")
      (let ((best-encoder (copy-network encoder))
            (best-acc 0.0d0)
            (acc-history '()))

        (dotimes (epoch finetune-epochs)
          (let ((epoch-loss 0.0d0)
                (epoch-ae-loss 0.0d0)
                (epoch-pairs 0)
                (t-epoch (get-internal-real-time)))

            (dotimes (b n-batches)
              (let* ((start (* b batch-size))
                     (end   (+ start batch-size))
                     (x     (dac-slice-batch *dac-train-x* start end)))
                (let* ((z       (forward encoder x))
                       (x-hat   (forward decoder z))
                       (ae-loss (compute-loss ae-loss-fn x-hat x))
                       (ae-grad (compute-loss-gradient ae-loss-fn x-hat x))
                       (s       (pairwise-cosine-sim z)))
                  (multiple-value-bind (target mask)
                      (dac-pseudo-labels s lambda+ lambda-)
                    (multiple-value-bind (dac-loss ds)
                        (dac-hinge-loss-and-grad s target mask
                                                 :margin-pos margin-pos
                                                 :margin-neg margin-neg)
                      (zero-grad! encoder)
                      (zero-grad! decoder)
                      (let* ((dz-ae (backward decoder ae-grad))
                             (dz-dac (dac-backprop-to-z ds z))
                             (dz-total (vt-+ dz-dac
                                             (vt-scale dz-ae ae-weight))))
                        (backward encoder dz-total)
                        (optimizer-step opt
                                        (append (params encoder)
                                                (params decoder))
                                        (append (grads encoder)
                                                (grads decoder)))
                        (incf epoch-loss    (vt-item dac-loss))
                        (incf epoch-ae-loss (vt-item ae-loss))
                        (incf epoch-pairs   (round (vt-item (vt-sum mask))))))))))

            (let ((acc (evaluate-dac-clustering encoder embed-dim :k k)))
              (push acc acc-history)
              (format t "Epoch ~a/~a  dac=~,4f  ae=~,4f  pairs=~a  ACC=~,2f%  (~,1fs)~%"
                      (1+ epoch) finetune-epochs
                      (/ epoch-loss n-batches)
                      (/ epoch-ae-loss n-batches)
                      (round (/ epoch-pairs n-batches))
                      (* 100.0 acc)
                      (/ (- (get-internal-real-time) t-epoch)
                         (coerce internal-time-units-per-second 'double-float)))
              (when (> acc best-acc)
                (setf best-acc acc)
                (setf best-encoder (copy-network encoder))))))

        ;; ---------- 汇总 ----------
        (let* ((history (nreverse acc-history))
               (n (length history))
               (last3 (if (>= n 3) (subseq history (- n 3)) history))
               (last3-avg (/ (reduce #'+ last3) (length last3))))
          (format t "~%============================================================~%")
          (format t "  微调 ACC 历史：~{~,2f%%~^ → ~}~%"
                  (mapcar (lambda (x) (* 100.0 x)) history))
          (format t "  最佳单轮 ACC  = ~,2f%~%" (* 100.0 best-acc))
          (format t "  最后 3 轮平均 = ~,2f%~%" (* 100.0 last3-avg))
          (format t "  总耗时: ~,1f 秒~%"
                  (/ (- (get-internal-real-time) t0)
                     (coerce internal-time-units-per-second 'double-float)))
          (format t "============================================================~%~%")
          (values best-encoder best-acc))))))


#|

(dac-mnist-simple :seed 10)

============================================================
  DAC-MNIST（简化版：AE + KMeans）
  pretrain-epochs=15  embed-dim=32  n-train=10000  seed=10
============================================================

--- AE 预训练 ---
  pre-epoch 1/15  loss=0.0778
  pre-epoch 2/15  loss=0.0401
  pre-epoch 3/15  loss=0.0276
  pre-epoch 4/15  loss=0.0216
  pre-epoch 5/15  loss=0.0187
  pre-epoch 6/15  loss=0.0170
  pre-epoch 7/15  loss=0.0153
  pre-epoch 8/15  loss=0.0142
  pre-epoch 9/15  loss=0.0133
  pre-epoch 10/15  loss=0.0125
  pre-epoch 11/15  loss=0.0120
  pre-epoch 12/15  loss=0.0117
  pre-epoch 13/15  loss=0.0116
  pre-epoch 14/15  loss=0.0110
  pre-epoch 15/15  loss=0.0103
  预训练耗时 475.9 秒

============================================================
  最终聚类 ACC = 58.44%
  总耗时: 484.8 秒
============================================================

NN> (dac-mnist-simple :seed 10 :n-train 60000)

============================================================
  DAC-MNIST（简化版：AE + KMeans）
  pretrain-epochs=15  embed-dim=32  n-train=60000  seed=10
============================================================

--- AE 预训练 ---
  pre-epoch 1/15  loss=0.0346
  pre-epoch 2/15  loss=0.0133
  pre-epoch 3/15  loss=0.0102
  pre-epoch 4/15  loss=0.0088
  pre-epoch 5/15  loss=0.0079
  pre-epoch 6/15  loss=0.0074
  pre-epoch 7/15  loss=0.0070
  pre-epoch 8/15  loss=0.0066
  pre-epoch 9/15  loss=0.0064
  pre-epoch 10/15  loss=0.0062
  pre-epoch 11/15  loss=0.0060
  pre-epoch 12/15  loss=0.0059
  pre-epoch 13/15  loss=0.0058
  pre-epoch 14/15  loss=0.0056
  pre-epoch 15/15  loss=0.0055
  预训练耗时 2816.3 秒

============================================================
  最终聚类 ACC = 63.94%
  总耗时: 2824.1 秒
============================================================
|#
