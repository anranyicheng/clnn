;;;; test-architecture-fixes.lisp
;;;;
;;;; 针对 2026-10 架构审查中定位并修复的缺陷的回归测试。
;;;;
;;;; 每个测试对应报告中的一条缺陷，先给出「可观测的错误行为」，
;;;; 再断言修复后的正确行为。全部测试都是确定性的（不依赖随机种子）。
;;;;
;;;; 用法:
;;;;   sbcl --dynamic-space-size 4096 --load test-architecture-fixes.lisp

(require :asdf)
(asdf:load-system :clnn)

(in-package :nn)

(defparameter *pass* 0)
(defparameter *fail* 0)
(defparameter *fail-msgs* '())

(defmacro ck (name cond)
  `(handler-case
       (if ,cond
           (progn (incf *pass*) (format t "~&  [PASS] ~a~%" ,name))
           (progn (incf *fail*)
                  (push ,name *fail-msgs*)
                  (format t "~&  [FAIL] ~a~%" ,name)))
     (error (e)
       (incf *fail*)
       (push (format nil "~a (异常: ~a)" ,name e) *fail-msgs*)
       (format t "~&  [FAIL] ~a (异常: ~a)~%" ,name e))))

(defmacro ck-error (name form)
  "断言 FORM 抛出错误（用于校验防御性检查）。"
  `(ck ,name (handler-case (progn ,form nil) (error () t))))

(defun ms (thunk)
  (let ((t0 (get-internal-real-time)))
    (funcall thunk)
    (/ (* 1000.0d0 (- (get-internal-real-time) t0))
       internal-time-units-per-second)))

(defun allclose (a b &key (tol 1.0d-8))
  (let ((x (vt-to-list (vt-flatten a)))
        (y (vt-to-list (vt-flatten b))))
    (and (= (length x) (length y))
         (every (lambda (p q)
                  (<= (abs (- p q)) (* tol (max 1.0d0 (abs p) (abs q)))))
                x y))))

(defun row-norm (w i d)
  (let ((sq 0.0d0))
    (dotimes (j d)
      (let ((v (coerce (vt-ref w i j) 'double-float)))
        (incf sq (* v v))))
    (sqrt sq)))

;;; ==========================================================
(format t "~%=== 1. 安全：load-model 不得执行读宏求值 ===~%")
;;; ==========================================================

(let ((path "/tmp/clnn-evil-model.sexp")
      (marker "/tmp/clnn-pwned"))
  (ignore-errors (delete-file marker))
  (with-open-file (o path :direction :output :if-exists :supersede)
    (write-line
     (concatenate 'string
                  "(:type sequential :name \"x\" :trainable t :layers "
                  "#.(progn (with-open-file (o \"" marker "\" :direction :output "
                  ":if-exists :supersede) (write-line \"pwned\" o)) nil))")
     o))
  (let ((threw nil))
    (handler-case (load-model path)
      (error () (setf threw t)))
    (ck "load-model 遇到 #. 读宏时拒绝加载（抛错）" threw)
    (ck "load-model 未执行模型文件中的代码（无副作用文件）"
        (not (probe-file marker))))
  (ignore-errors (delete-file path))
  (ignore-errors (delete-file marker)))

;;; ==========================================================
(format t "~%=== 2. ce-loss：one-hot 不得使用 C×C 临时矩阵 ===~%")
;;; ==========================================================

;;; 旧实现用 (vt-eye C) + vt-take 构造 one-hot：C=20000 时需要
;;; 20000² × 8B = 3.2 GB 临时内存，直接把进程打死。
(let* ((c 20000)
       (batch 8)
       (lbl (make-array batch :element-type '(signed-byte 64)))
       (logits (vt-zeros (list batch c))))
  (dotimes (i batch) (setf (aref lbl i) (* i 7)))
  (let* ((lv (vt-from-array lbl :dtype :int64))
         (loss (make-ce-loss))
         (elapsed (ms (lambda () (compute-loss loss logits lv)))))
    (ck (format nil "ce-loss 在 C=~a 下不分配 C×C 矩阵（单次 < 200ms，实测 ~,2fms）"
                c elapsed)
        (< elapsed 200.0d0))
    (let ((g (compute-loss-gradient loss logits lv)))
      (ck "ce-loss 在 C=20000 下梯度形状正确"
          (equal (vt-shape g) (list batch c))))))

;;; 数值正确性：与手工构造的 one-hot 目标对比
(let* ((batch 4) (c 6)
       (logits (vt-random (list batch c)))
       (targets (list 0 3 1 5))
       (idx (make-array batch :element-type '(signed-byte 64))))
  (loop for l in targets for i from 0 do (setf (aref idx i) l))
  (let* ((lv (vt-from-array idx :dtype :int64))
         (loss (make-ce-loss))
         (ce (compute-loss loss logits lv))
         ;; 参考实现：log-softmax 后取 -log p_y（逐行手工计算）
         (ref 0.0d0)
         (rows (vt-to-list logits)))
    (dotimes (i batch)
      (let* ((row (nth i rows))
             (mx (apply #'max row))
             (se (loop for v in row sum (exp (- v mx))))
             (lse (+ mx (log se)))
             (y (nth i targets)))
        (decf ref (- (nth y row) lse))))
    (setf ref (/ ref (coerce batch 'double-float)))
    (ck "ce-loss 数值与参考实现一致"
        (< (abs (- (vt-item ce) ref)) 1d-9))
    ;; 梯度每行和必须为 0（softmax - one-hot）
    (let* ((g (compute-loss-gradient loss logits lv))
           (grows (vt-to-list g))
           (ok t))
      (dotimes (i batch)
        (let ((s (reduce #'+ (nth i grows))))
          (when (> (abs s) 1d-12) (setf ok nil))))
      (ck "ce-loss 梯度每行和为 0" ok))))

;;; label smoothing 的行为：
;;; 均匀分布下平滑不改变损失（对称性），但对「过度自信」的预测会把损失抬高，
;;; 这正是 label smoothing 的作用。
(let* ((batch 2) (c 5)
       ;; 让正确类别的 logit 很大 => 模型过度自信
       (logits (vt-const (list batch c) 0.0d0))
       (lv (vt-from-array (let ((a (make-array batch :element-type '(signed-byte 64))))
                            (setf (aref a 0) 0 (aref a 1) 1) a)
                          :dtype :int64)))
  (setf (vt-ref logits 0 0) 10.0d0)
  (setf (vt-ref logits 1 1) 10.0d0)
  (let ((plain (compute-loss (make-ce-loss) logits lv))
        (smooth (compute-loss (make-ce-loss :label-smoothing 0.2d0) logits lv)))
    (ck "ce-loss label-smoothing 提高过度自信预测的损失"
        (> (vt-item smooth) (vt-item plain)))
    ;; 均匀 logits 下两者应相等（对称性，用于确认平滑项本身没有多余常数偏移）
    (let* ((flat (vt-zeros (list batch c)))
           (p2 (compute-loss (make-ce-loss) flat lv))
           (s2 (compute-loss (make-ce-loss :label-smoothing 0.2d0) flat lv)))
      (ck "ce-loss label-smoothing 在均匀预测下不改变损失"
          (< (abs (- (vt-item p2) (vt-item s2))) 1d-9)))))

(ck-error "ce-loss 对 3D logits 给出明确错误"
          (compute-loss (make-ce-loss)
                        (vt-zeros (list 2 3 5))
                        (vt-from-array (let ((a (make-array 2 :element-type '(signed-byte 64))))
                                         (setf (aref a 0) 0 (aref a 1) 1) a)
                                       :dtype :int64)))
(ck-error "ce-loss 标签越界时报错"
          (compute-loss (make-ce-loss)
                        (vt-zeros (list 2 3))
                        (vt-from-array (let ((a (make-array 2 :element-type '(signed-byte 64))))
                                         (setf (aref a 0) 0 (aref a 1) 99) a)
                                       :dtype :int64)))

;;; ==========================================================
(format t "~%=== 3. embedding max-norm：只处理本次用到的行 ===~%")
;;; ==========================================================

(let* ((emb (make-embedding 50000 64 :max-norm 1.0d0))
       (idx (vt-from-sequence '(1 2 3 4) :dtype :int64)))
  (forward emb idx)                     ; 初始化权重并预热
  (let* ((iters 50)
         (t-big (ms (lambda () (dotimes (k iters) (forward emb idx)))))
         (per (/ t-big (coerce iters 'double-float))))
    (ck (format nil "V=50000/D=64 单次 forward < 20ms（实测 ~,3fms，旧实现约 70ms/次）"
                per)
        (< per 20.0d0))))

;;; 只约束用到的行，未用到的行必须逐位不变
(let* ((emb (make-embedding 30 8 :max-norm 0.02d0))
       (i0 (vt-from-sequence '(0) :dtype :int64))
       (i1 (vt-from-sequence '(5 6) :dtype :int64)))
  (forward emb i0)
  (let* ((w (emb-weight emb))
         (snapshot (vt-to-list (vt-flatten w))))
    (forward emb i1)
    (let* ((after (vt-to-list (vt-flatten w)))
           (row 8)
           (untouched t))
      ;; 行 0..4 与 7..29 不应改变（只有 5、6 可能被缩放）
      (dotimes (i 30)
        (unless (member i '(5 6))
          (dotimes (j row)
            (unless (= (nth (+ (* i row) j) snapshot)
                       (nth (+ (* i row) j) after))
              (setf untouched nil)))))
      (ck "max-norm 不修改本次未使用的行" untouched)
      ;; 被使用的行满足约束
      (ck "max-norm 约束本次使用的行"
          (and (<= (row-norm w 5 row) (+ 0.02d0 1d-9))
               (<= (row-norm w 6 row) (+ 0.02d0 1d-9)))))))

;;; 关闭 max-norm 后必须清掉残留的缩放系数
(let* ((emb (make-embedding 10 4 :max-norm 0.01d0))
       (idx (vt-from-sequence '(1 2) :dtype :int64)))
  (forward emb idx)
  (ck "启用 max-norm 时 norm-scale 已建立" (emb-norm-scale emb))
  (setf (emb-max-norm emb) nil)
  (forward emb idx)
  (ck "关闭 max-norm 后 norm-scale 被清空" (null (emb-norm-scale emb))))

;;; ==========================================================
(format t "~%=== 4. copy-network 必须继承 trainable ===~%")
;;; ==========================================================

(flet ((check (label layer input)
         (ignore-errors (forward layer input))
         (let ((copy (copy-network layer)))
           (ck (format nil "copy-network 继承 trainable: ~a" label)
               (eq (layer-trainable-p copy) (layer-trainable-p layer))))))
  (let ((x2 (vt-zeros (list 2 4)))
        (x4 (vt-zeros (list 1 2 6 6))))
    (let ((m (make-sequential)))
      (seq-add! m (make-dense 4 :in-dim 4))
      (setf (layer-trainable-p m) nil)
      (check "sequential(冻结)" m x2))
    (let ((a (make-activation-layer :relu)))
      (setf (layer-trainable-p a) t)
      (check "activation-layer" a x2))
    (let ((f (make-flatten)))
      (setf (layer-trainable-p f) t)
      (check "flatten" f x4))
    (let ((d (make-dropout 0.3d0)))
      (setf (layer-trainable-p d) t)
      (check "dropout" d x2))
    (let ((r (make-residual (make-dense 4 :in-dim 4))))
      (setf (layer-trainable-p r) nil)
      (check "residual(冻结)" r x2))))

;;; ==========================================================
(format t "~%=== 5. BatchNorm 推理态 forward 后的 backward 必须正确 ===~%")
;;; ==========================================================

;;; 用有限差分校验「推理态前向函数」的梯度。
(let* ((bn (make-batch-norm 3))
       (x (vt-random (list 5 3))))
  ;; 先跑一轮训练，让 running stats 与缓存都有内容
  (with-training t (forward bn x) (backward bn (vt-random (list 5 3))))
  ;; 再按推理态前向，紧接着反向
  (set-model-training! bn nil)
  (let* ((out (forward bn x))
         (r (vt-random (vt-shape out)))
         (dx (backward bn r))
         (data (vt-data x))
         (analytic (vt-to-list (vt-flatten dx)))
         (max-err 0.0d0))
    (dotimes (i 15)
      (let* ((orig (aref data i))
             (h (* 1d-6 (max 1d0 (abs orig))))
             (lp (progn (setf (aref data i) (+ orig h))
                        (vt-item (vt-sum (vt-* (forward bn x) r)))))
             (lm (progn (setf (aref data i) (- orig h))
                        (vt-item (vt-sum (vt-* (forward bn x) r))))))
        (setf (aref data i) orig)
        (let* ((num (/ (- lp lm) (* 2d0 h)))
               (an (nth i analytic))
               (err (abs (- an num))))
          (setf max-err (max max-err err)))))
    (ck (format nil "BN 推理态 dx 与有限差分一致（最大绝对误差 ~,2e）" max-err)
        (< max-err 1d-6))))

;;; ==========================================================
(format t "~%=== 6. build-model 必须初始化 rnn-sequence 的 cell ===~%")
;;; ==========================================================

(let* ((m (make-sequential))
       (rs (make-rnn-sequence 6 4)))
  (seq-add! m rs)
  (ck "build-model 之前 rnn-sequence 参数量为 0（延迟初始化）"
      (zerop (param-count m)))
  (build-model m)
  (ck "build-model 后 rnn-sequence 参数量 = 4*6+4*4+4 = 44"
      (= (param-count m) 44)))

;;; ==========================================================
(format t "~%=== 7. 优化器必须校验 params/grads 对齐 ===~%")
;;; ==========================================================

(let* ((opt (make-sgd :lr 0.1d0))
       (owner (make-dense 2 :in-dim 2))
       (p1 (vt-ones (list 2)))
       (p2 (vt-ones (list 2)))
       (params (list (list owner "a" p1 (lambda (v) (setf p1 v)))
                     (list owner "b" p2 (lambda (v) (setf p2 v)))))
       (grads (list (cons "a" (vt-ones (list 2))))))
  (ck-error "梯度项数少于参数项数时报错"
            (optimizer-step opt params grads))
  (ck "报错发生在更新参数之前（p2 未被改动）"
      (equal (vt-to-list p2) '(1.0d0 1.0d0))))

(let* ((opt (make-adam :lr 0.1d0))
       (owner (make-dense 2 :in-dim 2))
       (p1 (vt-ones (list 2)))
       (params (list (list owner "a" p1 (lambda (v) (setf p1 v)))))
       (grads (list (cons "a" (vt-ones (list 2)))
                    (cons "b" (vt-ones (list 2))))))
  (ck-error "梯度项数多于参数项数时报错"
            (optimizer-step opt params grads)))

;;; 正常模型不受影响
(let* ((m (make-sequential))
       (d1 (make-dense 8 :in-dim 4))
       (d2 (make-dense 3 :in-dim 8)))
  (seq-add! m d1)
  (seq-add! m d2)
  (let ((x (vt-random (list 6 4))))
    (forward m x)
    (backward m (vt-random (list 6 3)))
    (ck "正常模型 optimizer-step 不误报"
        (progn (model-update! m (make-sgd :lr 0.01d0)) t))))

;;; ==========================================================
(format t "~%=== 8. tensor-one-hot 支持多维索引 ===~%")
;;; ==========================================================

(let* ((idx2 (vt-from-array
              (let ((a (make-array (list 2 3) :element-type '(signed-byte 64))))
                (dotimes (i 2) (dotimes (j 3) (setf (aref a i j) j)))
                a)
              :dtype :int64))
       (oh (tensor-one-hot idx2 5)))
  (ck "tensor-one-hot 接受 2D 索引并返回 (6 5)"
      (equal (vt-shape oh) '(6 5)))
  (ck "tensor-one-hot 每行恰有一个 1"
      (let ((rows (vt-to-list oh)) (ok t))
        (dotimes (i 6)
          (unless (= (reduce #'+ (nth i rows)) 1.0d0) (setf ok nil)))
        ok))
  (ck "tensor-one-hot 的 1 落在对应索引位置"
      (let ((rows (vt-to-list oh)) (ok t))
        (dotimes (i 6)
          ;; 索引按行主序展开：第 i 行对应 (i mod 3)
          (let ((want (mod i 3)))
            (dotimes (j 5)
              (unless (= (nth j (nth i rows))
                         (if (= j want) 1.0d0 0.0d0))
                (setf ok nil)))))
        ok)))

;;; ==========================================================
(format t "~%=== 9. 协议泛型已在协议层声明（加载顺序无关）===~%")
;;; ==========================================================

(dolist (name '("ZERO-GRAD-CHILDREN" "ZERO-GRAD!" "TRANSIENT-CACHE-SLOTS"
                "CLEAR-FORWARD-CACHE!" "CLEAR-STEP-CACHES!"))
  (let* ((sym (find-symbol name :nn))
         (fn (and sym (fboundp sym) (fdefinition sym))))
    (ck (format nil "~a 是泛型函数" name)
        (typep fn 'generic-function))
    ;; 协议层必须自带默认方法，否则「协议已声明但无兜底」会让任意
    ;; 未实现该泛型的自定义层在 zero-grad!/clear-*-cache! 时崩掉。
    (ck (format nil "~a 对任意对象有可适用方法" name)
        (and (typep fn 'generic-function)
             (not (null (compute-applicable-methods fn (list (make-dense 2)))))))))

;;; ==========================================================
(format t "~%=== 10. flops-estimate 与手工计数一致 ===~%")
;;; ==========================================================

(let ((d (make-dense 5 :in-dim 4)))
  (forward d (vt-zeros (list 2 4)))
  (multiple-value-bind (macs pars) (flops-estimate d)
    (ck "dense(4->5) macs=20 params=25" (and (= macs 20) (= pars 25)))))
(let ((l (make-lstm 4 3)))
  (multiple-value-bind (macs pars) (flops-estimate l (list 2 5 4))
    (ck "lstm(4->3,T=5) macs=420 params=108"
        (and (= macs 420) (= pars 108)))))
(let ((c (make-conv2d 4 3 :in-channels 2 :padding 1)))
  (forward c (vt-zeros (list 1 2 8 8)))
  (multiple-value-bind (macs pars) (flops-estimate c (list 1 2 8 8))
    (ck "conv2d(2->4,k3) macs=4608 params=76"
        (and (= macs 4608) (= pars 76)))))

;;; ==========================================================
(format t "~%========================================~%")
(format t "  结果: ~a PASS / ~a FAIL~%" *pass* *fail*)
(when *fail-msgs*
  (format t "~%  失败项:~%")
  (dolist (m (reverse *fail-msgs*)) (format t "   - ~a~%" m)))
(format t "========================================~%")

(finish-output)
(when (plusp *fail*) (sb-ext:exit :code 1))
