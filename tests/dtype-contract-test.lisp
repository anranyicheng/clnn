;;;; dtype-contract-test.lisp
;;;;
;;;; dtype 契约测试（对应《框架设计契约与接口契约文档》§2 与 §6）。
;;;;
;;;; 背景：clvt 于 v0.4.x 引入第九种逻辑 dtype `:bool`（CONVENTIONS §17）
;;;; 与 `:complex128/:complex64`（§16）。这两族类型改变了 clnn 依赖的
;;;; 若干底层语义（比较结果不再是 :int8、VT-MAP 的写回会真值化、
;;;; VT-ITEM 对布尔返回 T/NIL、布尔索引被当成 0/1 下标……）。
;;;; 本文件把「clnn 侧承诺的类型行为」钉成可执行断言：
;;;;
;;;;   A 组：clvt 新语义基线 —— 契约的前提，变了就要重审本框架
;;;;   B 组：掩码边界（:bool 作为条件，不得作为算术因子）
;;;;   C 组：dropout / sdpa 掩码 dtype 不被输入传染
;;;;   D 组：优化器状态继承参数 dtype（float32 不再静默升级）
;;;;   E 组：索引 / 标签必须是整数（布尔索引不再静默取错行）
;;;;   F 组：零张量继承 dtype
;;;;   G 组：布尔 / 复数张量的序列化 roundtrip
;;;;   H 组：复数拒绝边界（含一处**已记录的缺口**，见 §2.6）
;;;;   I 组：单一计算域（v2.2 的三域模型：计算域 f64 唯一 / 索引域 / 掩码域）
;;;;
;;;; 用法（需先加载系统）:
;;;;   sbcl --dynamic-space-size 4096 --non-interactive \
;;;;        --eval '(require :asdf)' --eval '(asdf:load-system :clnn)' \
;;;;        --load tests/dtype-contract-test.lisp
;;;;
;;;; 加载即执行；任何断言失败都会抛错使进程以非 0 退出。

(in-package :clnn)

(defvar *pass* 0)
(defvar *fail* 0)

(defmacro check (form &optional (label nil label-p))
  `(handler-case
       (let ((ok ,form))
         (if ok
             (progn (incf *pass*)
                    (format t "  [PASS] ~a~%" ,(if label-p label `',form)))
             (progn (incf *fail*)
                    (format t "  [FAIL] ~a（返回 NIL）~%"
                            ,(if label-p label `',form)))))
     (error (e)
       (incf *fail*)
       (format t "  [FAIL] ~a（意外错误：~a）~%"
               ,(if label-p label `',form) e))))

(defmacro check-signals (form &optional (label nil label-p))
  "FORM 必须抛出错误（用于「确定性拒绝」类契约）。"
  `(handler-case (progn ,form
                        (incf *fail*)
                        (format t "  [FAIL] ~a（本应报错，却正常返回）~%"
                                ,(if label-p label `',form)))
     (error () (incf *pass*)
               (format t "  [PASS] ~a（按契约报错）~%"
                       ,(if label-p label `',form)))))

(defun head (n seq) (subseq seq 0 (min n (length seq))))

(format t "~&==== dtype 契约测试 ====~%")

;;; ================================================================
;;; A. clvt 语义基线
;;; ================================================================
(format t "~&-- A. clvt 语义基线 --~%")

(check (eq :bool (vt-dtype (clvt:vt-= (vt-const '(3) 1.0d0) (vt-const '(3) 2.0d0))))
       "A1 (vt-= …) 返回 :bool（不再是 :int8）")

(check (eq :bool (vt-dtype (clvt:vt-> (vt-const '(3) 2.0d0) (vt-const '(3) 1.0d0))))
       "A2 (vt-> …) 返回 :bool")

(let* ((b (vt-from-sequence '(1 1 0 1) :dtype :bool)))
  (check (equal '(1 1 0 1) (vt-to-list (vt-* b b)))
         "A3 bool ⊗ bool 是逻辑与（不是 2）")
  (check (eq :bool (vt-dtype (vt-* b b)))
         "A4 bool ⊗ bool 仍是 :bool")
  (check-signals (vt-- b b) "A5 bool 减法按 numpy 拒绝（契约：掩码不得参与减法）")
  (check (eq :bool (vt-dtype (vt-map (lambda (x) (declare (ignore x)) 2) b)))
         "A6 vt-map 写回 :bool 时真值化，且结果仍是 :bool")
  (check (equal '(1 1 1 1) (vt-to-list (vt-map (lambda (x) (declare (ignore x)) 2) b)))
         "A7 真值化：常数 2 → 1（不是截断成 0）")
  (check (eq t (vt-item (clvt:vt-all (vt-from-sequence '(1 1 1) :dtype :bool))))
         "A8 vt-item 对布尔标量返回 Lisp 布尔 T")
  (check (null (vt-item (clvt:vt-all b)))
         "A8b vt-item 对布尔标量返回 Lisp 布尔 NIL（含 0 时为假）"))

(check (eq :int64 (vt-dtype (vt-map (lambda (x) (declare (ignore x)) 0.5d0)
                                    (vt-from-sequence '(1 1) :dtype :int64))))
       "A9 vt-map 在整数输入上会截断浮点结果（故掩码必须显式 :dtype）")

(check (eq :float64 (vt-dtype (vt-zeros '(2 2))))
       "A10 vt-zeros 缺省 :float64（故内部零张量一律改用 vt-zeros-like）")

;;; ---- A11…：NEP 50 弱标量的「自封闭判据」 ----
;;; clvt CONVENTIONS §16.31 约定标量只贡献「种类」、宽度由张量决定（NEP 50），
;;; §16.31.7 补齐了剩下十一个「交换顺序就换 dtype」的算子。判据不需要 numpy：
;;;   **二元提升必须与操作数顺序无关**（a ⊔ b == b ⊔ a）。
;;; 交换一下顺序就换 dtype，只可能是标量的「弱身份」在某个位置丢了。
;;; clnn 的缺口 G-11 曾经完全依赖这条不对称（`(vt-- 1.0d0 x-f32)` → f64），
;;; 上游修复后这里把它钉住：一旦回退，G-11 的整套 f32 结论立刻失效。
(format t "~&-- A11 clvt 弱标量对称性（G-11 的上游基线） --~%")
(let ((x (vt-astype (vt-random-normal '(3)) :float32))
      (y (vt-astype (vt-random-normal '(3)) :float32))
      (s 1.5d0))
  (flet ((sym (label fn)
           (handler-case
               (let ((ab (vt-dtype (funcall fn s x)))
                     (ba (vt-dtype (funcall fn x s))))
                 (check (and (eq :float32 ab) (eq :float32 ba))
                        (format nil "A11[~a] 标量在任一位置都保持 f32（ab=~a ba=~a）"
                                label ab ba)))
             (error (e)
               (incf *fail*)
               (format t "  [FAIL] A11[~a] 抛错：~a~%" label e)))))
    ;; 这六个算子 clnn 从不调用（契约 §6 的调用面之外），因此没有导入 NN 包，
    ;; 用 CLVT 前缀直接调 —— 本组测的是 clvt 的语义基线，不是 clnn 的用法。
    (sym "vt--" #'vt--)
    (sym "vt-/" #'vt-/)
    (sym "vt-pow" #'clvt:vt-pow)
    (sym "vt-atan2" #'clvt:vt-atan2)
    (sym "vt-hypot" #'clvt:vt-hypot)
    (sym "vt-copysign" #'clvt:vt-copysign)
    (sym "vt-logaddexp" #'clvt:vt-logaddexp)
    (sym "vt-nextafter" #'clvt:vt-nextafter))
  ;; 全张量 + 全张量必须仍是 f32（这是交换律的平凡情形，作为对照）
  (check (eq :float32 (vt-dtype (vt-- x y)))
         "A11b 张量⊔张量 = f32（对照：不含标量时本来就没有不对称）")
  ;; vt-where 的分支标量同理（曾把结果拉到 f64）
  (let ((cond (clvt:vt-> x (vt-const '(3) 0.0d0))))
    (check (eq :float32 (vt-dtype (clvt:vt-where cond x s)))
           "A11c vt-where 的标量分支不把结果拉成 f64")))

;;; ================================================================
;;; B. 掩码边界
;;; ================================================================
(format t "~&-- B. 掩码边界（:bool 只作条件） --~%")

(let* ((cond-bool (clvt:vt-> (vt-from-sequence '(1.0d0 -1.0d0 2.0d0 -2.0d0))
                             (vt-const '(4) 0.0d0)))
       (cond-int (vt-astype cond-bool :int8))
       (x (vt-const '(4) 1.0d0))
       (y (vt-const '(4) -7.0d0)))
  (check (equal '(7.0d0 -7.0d0 7.0d0 -7.0d0)
                (vt-to-list (tensor-where cond-bool (vt-const '(4) 7.0d0) y)))
         "B1 tensor-where 接受 :bool 条件")
  (check (equal (vt-to-list (tensor-where cond-bool (vt-const '(4) 7.0d0) y))
                (vt-to-list (tensor-where cond-int (vt-const '(4) 7.0d0) y)))
         "B2 :bool 条件与 :int8 条件结果逐位一致")
  (check (equal '(9.0d0 1.0d0 9.0d0 1.0d0)
                (vt-to-list (tensor-masked-fill x cond-bool 9.0d0)))
         "B3 tensor-masked-fill 接受 :bool 掩码")
  ;; 反面：布尔掩码不能当算术因子用（真⊗真=真，1/(1-p) 的缩放会丢）
  (check (equal '(1 0) (vt-to-list (vt-* (vt-from-sequence '(1 1) :dtype :bool)
                                         (vt-from-sequence '(1 0) :dtype :bool))))
         "B4 布尔⊗布尔=逻辑与（故乘性掩码必须先 vt-astype :float64）")
  (check (equal '(2.0d0 0.0d0)
                (vt-to-list (vt-* (vt-from-sequence '(2.0d0 3.0d0))
                                  (vt-astype (vt-from-sequence '(1 0) :dtype :bool)
                                             :float64))))
         "B5 正确的乘性掩码写法：bool → astype :float64"))

;;; ================================================================
;;; C. dropout / sdpa 掩码 dtype
;;; ================================================================
(format t "~&-- C. dropout 掩码 dtype 不被输入传染 --~%")

;;; 四种输入都指向同一个期望值 :float64 —— 因为 v2.2 起计算域唯一。
;;; 历史：v2.0/v2.1 的写法是 (:float32 :float32)，即「掩码跟随输入 dtype」。
;;; 那在多精度路线下成立，但它同时意味着遮罩矩阵也进了精度传播链；
;;; 单一计算域后统一为 f64，掩码的「语义」仍由 0/1 与 1/(1-p) 承载（见 C5）。
(dolist (case '((:float64 :float64) (:float32 :float64)
                (:int64 :float64) (:bool :float64)))
  (destructuring-bind (in-dtype want) case
    (let* ((x (if (eq in-dtype :bool)
                  (clvt:vt-> (vt-const '(64) 1.0d0) (vt-const '(64) 0.0d0))
                  (vt-astype (vt-const '(64) 1.0d0) in-dtype)))
           (d (make-dropout 0.3d0)))
      (let ((*training-mode* t))
        (forward d x)
        (check (eq want (vt-dtype (dropout-mask-cache d)))
               (format nil "C[~a] dropout 掩码 dtype = ~a" in-dtype want))))))

;; 掩码的保留位必须等于 1/(1-p)，整数掩码下这一条会静默变成 1
(let* ((x (vt-astype (vt-const '(64) 1.0d0) :int64))
       (d (make-dropout 0.3d0)))
  (let ((*training-mode* t))
    (forward d x)
    (let ((m (vt-to-list (dropout-mask-cache d))))
      (check (every (lambda (v) (or (= v 0.0d0) (> v 1.4d0))) m)
             "C5 整数输入下 inverted dropout 的缩放 1/(1-p) 未被截断"))))

;;; ================================================================
;;; D. 优化器状态 dtype
;;; ================================================================
(format t "~&-- D. 优化器状态继承参数 dtype --~%")

(defun %tiny-dense (dtype)
  (make-instance 'dense :out-dim 2 :in-dim 3 :activation :none
                        :weights (vt-astype (vt-const '(3 2) 0.5d0) dtype)
                        :bias (vt-astype (vt-zeros '(2)) dtype)))

;;; v2.2：参数的 dtype 不再是「用户传什么就是什么」—— 构造时就已经是
;;; +COMPUTE-DTYPE+（闸门 + INIT-WEIGHT 的 :AROUND 两重保障）。因此这里
;;; 测的是**不变量**：无论用户怎么请求，参数恒 f64，优化器状态恒与参数一致。
;;; 历史写法是 (dolist (dtype '(:float64 :float32)) … (eq dtype …))，在
;;; 多精度路线下成立；单一计算域后它必须反转，否则等于在测一个不存在的自由度。
(dolist (requested '(:float64 :float32))
  (let* ((d (%tiny-dense requested))
         (m (make-sequential))
         (opt (make-sgd :lr 0.1d0)))
    (seq-add! m d)
    (let ((*training-mode* nil))
      (forward m (vt-astype (vt-const '(1 3) 1.0d0) requested))
      (backward m (vt-astype (vt-const '(1 2) 1.0d0) requested))
      (optimizer-step opt (params m) (grads m))
      (check (eq :float64 (vt-dtype (dense-weights d)))
             (format nil "D[请求 ~a] 参数恒为计算域 :float64" requested))
      (check (eq (vt-dtype (dense-weights d))
                 (vt-dtype (gethash (list d "weights")
                                    (optimizer-state-registry opt))))
             (format nil "D[请求 ~a] 优化器状态 dtype = 参数 dtype" requested)))))

;;; ================================================================
;;; E. 索引 / 标签 dtype
;;; ================================================================
(format t "~&-- E. 索引 / 标签必须是整数 --~%")

(let ((mask (clvt:vt-> (vt-const '(2) 1.0d0) (vt-const '(2) 0.0d0)))
      (tab (vt-from-sequence '(10.0d0 20.0d0 30.0d0))))
  (check-signals (tensor-gather tab mask)
                 "E1 tensor-gather 拒绝 :bool 索引（旧行为：静默取第 1 行）")
  (check-signals (tensor-gather tab (vt-from-sequence '(0.0d0 1.0d0)))
                 "E2 tensor-gather 拒绝浮点索引")
  (check (equal '(10.0d0 30.0d0)
                (vt-to-list (tensor-gather tab (vt-from-sequence '(0 2) :dtype :int64))))
         "E3 整数索引正常")
  (check-signals (tensor-one-hot mask 3) "E4 tensor-one-hot 拒绝 :bool 索引")
  (check-signals (let ((e (make-embedding 10 4)))
                   (forward e mask))
                 "E5 embedding 拒绝 :bool 索引")
  (check-signals (compute-loss (make-ce-loss)
                               (vt-from-sequence '((1.0d0 2.0d0 3.0d0)))
                               (vt-from-sequence '(0.0d0)))
                 "E6 ce-loss 拒绝浮点标签")
  (check-signals (compute-loss (make-ce-loss)
                               (vt-from-sequence '((1.0d0 2.0d0 3.0d0)))
                               mask)
                 "E7 ce-loss 拒绝 :bool 标签")
  (check (= 2.4076059644443806d0
            (vt-item (compute-loss (make-ce-loss)
                                   (vt-from-sequence '((1.0d0 2.0d0 3.0d0)))
                                   (vt-from-sequence '(0) :dtype :int64))))
         "E8 int64 标签正常（与 torch 逐位一致）"))

;;; ================================================================
;;; F. 零张量继承 dtype
;;; ================================================================
(format t "~&-- F. 零张量继承 dtype --~%")

(let ((node (make-stop-gradient)))
  ;; v2.2：backward 的 :AROUND 闸门先把 f32 的上游梯度折成 f64，因此这里
  ;; 拿回的零梯度必然是 f64。它仍然是「与上游同类」的零（VT-ZEROS-LIKE 的
  ;; 语义没变），只是上游本身已被规约 —— 这与 C 组的掩码、D 组的参数是同
  ;; 一条规则：**进入计算链的东西一律 f64**。
  (check (eq :float64 (vt-dtype (backward node (vt-astype (vt-const '(2) 1.0d0)
                                                         :float32))))
         "F1 stop-gradient 的零梯度为计算域（f32 上游已由闸门规约）")
  (check (eq :float64 (vt-dtype (backward node (vt-const '(2) 1.0d0))))
         "F2 stop-gradient 在 float64 下不变"))

;;; ================================================================
;;; G. 布尔 / 复数序列化
;;; ================================================================
(format t "~&-- G. 布尔与复数的序列化 --~%")

(let* ((b (clvt:vt-> (vt-const '(2 2) 1.0d0) (vt-const '(2 2) 0.0d0)))
       (r (nn::%restore-vt (nn::%serialize-vt b))))
  (check (and (eq :bool (vt-dtype r)) (equal (vt-shape b) (vt-shape r)))
         "G1 :bool 张量 save/load roundtrip 保持 dtype 与形状"))

(let* ((c (vt-from-sequence (list #c(1.0d0 2.0d0) #c(-1.0d0 0.5d0)) :dtype :complex128))
       (r (nn::%restore-vt (nn::%serialize-vt c))))
  (check (and (eq :complex128 (vt-dtype r))
              (equal (vt-to-list c) (vt-to-list r)))
         "G2 :complex128 张量 roundtrip 逐值一致"))

;;; ================================================================
;;; H. 复数拒绝边界
;;; ================================================================
(format t "~&-- H. 复数边界 --~%")

(let ((z (vt-from-sequence (list #c(1.0d0 1.0d0)) :dtype :complex128)))
  (check-signals (vt-relu z) "H1 vt-relu 拒绝复数（clvt §16.11）")
  (check-signals (vt-sigmoid z) "H2 vt-sigmoid 拒绝复数"))

;; 已记录的缺口：dense 不对复数输入做守卫，前向会静默产出 complex 输出，
;; 直到 grad-norm 的 (> n clip) 才抛类型错。这里只**记录现状**，
;; 一旦加入守卫，本断言会失败并提醒同步契约文档 §2.6。
(let* ((d (%tiny-dense :complex128))
       (m (make-sequential)))
  (seq-add! m d)
  (let ((*training-mode* nil))
    (check (eq :complex128 (vt-dtype (forward m (vt-from-sequence
                                                 (list #c(1.0d0 0.0d0)
                                                       #c(0.0d0 1.0d0)
                                                       #c(1.0d0 1.0d0))
                                                 :dtype :complex128))))
           "H3 [记录] dense 复数输入当前不设防（缺口 G-4，见 §2.6）")))

;;; ================================================================
;;; I. 单一计算域（三域模型的验收）
;;; ================================================================
;;; v2.2 的架构决定：**clnn 只有一个计算精度**（+COMPUTE-DTYPE+ = :float64），
;;; 浮点数据在进入层的边界处规约一次，内部永不再转换。理由有两条，一条是
;;; 可维护性、一条是实测性能：
;;;
;;;   ① clnn 是纯手写求导（§1.5），没有 autograd 统一管理精度传播。多精度
;;;      意味着每层的状态缓冲、梯度累加器、running 统计量、掩码都要人肉追
;;;      一遍「该跟谁同 dtype」——v2.1 的缺口 G-11 就是这么攒出来的。
;;;   ② 实测（4000×4000，20 次）：同 dtype 的 f64 逐元素加法 852 ms，
;;;      f32 584 ms，**f32 ⊔ f64 要 7165 ms —— 慢 8.4 倍**。因为每个算子
;;;      都要按提升表定结果 dtype、分配 f64 临时数组、再逐元素转换。
;;;
;;; 所以本组不再问「f32 能不能保住」，而是问三件事：
;;;   · f32 进来之后，**是否已折成 f64**（I1–I10）
;;;   · 索引域 / 掩码域**是否没被误折**（I11）
;;;   · 外部导入的 f32 参数**有没有一条正规的归一通道**（I12）
;;;
;;; 历史：v2.0/v2.1 的 I 组问的恰恰是「f32 能否全链保住」，46→70 断言那次
;;; 扩的就是它。那套断言在 v2.2 下全部反转 —— 保留本条注释是为了让读到
;;; 这里的人知道「不是断言写错了，是架构决定变了」。

(format t "~&-- I. 单一计算域 --~%")

(defun %f32 () (vt-astype (vt-random-normal '(2 3)) :float32))

;; I1–I3：入口规约本身
(let* ((x (%f32))
       (d (make-dense 4 :in-dim 3 :activation :relu))
       (out (forward d x)))
  (check (eq :float64 (vt-dtype out))
         "I1 f32 输入 → forward 输出为计算域 f64")
  (check (eq :float32 (vt-dtype x))
         "I2 闸门返回新张量，不动调用方持有的原对象（f32 仍是 f32）")
  (check (eq :float64 (vt-dtype (backward d (vt-astype (vt-random-normal '(2 4))
                                                       :float32))))
         "I3 f32 上游梯度 → dX 为计算域 f64"))

;; I4：dense(:tanh) —— G-11 的原发现场（1 − tanh²）
;; G-11 已不再是缺口，但这条留着：它是最容易复发的写法。
(let* ((d (make-instance 'dense :out-dim 4 :in-dim 3 :activation :tanh))
       (out (forward d (%f32)))
       (dx (backward d (vt-astype (vt-random-normal '(2 4)) :float32))))
  (check (and (eq :float64 (vt-dtype out))
              (eq :float64 (vt-dtype dx))
              (eq :float64 (vt-dtype (dense-dw d)))
              (eq :float64 (vt-dtype (dense-db d))))
         "I4 dense(:tanh) 的 fwd/dX/dw/db 全为 f64（G-11 现场已封闭）"))

;; I5：activation-layer 的 tanh/sigmoid 导数（1 − x 与 x(1 − x)）
(dolist (kind '(:tanh :sigmoid))
  (let* ((a (make-activation-layer kind))
         (out (forward a (%f32)))
         (dx (backward a (vt-astype (vt-random-normal '(2 3)) :float32))))
    (check (and (eq :float64 (vt-dtype out)) (eq :float64 (vt-dtype dx)))
           (format nil "I5 activation(:~a) 的 fwd/dX 均为 f64" kind))))

;; I6：batch-norm 训练态与推理态（running 统计量是 buffer，不在 params 里）
(let ((bn (make-batch-norm 4))
      (x (vt-astype (vt-random-normal '(6 4)) :float32)))
  (check (eq :float64 (vt-dtype (with-training t (forward bn x))))
         "I6a batch-norm 训练态输出 f64")
  (check (eq :float64 (vt-dtype (bn-running-mean bn)))
         "I6b batch-norm 的 running 统计量在计算域")
  (let ((out (with-training nil (forward bn x))))
    (check (eq :float64 (vt-dtype out))
           "I6c batch-norm 推理态输出 f64（buffer 不会引入窄 dtype）")
    (check (eq :float64 (vt-dtype (backward bn
                                            (vt-astype (vt-random-normal '(6 4))
                                                       :float32))))
           "I6d batch-norm 推理态 dX 为 f64")))

;; I7：RNN 四支（状态缓冲 / 梯度累加器）
(let ((c (make-rnn-cell 3 4))
      (x (vt-astype (vt-random-normal '(2 3)) :float32)))
  (reset-rnn-cell-state! c)
  (let ((out (forward c x))
        (dx (backward c (vt-astype (vt-random-normal '(2 4)) :float32))))
    (check (and (eq :float64 (vt-dtype out)) (eq :float64 (vt-dtype dx))
                (eq :float64 (vt-dtype (rnn-dwih c))))
           "I7a rnn-cell 的 fwd/dX/dwih 均为 f64")))

(dolist (mk (list (cons "lstm" #'make-lstm) (cons "gru" #'make-gru)
                  (cons "rnn-sequence" #'make-rnn-sequence)))
  (let* ((l (funcall (cdr mk) 5 4))
         (x (vt-astype (vt-random-normal '(2 3 5)) :float32))
         (out (forward l x))
         (dx (backward l (vt-astype (vt-random-normal '(2 3 4)) :float32))))
    (check (and (eq :float64 (vt-dtype out))
                (eq :float64 (vt-dtype dx))
                (every (lambda (g) (eq :float64 (vt-dtype (cdr g)))) (grads l)))
           (format nil "I7[~a] fwd/dX/全部梯度均 f64" (car mk)))))

;; I8：embedding —— 权重进计算域，但**索引留在索引域**
(let ((e (make-embedding 4 3))
      (idx (vt-from-sequence '(1 2) :dtype :int64)))
  (forward e idx)
  (backward e (vt-astype (vt-random-normal '(2 3)) :float32))
  (check (and (eq :float64 (vt-dtype (emb-weight e)))
              (eq :float64 (vt-dtype (emb-dw e))))
         "I8a embedding 的权重与 dw 都在计算域")
  (check (eq :int64 (vt-dtype idx))
         "I8b 索引仍是 int64（闸门不得把索引域折成 f64）"))

;; I9：交叉熵 —— f32 logits 进，f64 损失出
(let ((logits (vt-astype (vt-random-normal '(4 5)) :float32))
      (labels (vt-from-sequence '(0 1 2 3) :dtype :int64)))
  (let ((l (make-ce-loss)))
    (check (eq :float64 (vt-dtype (compute-loss l logits labels)))
           "I9a ce-loss 在 f32 logits 上返回 f64 标量")
    (check (eq :float64 (vt-dtype (compute-loss-gradient l logits labels)))
           "I9b ce-loss 的 dP 为 f64")))

;; I10：conv2d 与池化 —— v2.1 及以前对 f32 输入**显式报错**，现在由闸门规约
;; （内核仍然只实现 f64，见 §2.6；变的是「谁来转」）
(let* ((c (make-conv2d 3 3 :in-channels 1 :stride 1 :padding 1))
       (x (vt-astype (vt-random-normal '(1 1 5 5)) :float32)))
  (check (eq :float64 (vt-dtype (forward c x)))
         "I10 conv2d 接受 f32 输入并输出 f64（不再报错，闸门负责转换）"))

;; I11：另外两个域不得被误折
(let* ((mask (clvt:vt-> (vt-const '(4) 1.0d0) (vt-const '(4) 0.0d0)))
       (idx  (vt-from-sequence '(0 1) :dtype :int32)))
  (check (eq :bool (vt-dtype (ensure-compute-dtype mask "test")))
         "I11a 掩码域（:bool）原样通过闸门")
  (check (eq :int32 (vt-dtype (ensure-compute-dtype idx "test")))
         "I11b 索引域（:int32）原样通过闸门")
  (check (eq :float64 (vt-dtype (ensure-compute-dtype (%f32) "test")))
         "I11c 计算域窄版本（:float32）被折成 f64"))

;; I12：外部导入的 f32 参数有一条正规归一通道
(let ((d (make-dense 4 :in-dim 3 :activation :none)))
  (forward d (%f32))                       ; 先把参数惰性初始化出来
  (setf (dense-weights d) (vt-astype (dense-weights d) :float32)
        (dense-bias d)    (vt-astype (dense-bias d) :float32))
  (check (eq :float32 (vt-dtype (dense-weights d)))
         "I12a 手工塞入的 f32 参数在归一前仍是 f32（闸门不碰参数槽）")
  (let ((n (normalize-params! d)))
    (check (and (= n 2)
                (eq :float64 (vt-dtype (dense-weights d)))
                (eq :float64 (vt-dtype (dense-bias d))))
           "I12b normalize-params! 把 2 个 f32 参数归一到计算域")))

;; I13：警告只出一次（训练循环里每步传 f32 不该把终端刷死）
(let ((*compute-dtype-warned* nil))
  (let ((d (make-dense 4 :in-dim 3 :activation :none)))
    (let ((count 0))
      (handler-bind ((warning (lambda (w)
                                (incf count)
                                (muffle-warning w))))
        (dotimes (i 5) (forward d (%f32))))
      (check (= count 1)
             (format nil "I13a 连续 5 次 f32 输入只产生 1 条警告（实得 ~a）" count)))))

;; I13b：跨层也只报一次 —— 键里若含层名，20 层模型就是 20 条警告，
;; 那不是信息是噪音（见 *COMPUTE-DTYPE-WARNED* 的 docstring）。
(let ((*compute-dtype-warned* nil))
  (let ((d1 (make-dense 4 :in-dim 3 :activation :none))
        (d2 (make-dense 4 :in-dim 3 :activation :none))
        (ln (make-layer-norm '(4)))
        (x3 (vt-astype (vt-random-normal '(2 3)) :float32))
        (x4 (vt-astype (vt-random-normal '(2 4)) :float32)))
    (let ((count 0))
      (handler-bind ((warning (lambda (w)
                                (incf count)
                                (muffle-warning w))))
        (forward d1 x3)
        (forward d2 x3)
        (forward ln x4))
      (check (= count 1)
             (format nil "I13b 三个不同的层各收一次 f32，仍只产生 1 条警告（实得 ~a）"
                     count)))))

;; I13c：清空警告记录后可以再报一次（给「想知道还有哪些层在收 f32」的人）
(let ((*compute-dtype-warned* nil))
  (let ((d (make-dense 4 :in-dim 3 :activation :none))
        (count 0))
    (handler-bind ((warning (lambda (w)
                              (incf count)
                              (muffle-warning w))))
      (forward d (%f32))
      (forward d (%f32))
      (setf *compute-dtype-warned* nil)
      (forward d (%f32)))
    (check (= count 2)
           (format nil "I13c 清空 *compute-dtype-warned* 后可再报一次（实得 ~a）" count))))

(format t "~&==== 汇总: ~a PASS / ~a FAIL ====~%" *pass* *fail*)
(when (plusp *fail*)
  (error "dtype 契约测试失败 ~a 项" *fail*))
