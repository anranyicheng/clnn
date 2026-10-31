;;;; test-runtime-contract.lisp
;;;;
;;;; 运行期契约测试，对应《框架设计契约与接口契约文档》§2.7（损失 target 的域）、
;;;; §4.5（优化步步的边界）、§5.6（模型加载态）。
;;;;
;;;; 这三组契约的共同点是：**违反时不会报错，只会给出一个数值上讲得通、
;;;; 效果上悄悄变差的结果**。dtype 契约（dtype-contract-test）管的是「类型
;;;; 对不对」，这里管的是「动作做对了几次、状态切没切」。
;;;;
;;;;   J 组：损失 target 的域归属 —— 索引域的标签原样透传，计算域的数值目标
;;;;         必须被规约（否则与 f64 的 predicted 混跑）
;;;;   K 组：优化步步号 —— 一次 UPDATE! = 一步，与「OPTIMIZER-STEP 被调了几次」
;;;;         解耦；分片更新用 WITH-OPTIMIZER-STEP 归并
;;;;   L 组：模型加载态 —— LOAD-MODEL 后必须是推理态（dropout 关闭），
;;;;         否则 save → load → predict 这个最常见流程给出随机结果
;;;;
;;;; 用法（需先加载系统）:
;;;;   sbcl --dynamic-space-size 4096 --non-interactive \
;;;;        --eval '(require :asdf)' --eval '(asdf:load-system :clnn)' \
;;;;        --load tests/test-runtime-contract.lisp
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

(defun %rand (shape &optional (dtype :float64))
  (vt-astype (vt-random-normal shape) dtype))

;;; ==================================================================
;;; J 组：损失 target 的域归属
;;; ==================================================================
;;; 闸门按「域」分流而不是按「参数位置」分流：
;;;   · CE 的 target 是整数类别标签 -> 索引域 -> 原样透传
;;;   · mse / bce / huber / cosine / kl 的 target 是数值目标 -> 计算域 -> 规约
;;; 旧实现只规约 predicted、对 target 完全不碰，于是 mse 在 f32 target 下
;;; 每一步都跑一次 f32⊔f64 的提升表（实测损失 4.8×、梯度 6.6×）。

(format t "~&~%==== J 组：损失 target 的域归属 ====~%")

;; J1–J2：数值目标被规约到计算域
(let* ((p (%rand '(8 4) :float32))
       (t32 (%rand '(8 4) :float32))
       (mse (make-mse-loss)))
  (check (eq :float64 (vt-dtype (compute-loss mse p t32)))
         "J1a mse: f32 target 的损失值是计算域")
  (check (eq :float64 (vt-dtype (compute-loss-gradient mse p t32)))
         "J1b mse: f32 target 的梯度是计算域")
  ;; 与 f64 target 逐位一致：规约不能改变数值
  (let ((t64 (vt-astype t32 :float64)))
    (check (zerop (abs (- (vt-item (compute-loss mse p t32))
                          (vt-item (compute-loss mse p t64)))))
           "J2 f32 target 与 f64 target 给出同一个损失值（规约是纯转换）")))

;; J3：CE 的整数标签不得被折成浮点
(let* ((logits (vt-random-normal '(8 5)))
       (labels (vt-from-sequence (list 0 1 2 3 4 0 1 2) :dtype :int64))
       (ce (make-ce-loss)))
  (check (eq :float64 (vt-dtype (compute-loss ce logits labels)))
         "J3a ce: int64 标签可正常计算（未被折成浮点）")
  (check (eq :float64 (vt-dtype (compute-loss-gradient ce logits labels)))
         "J3b ce: 梯度是计算域"))

;; J4：掩码域（:bool）目标不被折成 0.0/1.0
(let ((b (clvt:vt-< (vt-const '(4) 2.0d0) (vt-const '(4) 3.0d0))))
  (check (eq :bool (vt-dtype (ensure-compute-dtype b "test")))
         "J4 :bool target 原样透传（不用浮点冒充布尔）"))

;; J5：索引域（各种整数宽）原样透传
(dolist (dt '(:int8 :int16 :int32 :int64 :uint8 :uint16))
  (let ((idx (vt-astype (vt-zeros '(4) :dtype :int64) dt)))
    (check (eq dt (vt-dtype (ensure-compute-dtype idx "test")))
           (format nil "J5 ~a 索引原样透传" dt))))

;; J6：kl / bce 的数值目标同样被规约
(let ((p (%rand '(4 3) :float32))
      (t32 (%rand '(4 3) :float32)))
  (check (eq :float64 (vt-dtype (compute-loss (make-bce-loss) p t32)))
         "J6a bce: f32 target 的损失值是计算域")
  (check (eq :float64 (vt-dtype (compute-loss (make-kl-divergence-loss) p t32)))
         "J6b kl: f32 target 的损失值是计算域"))

;;; ==================================================================
;;; K 组：优化步步号
;;; ==================================================================
;;; STEP-COUNT 记的是「执行了多少个参数更新步」。旧实现把它等同于
;;; 「OPTIMIZER-STEP 被调了几次」，于是整模型一次 UPDATE! 与逐层 N 次
;;; UPDATE! 在 Adam 偏差校正上差 N 倍。现在步号由 UPDATE! 授予。

(format t "~&~%==== K 组：优化步步号 ====~%")

(defun %toy-model (&optional (dropout nil))
  (let ((m (make-sequential)))
    (seq-add! m (make-dense 4 :in-dim 3))
    (when dropout (seq-add! m (make-dropout 0.5)))
    (seq-add! m (make-dense 2 :in-dim 4))
    m))

(defun %prime (m &optional (in 3) (out 2))
  "跑一次 forward+backward，让每层都有梯度可更新。"
  (forward m (%rand (list 2 in)))
  (backward m (vt-zeros (list 2 out)))
  m)

;; K1：整模型一次 UPDATE! = 一步
(let* ((m (%prime (%toy-model)))
       (opt (make-adam :lr 1e-3)))
  (update! m opt)
  (check (= 1 (optimizer-step-count opt))
         "K1 整模型一次 update! 只推进 1 步"))

;; K2：逐层 UPDATE! 是 N 次独立更新，就应该是 N 步
;;     （旧实现在这里也是 N，但同一份文档又说「一次更新」，两者矛盾；
;;      现在语义统一：逐层调 N 次 = N 步，要合并用 K3 的写法）
(let* ((m (%prime (%toy-model)))
       (opt (make-sgd :lr 1e-3)))
  (dolist (l (seq-layers m)) (update! l opt))
  (check (= 2 (optimizer-step-count opt))
         "K2 逐层 update! 两次 = 2 步（语义自洽，不是 bug）"))

;; K3：WITH-OPTIMIZER-STEP 把多次 UPDATE! 归并为一步
(let* ((m (%prime (%toy-model)))
       (opt (make-adam :lr 1e-3)))
  (with-optimizer-step (opt)
    (dolist (l (seq-layers m)) (update! l opt)))
  (check (= 1 (optimizer-step-count opt))
         "K3 with-optimizer-step 包两次 update! 仍只推进 1 步"))

;; K4：MODEL-UPDATE! 与 UPDATE! 必须是同一条路径
(let* ((m (%prime (%toy-model)))
       (opt (make-adam :lr 1e-3)))
  (model-update! m opt)
  (check (= 1 (optimizer-step-count opt))
         "K4 model-update! 与 update! 等价（都只推进 1 步）"))

;; K5：嵌套同一优化器时不重复推进（with-optimizer-step 里再 update! 整个模型）
(let* ((m (%prime (%toy-model)))
       (opt (make-adam :lr 1e-3)))
  (with-optimizer-step (opt)
    (update! m opt)
    (update! m opt))
  (check (= 1 (optimizer-step-count opt))
         "K5 同一优化器的嵌套步作用域不重复计数"))

;; K6：SGD 也计数（旧实现只有 adam/adamw/adagrad 自增，sgd/rmsprop 恒为 0）
(let* ((m (%prime (%toy-model)))
       (opt (make-sgd :lr 1e-3)))
  (update! m opt)
  (check (= 1 (optimizer-step-count opt))
         "K6 sgd 也正确记录步数（旧实现恒为 0）"))

;; K7：偏差校正用的是步号 —— 第 1 步的更新量必须能吃满学习率量级。
;;     用「同样的梯度、两步之后参数位移与 Adam 解析解一致」来钉住：
;;     第 1 步 m̂=g, v̂=g²，更新量 = lr * g/|g| = lr * sign(g)。
(let* ((m (%toy-model))
       (opt (make-adam :lr 0.1d0 :beta1 0.9d0 :beta2 0.999d0 :eps 1d-8)))
  ;; 造一个恒定的梯度，让第一次 update 的解析解可预测
  (forward m (%rand '(2 3)))
  (backward m (vt-const '(2 2) 1.0d0))
  (let ((before (vt-to-list (vt-flatten (third (first (params m)))))))
    (update! m opt)
    (let* ((after (vt-to-list (vt-flatten (third (first (params m))))))
           (deltas (mapcar (lambda (a b) (abs (- a b))) before after))
           (nonzero (remove-if (lambda (d) (< d 1d-12)) deltas)))
      ;; 所有非零位移应约等于 lr（0.1），说明 t=1 的偏差校正生效；
      ;; 若 t 被算成 N（逐层写法），位移会小一个量级。
      (check (and nonzero
                  (every (lambda (d) (< (abs (- d 0.1d0)) 1d-3)) nonzero))
             (format nil "K7 第 1 步位移 ≈ lr（偏差校正取 t=1，实得 ~a）"
                     (if nonzero (first nonzero) 0.0d0))))))

;;; ==================================================================
;;; L 组：模型加载态
;;; ==================================================================
;;; *TRAINING-MODE* 默认 T，而 LOAD-MODEL 过去不碰它 ——
;;; save → load → predict 是最常见流程，结果 dropout 仍然生效，
;;; 同一输入两次 forward 给出不同答案，且没有任何提示。

(format t "~&~%==== L 组：模型加载态 ====~%")

(defvar *probe-model-path* "/tmp/clnn-runtime-contract-model.sexp")

(let ((m (%toy-model t)))
  (set-model-training! m t)
  (save-model m *probe-model-path*)
  (let ((m2 (load-model *probe-model-path*)))
    (check (null (training-p m2))
           "L1 load-model 后是推理态")
    (let* ((x (%rand '(2 3)))
           (a (forward m2 x))
           (b (forward m2 x)))
      (check (equalp (vt-to-list (vt-flatten a)) (vt-to-list (vt-flatten b)))
             "L2 load 后同一输入两次 forward 结果相同（dropout 已关闭）"))
    (check (eq t (training-p (train! m2)))
           "L3 train! 切回训练态")
    (check (null (training-p (eval! m2)))
           "L4 eval! 切到推理态")
    ;; 加载后的参数必须全部是计算域（normalize-params! 的验收）
    (check (every (lambda (pg)
                    (let ((v (third pg)))
                      (or (not v) (eq :float64 (vt-dtype v)))))
                  (params m2))
           "L5 load 后所有参数都是 float64")))

(ignore-errors (delete-file *probe-model-path*))

;;; ==================================================================
(format t "~&~%==== 汇总: ~a PASS / ~a FAIL ====~%" *pass* *fail*)
(when (plusp *fail*)
  (error "运行期契约测试失败 ~a 项" *fail*))
