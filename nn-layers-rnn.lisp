(in-package #:nn)

(defun %rnn-zeros (shape like)
  "造一个与 LIKE 同 dtype 的全零张量（RNN 状态 / 累加器的起点）。

   **v2.2 起它已经是个「带形状参数的 VT-ZEROS」**：计算域唯一（§2.0），
   `(value-float-dtype anything)` 恒返回 :float64，所以本函数恒等于
   `(vt-zeros shape :dtype :float64)`。

   那为什么还留着？两个原因：
     1. 14 个调用点的形状参数与 SIG 保持稳定，改的话 diff 全是噪音；
     2. `LIKE` 参数是**一个不变量的书面记录**：「这个缓冲必须与某个真实输入
        或上游梯度同 dtype」。将来若真要支持别的计算域，参照量的选择仍然
        是唯一的判据 —— 而这条不变量在 v2.1 是踩过坑才立起来的。

   历史（值得留着，因为它是「为什么不要多精度」的原始证据）：
   v2.0 时代本函数要**按 LIKE 选 dtype**，因为要让 f32 模型的缓冲跟住 f32。
   当时踩过两次坑：用 `(vt-zeros shape)` 会缺省造 f64（只升不降）；用
   `(vt-zeros-like like)` 又只拷形状（LIKE 常是 (B,H) 而缓冲是 (B,T,H)）。
   最后还要求「参照量必须是真正的输入或梯度，不能是 f64 累加器」——继承
   累加器等于没改。这些坑合起来就是缺口 G-11。"
  (vt-zeros shape :dtype (value-float-dtype (vt-dtype like))))

(defvar *rnn-accumulate-grads* nil
  "非 NIL 时 RNN-CELL 的 backward 把权重梯度**累加**到已有槽上。

   默认 NIL（覆盖写），与 dense / conv2d / lstm / gru 保持同一条契约
   （§5.1：一次 forward+backward 之后，grads 给的是「这一次反向算出的梯度」，
   重复调用不得叠加）。

   唯一需要累加的场合是 RNN-SEQUENCE：它只持有一个 cell，按时间步倒序反复
   调用 cell 的 backward，权重梯度必须跨步累加。该层在自己的 backward 内部
   动态绑定本变量为 T，并在循环开始前清空 cell 的梯度槽 —— 于是「跨时间步
   累加」被封装在层内部，对外仍是「覆盖写」。")

;;; ------------------------------------------------------------------
;;; dtype 说明：本文件的计算链跟随输入/参数的浮点 dtype（f32 全链已通）
;;; ------------------------------------------------------------------
;;; 历史（契约文档 G-11）：本文件曾整体固定在 float64，即使传入 f32 参数与
;;; f32 输入，rnn-cell / lstm / gru / rnn-sequence 的前向输出、dX、dw 全是
;;; float64。原因有两处，缺一不可：
;;;
;;; 1) 状态/累加缓冲用 (VT-ZEROS (LIST BATCH HS)) 起始，缺省 :float64；
;;;    它与 VT-MATMUL 的结果相加后整条门控链变宽，dtype 只升不降。
;;; 2) 门控导数写成 (VT-- 1.0D0 X)。当时的 clvt 把 VT-- 的首个操作数无条件
;;;    ENSURE-VT 成 0 维张量，绕过 NEP 50 的弱标量提升，于是
;;;    「double 标量 − float32 张量」→ float64；而 (VT-- X 1.0D0) 仍是 f32。
;;;
;;; ② 已由上游 clvt 修正（CONVENTIONS §16.31.7：十一个算子补齐弱标量，
;;; 二元提升恢复可交换），因此 `(VT-- 1.0D0 X)` 现在跟随 X 的 dtype，
;;; 无需改写。① 只能在本文件解决：所有零张量起点一律经 %RNN-ZEROS 按
;;; 参照张量的 dtype 构造，见下。约束是**参照张量必须是真正的输入
;;; （x / grad-output）**，不能是某个 f64 累加器 —— 继承后者等于没改。

(defclass rnn-cell (layer)
  ((input-size :initarg :input-size
	       :reader rnn-input-size)
   (hidden-size :initarg :hidden-size
		:reader rnn-hidden-size)
   (activation :initarg :activation
	       :initform :tanh
	       :reader rnn-activation)
   (wih :initarg :wih
	:initform nil
	:accessor rnn-wih)
   (whh :initarg :whh
	:initform nil
	:accessor rnn-whh)
   (bih :initarg :bih
	:initform nil
	:accessor rnn-bih)
   (dwih :initarg :dwih
	 :initform nil
	 :accessor rnn-dwih)
   (dwhh :initarg :dwhh
	 :initform nil
	 :accessor rnn-dwhh)
   (dbih :initarg :dbih
	 :initform nil
	 :accessor rnn-dbih)
   (input-cache :initarg :input-cache
		:initform nil
		:accessor rnn-input-cache)
   (h-prev-cache :initarg :h-prev-cache
		 :initform nil
		 :accessor rnn-h-prev-cache)
   (h-cache :initarg :h-cache
	    :initform nil
	    :accessor rnn-h-cache)
   (state :initarg :state
          :initform nil
          :accessor rnn-state))
  (:documentation
   "h_t = act(x_t @ W_ih.T + h_{t-1} @ W_hh.T + b_ih)
    多步展开时可通过 (setf (rnn-state l) h-prev) 在每步前注入隐状态;
    forward 返回后 (rnn-state l) 自动保存最新 h, 可链式读取."))

(defun make-rnn-cell (input-size hidden-size
                      &key activation (name "rnn-cell") (trainable t))
  (make-instance 'rnn-cell
		 :input-size input-size
		 :hidden-size hidden-size
		 :activation (or activation :tanh)
		 :name name :trainable trainable))

(defun ensure-rnn-cell-params (l)
  (unless (rnn-wih l)
    (let ((is (rnn-input-size l))
          (hs (rnn-hidden-size l)))
      (setf (rnn-wih l)
            (vt-scale (vt-random-normal (list hs is))
                      (sqrt (/ 1.0d0 is))))
      (setf (rnn-whh l)
            (vt-scale (vt-random-normal (list hs hs))
                      (sqrt (/ 1.0d0 hs))))
      (setf (rnn-bih l) (vt-zeros (list hs))))))

(defmethod forward ((l rnn-cell) input)
  "单步 RNN: h_t = act(x_t Wih^T + h_{t-1} Whh^T + b).
   若 (rnn-state l) 已设置 (例如循环展开时上一步返回的 h), 则将其作为 h_{t-1};
   否则初始化为全零. forward 结束后 (rnn-state l) 会被更新为当前 h, 支持链式调用."
  (ensure-rnn-cell-params l)
  (let* ((x (if (= (length (vt-shape input)) 1)
                (vt-reshape input (list 1 (rnn-input-size l)))
                input))
         (batch (first (vt-shape x)))
         (h-prev (or (rnn-state l)
                     (%rnn-zeros (list batch (rnn-hidden-size l)) x)))
         (pre-act
           (vt-+ (vt-+ (vt-matmul x (vt-transpose (rnn-wih l)))
                       (vt-matmul h-prev (vt-transpose (rnn-whh l))))
                 (rnn-bih l)))
         (h (ecase (rnn-activation l)
              ((:tanh tanh) (vt-tanh pre-act))
              ((:relu relu) (vt-relu pre-act)))))
    (setf (rnn-input-cache l) x)
    (setf (rnn-h-prev-cache l) h-prev)
    (setf (rnn-h-cache l) h)
    (setf (rnn-state l) h)
    h))

(defun rnn-cell-step (l x h-prev)
  (ensure-rnn-cell-params l)
  (let* ((pre-act
           (vt-+ (vt-+ (vt-matmul x (vt-transpose (rnn-wih l)))
                       (vt-matmul h-prev (vt-transpose (rnn-whh l))))
                 (rnn-bih l)))
         (h (ecase (rnn-activation l)
              ((:tanh tanh) (vt-tanh pre-act))
              ((:relu relu) (vt-relu pre-act)))))
    h))

(defmethod backward ((l rnn-cell) grad-output)
  "返回 (values dx dh-prev); sequential 反向默认取 dx, 多步展开可用
   multiple-value-bind 取得 dh-prev 传给上一时刻."
  (let* ((x (rnn-input-cache l))
         (h-prev (rnn-h-prev-cache l))
         (h (rnn-h-cache l))
         (wih (rnn-wih l))
         (whh (rnn-whh l))
         (d-act (ecase (rnn-activation l)
                  ((:tanh tanh) (vt-- 1.0d0 (vt-* h h)))
                  ((:relu relu) (vt-relu-derivative h))))
         (dh (vt-* grad-output d-act))
         (dwih (vt-matmul (vt-transpose dh) x))
         (dwhh (vt-matmul (vt-transpose dh) h-prev))
         (dbih (vt-sum dh :axis 0))
         (dx (vt-matmul dh wih))
         (dh-prev (vt-matmul dh whh)))
    ;; 梯度语义（§5.1）：默认覆盖写。只有 RNN-SEQUENCE 通过动态变量要求累加
    ;; —— 它复用同一个 cell 走完整条序列，跨时间步必须叠加。
    (flet ((put (slot value)
             (setf (slot-value l slot)
                   (if *rnn-accumulate-grads*
                       (let ((old (slot-value l slot)))
                         (if old (vt-+ old value) value))
                       value))))
      (put 'dwih dwih)
      (put 'dwhh dwhh)
      (put 'dbih dbih))
    (values dx dh-prev)))

(defun reset-rnn-cell-state! (cell)
  "重置 rnn-cell 的隐状态缓存, 用于开始新一轮序列前清空."
  (setf (rnn-state cell) nil))

(defmethod params ((l rnn-cell))
  (list (list l "wih" (rnn-wih l)
              #'(lambda (v) (setf (rnn-wih l) v)))
        (list l "whh" (rnn-whh l)
              #'(lambda (v) (setf (rnn-whh l) v)))
        (list l "bih" (rnn-bih l)
              #'(lambda (v) (setf (rnn-bih l) v)))))

(defmethod grads ((l rnn-cell))
  (list (cons "wih" (rnn-dwih l))
        (cons "whh" (rnn-dwhh l))
        (cons "bih" (rnn-dbih l))))

(defclass lstm (layer)
  ((input-size :initarg :input-size
	       :initform nil
	       :reader lstm-input-size)
   (hidden-size :initarg :hidden-size
		:initform nil
		:reader lstm-hidden-size)
   (weight-ih :initarg :weight-ih
	      :initform nil
	      :accessor lstm-weight-ih)
   (weight-hh :initarg :weight-hh
	      :initform nil
	      :accessor lstm-weight-hh)
   (bias-ih :initarg :bias-ih
	    :initform nil
	    :accessor lstm-bias-ih)
   (bias-hh :initarg :bias-hh
	    :initform nil
	    :accessor lstm-bias-hh)
   (dweight-ih :initarg :dweight-ih
	       :initform nil
	       :accessor lstm-dweight-ih)
   (dweight-hh :initarg :dweight-hh
	       :initform nil
	       :accessor lstm-dweight-hh)
   (dbias-ih :initarg :dbias-ih
	     :initform nil
	     :accessor lstm-dbias-ih)
   (dbias-hh :initarg :dbias-hh
	     :initform nil
	     :accessor lstm-dbias-hh)
   (cache :initarg :cache
	  :initform nil
	  :accessor lstm-cache)
   ;; 外部可注入的初始状态
   (h-0 :initarg :h-0 :initform nil
	:accessor lstm-h-0 :type (or null vt)
	:documentation
	"可选的初始隐藏状态 h_0, 形状 (batch, hidden-size). NIL 则用零.")
   (c-0 :initarg :c-0 :initform nil
	:accessor lstm-c-0 :type (or null vt)
	:documentation
	"可选的初始细胞状态 c_0, 形状 (batch, hidden-size). NIL 则用零."))
  (:documentation "LSTM. 支持可选的外部初始状态 (h-0, c-0)."))

(defun make-lstm (input-size hidden-size
                  &key (name "lstm") (trainable t) h-0 c-0)
  (make-instance 'lstm
		 :input-size input-size
		 :hidden-size hidden-size
		 :name name :trainable trainable
		 :h-0 h-0 :c-0 c-0))

(defun ensure-lstm-params (l)
  (unless (lstm-weight-ih l)
    (let* ((is (lstm-input-size l))
           (hs (lstm-hidden-size l))
           (gate-size (* 4 hs)))
      (setf (lstm-weight-ih l)
            (vt-scale (vt-random-normal (list gate-size is))
                      (sqrt (/ 2.0d0 (+ is hs)))))
      (setf (lstm-weight-hh l)
            (vt-scale (vt-random-normal (list gate-size hs))
                      (sqrt (/ 2.0d0 (+ hs hs)))))
      (let ((b-ih (vt-zeros (list gate-size)))
            (b-hh (vt-zeros (list gate-size))))
        ;; PyTorch 标准：LSTM forget gate 初始 bias 设为 1（合计 2）
        ;; 使初始遗忘门 sigmoid(2) ≈ 0.88，避免训练初期梯度消失
        (dotimes (i hs)
          (setf (vt-ref b-ih (+ hs i)) 1.0d0)
          (setf (vt-ref b-hh (+ hs i)) 1.0d0))
        (setf (lstm-bias-ih l) b-ih)
        (setf (lstm-bias-hh l) b-hh)))))

(defmethod forward ((l lstm) input)
  (ensure-lstm-params l)
  (let* ((shape (vt-shape input))
         (batch (first shape))
         (seq-len (second shape))
         (hs (lstm-hidden-size l))
         (wih (lstm-weight-ih l))
         (whh (lstm-weight-hh l))
         (bih (lstm-bias-ih l))
         (bhh (lstm-bias-hh l))
         ;; 支持外部注入的初始状态，否则用零（dtype 跟随输入）
         (h (or (lstm-h-0 l) (%rnn-zeros (list batch hs) input)))
         (c (or (lstm-c-0 l) (%rnn-zeros (list batch hs) input)))
         (output (%rnn-zeros (list batch seq-len hs) input))
         (all-gates '()) (all-c '())
         (all-h '()) (all-x '()))
    (dotimes (i seq-len)
      (let* ((x-t (vt-slice input (list :all) (list i) (list :all)))
             (gates
               (vt-+ (vt-+
                      (vt-matmul x-t (vt-transpose wih))
                      (vt-matmul h (vt-transpose whh)))
                     (vt-+ bih bhh)))
             (i-gate (vt-sigmoid
                      (vt-slice gates (list :all) `(0 ,hs))))
             (f-gate (vt-sigmoid
                      (vt-slice gates (list :all) `(,hs ,(* 2 hs)))))
             (g-gate (vt-tanh
                      (vt-slice gates (list :all) `(,(* 2 hs) ,(* 3 hs)))))
             (o-gate (vt-sigmoid
                      (vt-slice gates (list :all) `(,(* 3 hs) ,(* 4 hs)))))
             (c-new (vt-+ (vt-* f-gate c)
                          (vt-* i-gate g-gate)))
             (h-new (vt-* o-gate (vt-tanh c-new))))
        (setf (vt-slice output (list :all) (list i) (list :all)) h-new)
        (setf h h-new)
        (setf c c-new)
        (push (list (vt-copy i-gate) (vt-copy f-gate)
                    (vt-copy g-gate) (vt-copy o-gate))
              all-gates)
        (push (vt-copy c-new) all-c)
        (push (vt-copy h-new) all-h)
        (push (vt-copy x-t) all-x)))
    (setf (lstm-cache l)
          (list :gates (nreverse all-gates)
                :c-states (nreverse all-c)
                :h-states (nreverse all-h)
                :x-states (nreverse all-x)
                :batch batch :seq-len seq-len :hs hs))
    (values output h c)))

(defmethod backward ((l lstm) grad-output)
  (let* ((cache (lstm-cache l))
         (all-gates (getf cache :gates))
         (all-c (getf cache :c-states))
         (all-h (getf cache :h-states))
         (all-x (getf cache :x-states))
         (batch (getf cache :batch))
         (seq-len (getf cache :seq-len))
         (hs (getf cache :hs))
         (is (lstm-input-size l))
         (wih (lstm-weight-ih l))
         (whh (lstm-weight-hh l))
         (dwih-acc (%rnn-zeros (list (* 4 hs) is) grad-output))
         (dwhh-acc (%rnn-zeros (list (* 4 hs) hs) grad-output))
         (dbih-acc (%rnn-zeros (list (* 4 hs)) grad-output))
         (dbhh-acc (%rnn-zeros (list (* 4 hs)) grad-output))
         (dh-next (%rnn-zeros (list batch hs) grad-output))
         (dc-next (%rnn-zeros (list batch hs) grad-output))
         (grad-input (%rnn-zeros (list batch seq-len is) grad-output))
         (zero-h (%rnn-zeros (list batch hs) grad-output))
         ;; 使用用户提供的初始状态（若有）
         (h-0 (or (lstm-h-0 l) zero-h))
         (c-0 (or (lstm-c-0 l) zero-h)))
    (dotimes (i seq-len)
      (let* ((idx (- seq-len i 1))
             (gates (nth idx all-gates))
             (i-gate (first gates))
             (f-gate (second gates))
             (g-gate (third gates))
             (o-gate (fourth gates))
             (c-prev (if (= idx 0)
                         c-0
                         (nth (1- idx) all-c)))
             (c-cur (nth idx all-c))
             (x-t (nth idx all-x))
             (dh (vt-+ (vt-slice grad-output (list :all) (list idx) (list :all))
                       dh-next))
             (tanh-c (vt-tanh c-cur))
             (dtanh-c (vt-- 1.0d0 (vt-* tanh-c tanh-c)))
             (dc (vt-+ (vt-* dh (vt-* o-gate dtanh-c))
                       dc-next))
             (di (vt-* dc (vt-* g-gate i-gate
                                (vt-- 1.0d0 i-gate))))
             (df (vt-* dc (vt-* c-prev f-gate
                                (vt-- 1.0d0 f-gate))))
             (dg (vt-* dc (vt-* i-gate
                                (vt-- 1.0d0 (vt-* g-gate g-gate)))))
             (do-g (vt-* dh tanh-c
                         (vt-* o-gate (vt-- 1.0d0 o-gate))))
             (d-gates (vt-concatenate -1 di df dg do-g)))
        (setf dwih-acc
              (vt-+ dwih-acc
                    (vt-matmul (vt-transpose d-gates) x-t)))
        (let ((h-prev (if (= idx 0)
                          h-0
                          (nth (1- idx) all-h))))
          (setf dwhh-acc
                (vt-+ dwhh-acc
                      (vt-matmul (vt-transpose d-gates) h-prev))))
        (setf dbih-acc
              (vt-+ dbih-acc (vt-sum d-gates :axis 0)))
        (setf dbhh-acc
              (vt-+ dbhh-acc (vt-sum d-gates :axis 0)))
        (setf dh-next (vt-matmul d-gates whh))
        (setf (vt-slice grad-input (list :all) (list idx) (list :all))
              (vt-matmul d-gates wih))
        (setf dc-next (vt-* dc f-gate))))
    (setf (lstm-dweight-ih l) dwih-acc)
    (setf (lstm-dweight-hh l) dwhh-acc)
    (setf (lstm-dbias-ih l) dbih-acc)
    (setf (lstm-dbias-hh l) dbhh-acc)
    (values grad-input
            (when (lstm-h-0 l) dh-next)
            (when (lstm-c-0 l) dc-next))))

(defmethod params ((l lstm))
  (list (list l "weight_ih" (lstm-weight-ih l)
              #'(lambda (v) (setf (lstm-weight-ih l) v)))
        (list l "weight_hh" (lstm-weight-hh l)
              #'(lambda (v) (setf (lstm-weight-hh l) v)))
        (list l "bias_ih" (lstm-bias-ih l)
              #'(lambda (v) (setf (lstm-bias-ih l) v)))
        (list l "bias_hh" (lstm-bias-hh l)
              #'(lambda (v) (setf (lstm-bias-hh l) v)))))

(defmethod grads ((l lstm))
  (list (cons "weight_ih" (lstm-dweight-ih l))
        (cons "weight_hh" (lstm-dweight-hh l))
        (cons "bias_ih" (lstm-dbias-ih l))
        (cons "bias_hh" (lstm-dbias-hh l))))

(defclass gru (layer)
  ((input-size :initform nil
	       :initarg :input-size
	       :reader gru-input-size)
   (hidden-size :initform nil
		:initarg :hidden-size
		:reader gru-hidden-size)
   (weight-ih :initform nil
	      :initarg :weight-ih
	      :accessor gru-weight-ih)
   (weight-hh :initform nil
	      :initarg :weight-hh
	      :accessor gru-weight-hh)
   (bias-ih :initform nil
	    :initarg :bias-ih
	    :accessor gru-bias-ih)
   (bias-hh :initform nil
	    :initarg :bias-hh
	    :accessor gru-bias-hh)
   (dweight-ih :initform nil
	       :initarg :dweight-ih
	       :accessor gru-dweight-ih)
   (dweight-hh :initform nil
	       :initarg :dweight-hh
	       :accessor gru-dweight-hh)
   (dbias-ih :initform nil
	     :initarg :dbias-ih
	     :accessor gru-dbias-ih)
   (dbias-hh :initform nil
	     :initarg :dbias-hh
	     :accessor gru-dbias-hh)
   (cache :initform nil
	  :initarg :cache
	  :accessor gru-cache))
  (:documentation "GRU"))

(defun make-gru (input-size hidden-size
                 &key (name "gru") (trainable t))
  (make-instance 'gru
		 :input-size input-size
		 :hidden-size hidden-size
		 :name name :trainable trainable))

(defun ensure-gru-params (l)
  (unless (gru-weight-ih l)
    (let* ((is (gru-input-size l))
           (hs (gru-hidden-size l))
           (gate-size (* 3 hs)))
      (setf (gru-weight-ih l)
            (vt-scale (vt-random-normal (list gate-size is))
                      (sqrt (/ 2.0d0 (+ is hs)))))
      (setf (gru-weight-hh l)
            (vt-scale (vt-random-normal (list gate-size hs))
                      (sqrt (/ 2.0d0 (+ hs hs)))))
      (setf (gru-bias-ih l) (vt-zeros (list gate-size)))
      (setf (gru-bias-hh l) (vt-zeros (list gate-size))))))

(defmethod forward ((l gru) input)
  (ensure-gru-params l)
  (let* ((shape (vt-shape input))
         (batch (first shape))
         (seq-len (second shape))
         (hs (gru-hidden-size l))
         (wih (gru-weight-ih l))
         (whh (gru-weight-hh l))
         (bih (gru-bias-ih l))
         (bhh (gru-bias-hh l))
         (w-in (vt-slice wih
                         `(,(* 2 hs) ,(* 3 hs)) (list :all)))
         (b-in (vt-slice bih
                         `(,(* 2 hs) ,(* 3 hs))))
         (w-hn (vt-slice whh
                         `(,(* 2 hs) ,(* 3 hs)) (list :all)))
         (b-hn (vt-slice bhh
                         `(,(* 2 hs) ,(* 3 hs))))
         (h (%rnn-zeros (list batch hs) input))
         (output (%rnn-zeros (list batch seq-len hs) input))
         (all-r '()) (all-z '()) (all-n '())
         (all-h '()) (all-x '()) (all-hn-gate '()))
    (dotimes (i seq-len)
      (let* ((x-t (vt-slice input (list :all) (list i) (list :all)))
             (gates
               (vt-+ (vt-+
                      (vt-matmul x-t (vt-transpose wih))
                      (vt-matmul h (vt-transpose whh)))
                     (vt-+ bih bhh)))
             (r (vt-sigmoid
                 (vt-slice gates (list :all) (list 0 hs))))
             (z (vt-sigmoid
                 (vt-slice gates (list :all) `(,hs ,(* 2 hs)))))
             (hn-linear
               (vt-+ (vt-matmul h (vt-transpose w-hn))
                     b-hn))
             (n (vt-tanh
                 (vt-+ (vt-+
                        (vt-matmul x-t
                                   (vt-transpose w-in))
                        (vt-* r hn-linear))
                       b-in)))
             (h-new (vt-+ (vt-* (vt-- 1.0d0 z) n)
                          (vt-* z h))))
        (setf (vt-slice output (list :all) (list i) (list :all)) h-new)
        (setf h h-new)
        (push (vt-copy r) all-r)
        (push (vt-copy z) all-z)
        (push (vt-copy n) all-n)
        (push (vt-copy h-new) all-h)
        (push (vt-copy x-t) all-x)
        (push (vt-copy hn-linear) all-hn-gate)))
    (setf (gru-cache l)
          (list :r (nreverse all-r)
                :z (nreverse all-z)
                :n (nreverse all-n)
                :h (nreverse all-h)
                :x (nreverse all-x)
                :hn-linear (nreverse all-hn-gate)
                :batch batch :seq-len seq-len :hs hs))
    (values output h)))

(defmethod backward ((l gru) grad-output)
  (let* ((cache (gru-cache l))
         (all-r (getf cache :r))
         (all-z (getf cache :z))
         (all-n (getf cache :n))
         (all-h (getf cache :h))
         (all-x (getf cache :x))
         (all-hn (getf cache :hn-linear))
         (batch (getf cache :batch))
         (seq-len (getf cache :seq-len))
         (hs (getf cache :hs))
         (is (gru-input-size l))
         (wih (gru-weight-ih l))
         (whh (gru-weight-hh l))
         (w-in (vt-slice wih `(,(* 2 hs) ,(* 3 hs)) (list :all)))
         (w-hn (vt-slice whh `(,(* 2 hs) ,(* 3 hs)) (list :all)))
         (dwih-acc (%rnn-zeros (list (* 3 hs) is) grad-output))
         (dwhh-acc (%rnn-zeros (list (* 3 hs) hs) grad-output))
         (dbih-acc (%rnn-zeros (list (* 3 hs)) grad-output))
         (dbhh-acc (%rnn-zeros (list (* 3 hs)) grad-output))
         (dh-next (%rnn-zeros (list batch hs) grad-output))
         (grad-input (%rnn-zeros (list batch seq-len is) grad-output))
         (zero-h (%rnn-zeros (list batch hs) grad-output)))
    (dotimes (i seq-len)
      (let* ((idx (- seq-len i 1))
             (r-t (nth idx all-r))
             (z-t (nth idx all-z))
             (n-t (nth idx all-n))
             (hn-t (nth idx all-hn))
             (h-prev (if (= idx 0)
                         zero-h
                         (nth (1- idx) all-h)))
             (x-t (nth idx all-x))
             (dh (vt-+ (vt-slice grad-output (list :all) (list idx) (list :all))
                       dh-next))
             (dz (vt-* dh (vt-- h-prev n-t)))
             (dn (vt-* dh (vt-- 1.0d0 z-t)))
             (dz-pre (vt-* dz (vt-* z-t (vt-- 1.0d0 z-t))))
             (dn-pre (vt-* dn
                           (vt-- 1.0d0 (vt-* n-t n-t))))
             (dr (vt-* dn-pre hn-t))
             (dhn-linear (vt-* dn-pre r-t))
             (dr-pre (vt-* dr (vt-* r-t
                                    (vt-- 1.0d0 r-t))))
             (d-gates-rz (vt-concatenate -1
                                         dr-pre dz-pre)))
        ;; 1. Wih
        (let ((full-dw (%rnn-zeros (list (* 3 hs) is) dwih-acc)))
          (setf (vt-slice full-dw `(0 ,(* 2 hs)) (list :all))
                (vt-matmul (vt-transpose d-gates-rz) x-t))
          (setf (vt-slice full-dw
                          `(,(* 2 hs) ,(* 3 hs)) (list :all))
                (vt-matmul (vt-transpose dn-pre) x-t))
          (setf dwih-acc (vt-+ dwih-acc full-dw)))
        ;; 2. Whh
        (let ((full-dwh (%rnn-zeros (list (* 3 hs) hs) dwhh-acc)))
          (setf (vt-slice full-dwh `(0 ,(* 2 hs)) (list :all))
                (vt-matmul (vt-transpose d-gates-rz) h-prev))
          (setf (vt-slice full-dwh
                          `(,(* 2 hs) ,(* 3 hs)) (list :all))
                (vt-matmul (vt-transpose dhn-linear) h-prev))
          (setf dwhh-acc (vt-+ dwhh-acc full-dwh)))
        ;; 3. bih
        (let ((full-db (%rnn-zeros (list (* 3 hs)) dbih-acc)))
          (setf (vt-slice full-db `(0 ,(* 2 hs)))
                (vt-sum d-gates-rz :axis 0))
          (setf (vt-slice full-db
                          `(,(* 2 hs) ,(* 3 hs)))
                (vt-sum dn-pre :axis 0))
          (setf dbih-acc (vt-+ dbih-acc full-db)))
        ;; 4. bhh
        (let ((full-dbh (%rnn-zeros (list (* 3 hs)) dbhh-acc)))
          (setf (vt-slice full-dbh `(0 ,(* 2 hs)))
                (vt-sum d-gates-rz :axis 0))
          (setf (vt-slice full-dbh
                          `(,(* 2 hs) ,(* 3 hs)))
                (vt-sum dhn-linear :axis 0))
          (setf dbhh-acc (vt-+ dbhh-acc full-dbh)))
        ;; 5. dX
        (let* ((wih-rz (vt-slice wih `(0 ,(* 2 hs)) (list :all)))
               (dx-t (vt-+ (vt-matmul d-gates-rz wih-rz)
                           (vt-matmul dn-pre w-in))))
          (setf (vt-slice grad-input (list :all) (list idx) (list :all)) dx-t))
        ;; 6. dh_prev
        (let* ((whh-rz (vt-slice whh `(0 ,(* 2 hs)) (list :all)))
               (dh-from-rz (vt-matmul d-gates-rz whh-rz))
               (dh-from-n (vt-matmul dhn-linear w-hn))
               (dh-from-z (vt-* z-t dh)))
          (setf dh-next (vt-+ (vt-+ dh-from-rz dh-from-n)
                              dh-from-z)))))
    (setf (gru-dweight-ih l) dwih-acc)
    (setf (gru-dweight-hh l) dwhh-acc)
    (setf (gru-dbias-ih l) dbih-acc)
    (setf (gru-dbias-hh l) dbhh-acc)
    grad-input))

(defmethod params ((l gru))
  (list (list l "weight_ih" (gru-weight-ih l)
              #'(lambda (v) (setf (gru-weight-ih l) v)))
        (list l "weight_hh" (gru-weight-hh l)
              #'(lambda (v) (setf (gru-weight-hh l) v)))
        (list l "bias_ih" (gru-bias-ih l)
              #'(lambda (v) (setf (gru-bias-ih l) v)))
        (list l "bias_hh" (gru-bias-hh l)
              #'(lambda (v) (setf (gru-bias-hh l) v)))))

(defmethod grads ((l gru))
  (list (cons "weight_ih" (gru-dweight-ih l))
        (cons "weight_hh" (gru-dweight-hh l))
        (cons "bias_ih" (gru-dbias-ih l))
        (cons "bias_hh" (gru-dbias-hh l))))


;; ---- grad-slots ----
(defmethod grad-slots ((l rnn-cell)) '(dwih dwhh dbih))

(defmethod grad-slots ((l lstm))
  '(dweight-ih dweight-hh dbias-ih dbias-hh))

(defmethod grad-slots ((l gru))
  '(dweight-ih dweight-hh dbias-ih dbias-hh))

(defmethod cache-slots ((l rnn-cell))
  ;; 注意：不含 state（链式调用状态，由 reset-rnn-cell-state! 单独管理）
  '(input-cache h-prev-cache h-cache))

(defmethod cache-slots ((l lstm))
  '(cache))

(defmethod cache-slots ((l gru))
  '(cache))


;;; ============================================================
;;; rnn-sequence: 在时间维度自动展开 rnn-cell 的包装层
;;; ============================================================
;;;
;;; 单步的 rnn-cell 通过 (values dx dh-prev) 支持手动时间展开，
;;; 但 sequential 的 backward 只取第一个返回值, 无法直接使用。
;;; 本层把整个序列的展开逻辑内聚到自身:
;;;   forward:  (B, T, D)  ->  (B, T, H)
;;;   backward: (B, T, H)  ->  (B, T, D)
;;; 内部按时间步倒序调用 cell 的 backward, 并把 dh_prev 传给上一步,
;;; 所有时间步的权重梯度自然累积到共享的 cell 上。

(defclass rnn-sequence (layer)
  ((cell :initarg :cell :reader rnn-seq-cell)
   ;; 本层自己缓存每步的中间量 (cell 自身只存最后一次 forward)
   (x-cache      :initform nil :accessor rnn-seq-x-cache)
   (h-prev-cache :initform nil :accessor rnn-seq-h-prev-cache)
   (h-cache      :initform nil :accessor rnn-seq-h-cache))
  (:documentation "在时间维度展开 rnn-cell 的包装层。输入 (B,T,D), 输出 (B,T,H)。"))

(defun make-rnn-sequence (input-size hidden-size
                          &key activation
                            (name "rnn-sequence")
                            (trainable t))
  (make-instance 'rnn-sequence
                 :cell (make-rnn-cell input-size hidden-size
                                      :activation (or activation :tanh))
                 :name name :trainable trainable))

(defmethod forward ((l rnn-sequence) input)
  (let* ((shape (vt-shape input))
         (batch (first shape))
         (seq-len (second shape))
         (cell (rnn-seq-cell l))
         (hs (rnn-hidden-size cell))
         (output (%rnn-zeros (list batch seq-len hs) input)))
    (ensure-rnn-cell-params cell)
    (let ((h-prev (%rnn-zeros (list batch hs) input))
          (xs '()) (h-preves '()) (hs-list '()))
      (dotimes (i seq-len)                                     
        (let* ((x-t (vt-slice input  (list :all) (list i) (list :all)))
               (h-t (rnn-cell-step cell x-t h-prev)))
          (setf (vt-slice output (list :all) (list i) (list :all)) h-t)
          (push x-t    xs)
          (push h-prev h-preves)
          (push h-t    hs-list)
          (setf h-prev h-t)))
      (setf (rnn-seq-x-cache l)      (nreverse xs))
      (setf (rnn-seq-h-prev-cache l) (nreverse h-preves))
      (setf (rnn-seq-h-cache l)      (nreverse hs-list))
      output)))

(defmethod backward ((l rnn-sequence) grad-output)
  (let* ((cell (rnn-seq-cell l))
         (xs       (rnn-seq-x-cache l))
         (h-preves (rnn-seq-h-prev-cache l))
         (hs-list  (rnn-seq-h-cache l))
         (seq-len (length xs))
         (batch (first (vt-shape (first xs))))
         (D (rnn-input-size cell))
         (H (rnn-hidden-size cell))
         (grad-input (%rnn-zeros (list batch seq-len D) grad-output))
         (dh-next (%rnn-zeros (list batch H) grad-output)))
    ;; 本层内部必须跨时间步累加 cell 的权重梯度（同一 cell 被调 T 次）；
    ;; 但跨「调用」不累加：先把 cell 的梯度槽清空，再绑 *RNN-ACCUMULATE-GRADS*，
    ;; 这样第二次 backward 的结果与第一次逐位相同（§5.1 的覆盖写语义）。
    (setf (rnn-dwih cell) nil
          (rnn-dwhh cell) nil
          (rnn-dbih cell) nil)
    (let ((*rnn-accumulate-grads* t))
      (dotimes (i seq-len)
        (let* ((idx  (- seq-len i 1))
               (x-t  (nth idx xs))
               (h-pv (nth idx h-preves))
               (h-t  (nth idx hs-list))
               (g-t  (vt-slice grad-output (list :all) (list idx) (list :all)))) ; idx
          (setf (rnn-input-cache  cell) x-t)
          (setf (rnn-h-prev-cache cell) h-pv)
          (setf (rnn-h-cache      cell) h-t)
          (let ((g-combined (vt-+ g-t dh-next)))
            (multiple-value-bind (dx dh-prev)
                (backward cell g-combined)
              (setf (vt-slice grad-input (list :all) (list idx) (list :all)) dx) ; idx
              (setf dh-next dh-prev))))))
    grad-input))


(defmethod params ((l rnn-sequence)) (params (rnn-seq-cell l)))
(defmethod grads  ((l rnn-sequence)) (grads  (rnn-seq-cell l)))

(defmethod grad-slots  ((l rnn-sequence)) '())
(defmethod cache-slots ((l rnn-sequence))
  '(x-cache h-prev-cache h-cache))

(defmethod set-training! ((l rnn-sequence) mode)
  (call-next-method)
  (when (rnn-seq-cell l)
    (set-training! (rnn-seq-cell l) mode)))
