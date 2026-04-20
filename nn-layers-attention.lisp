(in-package #:nn)

(defun sigmoid-tensor (x)
  (vt-sigmoid x))

(defun sdpa-forward (q k v &optional mask dropout-rate is-training)
  "SDPA 前向."
  (let* ((d-k (coerce (second (vt-shape q)) 'double-float))
         (scale (/ 1.0d0 (sqrt d-k)))
         (scores (vt-scale (vt-matmul q (vt-transpose k)) scale))
         (masked-scores (if mask (funcall mask scores) scores))
         (attn (vt-softmax masked-scores))
         (attn-dropped
	   (if (and is-training
		    (> (or dropout-rate 0.0d0) 0.0d0))
               (let ((dr (or dropout-rate 0.0d0)))
                 (vt-map (lambda (x)
			   (if (< (random 1.0d0) dr)
			       0.0d0
			       (/ x (- 1.0d0 dr))))
			 attn))
               attn))
         (output (vt-matmul attn-dropped v)))
    (values output attn-dropped)))


(defun sdpa-backward (d-output q k v attn)
  "SDPA 反向."
  (let* ((d-k (coerce (second (vt-shape q)) 'double-float))
         (scale (/ 1.0d0 (sqrt d-k)))
         (dv (vt-matmul (vt-transpose attn) d-output))
         (d-attn (vt-matmul d-output (vt-transpose v)))
         (sum-daat (vt-sum (vt-* d-attn attn)
                           :axis -1 :keepdims t))
         (d-scores (vt-* attn (vt-- d-attn sum-daat)))
         (d-scores-scaled (vt-scale d-scores scale))
         (dq (vt-matmul d-scores-scaled k))
         (dk (vt-matmul (vt-transpose d-scores-scaled) q)))
    (values dq dk dv)))

(defclass multi-head-attention (layer)
  ((embed-dim :initarg :embed-dim :reader mha-embed-dim)
   (num-heads :initarg :num-heads :reader mha-num-heads)
   (head-dim :reader mha-head-dim)
   (use-bias :initarg :use-bias :initform t
    :reader mha-use-bias-p)
   (dropout-rate :initarg :dropout-rate :initform 0.0d0
    :accessor mha-dropout-rate)
   (w-q :accessor mha-wq) (w-k :accessor mha-wk)
   (w-v :accessor mha-wv) (w-o :accessor mha-wo)
   (b-q :accessor mha-bq) (b-k :accessor mha-bk)
   (b-v :accessor mha-bv) (b-o :accessor mha-bo)
   (dw-q :accessor mha-dwq) (dw-k :accessor mha-dwk)
   (dw-v :accessor mha-dwv) (dw-o :accessor mha-dwo)
   (db-q :accessor mha-dbq) (db-k :accessor mha-dbk)
   (db-v :accessor mha-dbv) (db-o :accessor mha-dbo)
   (cache :accessor mha-cache))
  (:documentation "多头注意力 (完整实现)."))

(defun make-multi-head-attention
    (embed-dim num-heads &key use-bias dropout-rate
     (name "mha") (trainable t))
  (let ((head-dim (floor embed-dim num-heads)))
    (assert (= (* head-dim num-heads) embed-dim)
            () "embed-dim must be divisible by num-heads")
    (make-instance 'multi-head-attention
      :embed-dim embed-dim :num-heads num-heads
      :use-bias use-bias
      :dropout-rate (or dropout-rate 0.0d0)
      :name name :trainable trainable)))

(defmethod initialize-instance :after
    ((l multi-head-attention) &key)
  (setf (slot-value l 'head-dim)
        (floor (mha-embed-dim l) (mha-num-heads l))))

(defun ensure-mha-params (l)
  (unless (mha-wq l)
    (let* ((d (mha-embed-dim l))
           (std (sqrt (/ 2.0d0 (* 2.0d0 d)))))
      (setf (mha-wq l) (vt-scale (vt-random-normal
                                   (list d d)) std))
      (setf (mha-wk l) (vt-scale (vt-random-normal
                                   (list d d)) std))
      (setf (mha-wv l) (vt-scale (vt-random-normal
                                   (list d d)) std))
      (setf (mha-wo l) (vt-scale (vt-random-normal
                                   (list d d)) std))
      (when (mha-use-bias-p l)
        (setf (mha-bq l) (vt-zeros (list d)))
        (setf (mha-bk l) (vt-zeros (list d)))
        (setf (mha-bv l) (vt-zeros (list d)))
        (setf (mha-bo l) (vt-zeros (list d)))))))

(defmethod forward ((l multi-head-attention) inputs)
  (ensure-mha-params l)
  (destructuring-bind (query key value) inputs
    (let* ((shape-q (vt-shape query))
           (batch (first shape-q))
           (seq-q (second shape-q))
           (seq-k (second (vt-shape key)))
           (seq-v (second (vt-shape value)))
           (nh (mha-num-heads l))
           (hd (mha-head-dim l))
           (ed (mha-embed-dim l))
           (q (if (mha-use-bias-p l)
                  (vt-+ (vt-matmul query (mha-wq l))
                        (mha-bq l))
                  (vt-matmul query (mha-wq l))))
           (k (if (mha-use-bias-p l)
                  (vt-+ (vt-matmul key (mha-wk l))
                        (mha-bk l))
                  (vt-matmul key (mha-wk l))))
           (v (if (mha-use-bias-p l)
                  (vt-+ (vt-matmul value (mha-wv l))
                        (mha-bv l))
                  (vt-matmul value (mha-wv l))))
           (q-4d (vt-reshape q
                             (list batch seq-q nh hd)))
           (k-4d (vt-reshape k
                             (list batch seq-k nh hd)))
           (v-4d (vt-reshape v
                             (list batch seq-v nh hd)))
           (q-t (vt-transpose q-4d '(0 2 1 3)))
           (k-t (vt-transpose k-4d '(0 2 1 3)))
           (v-t (vt-transpose v-4d '(0 2 1 3)))
           (q-2d (vt-reshape
                   (vt-contiguous q-t)
                   (list (* batch nh) seq-q hd)))
           (k-2d (vt-reshape
                   (vt-contiguous k-t)
                   (list (* batch nh) seq-k hd)))
           (v-2d (vt-reshape
                   (vt-contiguous v-t)
                   (list (* batch nh) seq-v hd))))
      (multiple-value-bind
          (attn-out attn-w)
          (sdpa-forward q-2d k-2d v-2d nil
                        (mha-dropout-rate l))
        (let* ((attn-4d (vt-reshape
                          attn-out
                          (list batch nh seq-q hd)))
               (attn-transposed
                 (vt-transpose attn-4d '(0 2 1 3)))
               (attn-merged
                 (vt-reshape
                   (vt-contiguous attn-transposed)
                   (list batch seq-q ed)))
               (output
                 (if (mha-use-bias-p l)
                     (vt-+ (vt-matmul attn-merged
                                     (mha-wo l))
                           (mha-bo l))
                     (vt-matmul attn-merged
                               (mha-wo l)))))
          (setf (mha-cache l)
                (list :query query :key key
                      :value value
                      :q-2d q-2d :k-2d k-2d
                      :v-2d v-2d
                      :attn-w attn-w
                      :attn-merged attn-merged
                      :batch batch :seq-q seq-q
                      :seq-k seq-k
                      :nh nh :hd hd :ed ed))
          output)))))

(defmethod backward ((l multi-head-attention) grad-output)
  "完整 MHA 反向 (修复: 返回 dQ, dK, dV 三元组)."
  (let* ((cache (mha-cache l))
         (query (getf cache :query))
         (key (getf cache :key))
         (value (getf cache :value))
         (q-2d (getf cache :q-2d))
         (k-2d (getf cache :k-2d))
         (v-2d (getf cache :v-2d))
         (attn-w (getf cache :attn-w))
         (attn-merged (getf cache :attn-merged))
         (batch (getf cache :batch))
         (seq-q (getf cache :seq-q))
         (seq-k (getf cache :seq-k))
         (nh (getf cache :nh))
         (hd (getf cache :hd))
         (ed (getf cache :ed))
         (use-bias (mha-use-bias-p l))
         (d-attn-merged
           (vt-matmul grad-output
                      (vt-transpose (mha-wo l))))
         (d-wo (vt-matmul
                 (vt-transpose attn-merged)
                 grad-output))
         (d-bo (when use-bias
                 (vt-sum grad-output :axis 0)))
         (d-attn-4d
           (vt-reshape d-attn-merged
                       (list batch seq-q nh hd)))
         (d-attn-transposed
           (vt-transpose d-attn-4d '(0 2 1 3)))
         (d-attn-out
           (vt-reshape
             (vt-contiguous d-attn-transposed)
             (list (* batch nh) seq-q hd))))
    (multiple-value-bind
        (dq-2d dk-2d dv-2d)
        (sdpa-backward d-attn-out q-2d k-2d v-2d attn-w)
      (flet ((merge-heads (d-2d seq-len)
               (let* ((d-4d (vt-reshape
                             d-2d
                             (list batch nh seq-len hd)))
                      (d-t (vt-transpose d-4d '(0 2 1 3)))
                      (d-merged
                        (vt-reshape
                          (vt-contiguous d-t)
                          (list batch seq-len ed))))
                 d-merged)))
        (let* ((d-q-merged (merge-heads dq-2d seq-q))
               (d-k-merged (merge-heads dk-2d seq-k))
               (d-v-merged (merge-heads dv-2d seq-k))
               ;; 投影反向
               (d-query
                 (vt-matmul d-q-merged
                            (vt-transpose (mha-wq l))))
               (d-wq (vt-matmul
                       (vt-transpose query)
                       d-q-merged))
               (d-bq (when use-bias
                       (vt-sum d-q-merged :axis 0)))
               (d-key
                 (vt-matmul d-k-merged
                            (vt-transpose (mha-wk l))))
               (d-wk (vt-matmul
                       (vt-transpose key)
                       d-k-merged))
               (d-bk (when use-bias
                       (vt-sum d-k-merged :axis 0)))
               (d-value
                 (vt-matmul d-v-merged
                            (vt-transpose (mha-wv l))))
               (d-wv (vt-matmul
                       (vt-transpose value)
                       d-v-merged))
               (d-bv (when use-bias
                       (vt-sum d-v-merged :axis 0))))
          ;; 记录参数梯度
          (setf (mha-dwq l) d-wq
                (mha-dwk l) d-wk
                (mha-dwv l) d-wv
                (mha-dwo l) d-wo)
          (when use-bias
            (setf (mha-dbq l) d-bq
                  (mha-dbk l) d-bk
                  (mha-dbv l) d-bv
                  (mha-dbo l) d-bo))
          (values d-query d-key d-value))))))

(defmethod params ((l multi-head-attention))
  (let ((r (list
            (list "w_q" (mha-wq l)
                  #'(lambda (v) (setf (mha-wq l) v)))
            (list "w_k" (mha-wk l)
                  #'(lambda (v) (setf (mha-wk l) v)))
            (list "w_v" (mha-wv l)
                  #'(lambda (v) (setf (mha-wv l) v)))
            (list "w_o" (mha-wo l)
                  #'(lambda (v) (setf (mha-wo l) v))))))
    (when (mha-use-bias-p l)
      (setf r (nconc r (list
        (list "b_q" (mha-bq l)
              #'(lambda (v) (setf (mha-bq l) v)))
        (list "b_k" (mha-bk l)
              #'(lambda (v) (setf (mha-bk l) v)))
        (list "b_v" (mha-bv l)
              #'(lambda (v) (setf (mha-bv l) v)))
        (list "b_o" (mha-bo l)
              #'(lambda (v) (setf (mha-bo l) v)))))))
    r))

(defmethod grads ((l multi-head-attention))
  (let ((r (list
            (cons "w_q" (mha-dwq l))
            (cons "w_k" (mha-dwk l))
            (cons "w_v" (mha-dwv l))
            (cons "w_o" (mha-dwo l)))))
    (when (mha-use-bias-p l)
      (setf r (nconc r (list
        (cons "b_q" (mha-dbq l))
        (cons "b_k" (mha-dbk l))
        (cons "b_v" (mha-dbv l))
        (cons "b_o" (mha-dbo l))))))
    r))


(defclass transformer-block (layer)
  ((embed-dim :initarg :embed-dim :reader tb-embed-dim)
   (num-heads :initarg :num-heads :reader tb-num-heads)
   (ffn-dim :initarg :ffn-dim :initform nil
    :reader tb-ffn-dim)
   (dropout-rate :initarg :dropout-rate :initform 0.1d0
    :reader tb-dropout-rate)
   (eps :initarg :eps :initform 1.0d-5 :reader tb-eps)
   (mha :accessor tb-mha)
   (ffn-dense1 :accessor tb-ffn1)
   (ffn-dense2 :accessor tb-ffn2)
   (ln1 :accessor tb-ln1) (ln2 :accessor tb-ln2)
   (drop1 :accessor tb-drop1)
   (drop2 :accessor tb-drop2))
  (:documentation "Pre-Norm Transformer Block."))

(defun make-transformer-block
    (embed-dim num-heads &key ffn-dim dropout-rate eps
     (name "transformer-block") (trainable t))
  (make-instance 'transformer-block
    :embed-dim embed-dim :num-heads num-heads
    :ffn-dim ffn-dim
    :dropout-rate (or dropout-rate 0.1d0)
    :eps (or eps 1.0d-5)
    :name name :trainable trainable))

(defmethod initialize-instance :after
    ((l transformer-block) &key)
  (let* ((ed (tb-embed-dim l))
         (nh (tb-num-heads l))
         (fd (or (tb-ffn-dim l) (* 4 ed)))
         (dr (tb-dropout-rate l))
         (ep (tb-eps l)))
    (setf (tb-mha l)
          (make-multi-head-attention
            ed nh :dropout-rate dr))
    (setf (tb-ffn1 l)
          (make-dense fd :activation :gelu :name "ffn1"))
    (setf (tb-ffn2 l)
          (make-dense ed :activation :none :name "ffn2"))
    (setf (tb-ln1 l)
          (make-layer-norm (list ed) :eps ep))
    (setf (tb-ln2 l)
          (make-layer-norm (list ed) :eps ep))
    (setf (tb-drop1 l) (make-dropout dr))
    (setf (tb-drop2 l) (make-dropout dr))))

(defmethod forward ((l transformer-block) x)
  (let* ((normed1 (forward (tb-ln1 l) x))
         (attn-out (forward (tb-mha l)
                            (list normed1 normed1 normed1)))
         (dropped1 (forward (tb-drop1 l) attn-out))
         (x1 (vt-+ x dropped1))
         (normed2 (forward (tb-ln2 l) x1))
         (ffn-out (forward (tb-ffn1 l) normed2))
         (ffn-out2 (forward (tb-ffn2 l) ffn-out))
         (dropped2 (forward (tb-drop2 l) ffn-out2)))
    (vt-+ x1 dropped2)))

(defmethod backward ((l transformer-block) grad-output)
  (let* ((d-ffn-out2 (backward (tb-drop2 l) grad-output))
         (d-ffn-out (backward (tb-ffn2 l) d-ffn-out2))
         (d-normed2 (backward (tb-ffn1 l) d-ffn-out))
         (d-x1-from-ln2 (backward (tb-ln2 l) d-normed2))
         (dx1-total (vt-+ grad-output d-x1-from-ln2))
         (d-attn-out (backward (tb-drop1 l) dx1-total)))
    (multiple-value-bind (d-q d-k d-v) 
        (backward (tb-mha l) d-attn-out)
      (let* ((d-normed1 (vt-+ d-q (vt-+ d-k d-v)))
             (d-x-from-ln1 (backward (tb-ln1 l) d-normed1))
             (d-x-total (vt-+ dx1-total d-x-from-ln1)))
        d-x-total))))


(defmethod params ((l transformer-block))
  (append (params (tb-mha l))
          (params (tb-ffn1 l))
          (params (tb-ffn2 l))
          (params (tb-ln1 l))
          (params (tb-ln2 l))))

(defmethod grads ((l transformer-block))
  (append (grads (tb-mha l))
          (grads (tb-ffn1 l))
          (grads (tb-ffn2 l))
          (grads (tb-ln1 l))
          (grads (tb-ln2 l))))

(defmethod set-training! ((l transformer-block) mode)
  (call-next-method)
  (dolist (sub (list (tb-mha l) (tb-ffn1 l) (tb-ffn2 l)
                     (tb-ln1 l) (tb-ln2 l)
                     (tb-drop1 l) (tb-drop2 l)))
    (when sub (set-training! sub mode))))
