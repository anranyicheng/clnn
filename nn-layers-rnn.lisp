(in-package #:nn)

(defclass rnn-cell (layer)
  ((input-size :initarg :input-size
	       :initarg :input-size
	       :reader rnn-input-size)
   (hidden-size :initarg :hidden-size
		:initarg :hidden-size
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
	    :accessor rnn-h-cache))
  (:documentation
   "h_t = act(x_t @ W_ih.T + h_{t-1} @ W_hh.T + b_ih)"))

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
  (ensure-rnn-cell-params l)
  (let* ((x (if (= (length (vt-shape input)) 1)
                (vt-reshape input (list 1 (rnn-input-size l)))
                input))
         (batch (first (vt-shape x)))
         (h-prev (vt-zeros (list batch (rnn-hidden-size l))))
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
    (setf (rnn-dwih l)
          (if (rnn-dwih l) (vt-+ (rnn-dwih l) dwih) dwih))
    (setf (rnn-dwhh l)
          (if (rnn-dwhh l) (vt-+ (rnn-dwhh l) dwhh) dwhh))
    (setf (rnn-dbih l)
          (if (rnn-dbih l) (vt-+ (rnn-dbih l) dbih) dbih))
    (values dx dh-prev)))

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
	  :accessor lstm-cache))
  (:documentation "LSTM"))

(defun make-lstm (input-size hidden-size
                  &key (name "lstm") (trainable t))
  (make-instance 'lstm
		 :input-size input-size
		 :hidden-size hidden-size
		 :name name :trainable trainable))

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
        (dotimes (i hs)
          (setf (vt-ref b-ih (+ hs i)) 1.0d0))
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
         (h (vt-zeros (list batch hs)))
         (c (vt-zeros (list batch hs)))
         (output (vt-zeros (list batch seq-len hs)))
         (all-gates '()) (all-c '())
         (all-h '()) (all-x '()))
    (dotimes (i seq-len)
      (let* ((x-t (vt-slice input :all i :all))
             (gates
               (vt-+ (vt-+
                      (vt-matmul x-t (vt-transpose wih))
                      (vt-matmul h (vt-transpose whh)))
                     (vt-+ bih bhh)))
             (i-gate (vt-sigmoid
                      (vt-slice gates :all (list 0 hs))))
             (f-gate (vt-sigmoid
                      (vt-slice gates :all `(,hs ,(* 2 hs)))))
             (g-gate (vt-tanh
                      (vt-slice gates
                                :all `(,(* 2 hs) ,(* 3 hs)))))
             (o-gate (vt-sigmoid
                      (vt-slice gates
                                :all `(,(* 3 hs) ,(* 4 hs)))))
             (c-new (vt-+ (vt-* f-gate c)
                          (vt-* i-gate g-gate)))
             (h-new (vt-* o-gate (vt-tanh c-new))))
        (setf (vt-slice output :all i :all) h-new)
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
         (dwih-acc (vt-zeros (list (* 4 hs) is)))
         (dwhh-acc (vt-zeros (list (* 4 hs) hs)))
         (dbih-acc (vt-zeros (list (* 4 hs))))
         (dbhh-acc (vt-zeros (list (* 4 hs))))
         (dh-next (vt-zeros (list batch hs)))
         (dc-next (vt-zeros (list batch hs)))
         (grad-input (vt-zeros (list batch seq-len is)))
         (zero-h (vt-zeros (list batch hs))))
    (dotimes (i seq-len)
      (let* ((idx (- seq-len i 1))
             (gates (nth idx all-gates))
             (i-gate (first gates))
             (f-gate (second gates))
             (g-gate (third gates))
             (o-gate (fourth gates))
             (c-prev (if (= idx 0)
                         zero-h
                         (nth (1- idx) all-c)))
             (c-cur (nth idx all-c))
             (x-t (nth idx all-x))
             (dh (vt-+ (vt-slice grad-output :all idx :all)
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
        ;; 时间步内累加
        (setf dwih-acc
              (vt-+ dwih-acc
                    (vt-matmul (vt-transpose d-gates) x-t)))
        (let ((h-prev (if (= idx 0)
                          zero-h
                          (nth (1- idx) all-h))))
          (setf dwhh-acc
                (vt-+ dwhh-acc
                      (vt-matmul (vt-transpose d-gates) h-prev))))
        (setf dbih-acc
              (vt-+ dbih-acc (vt-sum d-gates :axis 0)))
        (setf dbhh-acc
              (vt-+ dbhh-acc (vt-sum d-gates :axis 0)))
        (setf dh-next (vt-matmul d-gates whh))
        (setf (vt-slice grad-input :all idx :all)
              (vt-matmul d-gates wih))
        (setf dc-next (vt-* dc f-gate))))
    (setf (lstm-dweight-ih l)
          (vt-+ (or (lstm-dweight-ih l)
                    (vt-zeros (vt-shape dwih-acc)))
                dwih-acc))
    (setf (lstm-dweight-hh l)
          (vt-+ (or (lstm-dweight-hh l)
                    (vt-zeros (vt-shape dwhh-acc)))
                dwhh-acc))
    (setf (lstm-dbias-ih l)
          (vt-+ (or (lstm-dbias-ih l)
                    (vt-zeros (vt-shape dbih-acc)))
                dbih-acc))
    (setf (lstm-dbias-hh l)
          (vt-+ (or (lstm-dbias-hh l)
                    (vt-zeros (vt-shape dbhh-acc)))
                dbhh-acc))
    grad-input))

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
                         :all `(,(* 2 hs) ,(* 3 hs))))
         (b-in (vt-slice bih
                         :all `(,(* 2 hs) ,(* 3 hs))))
         (w-hn (vt-slice whh
                         :all `(,(* 2 hs) ,(* 3 hs))))
         (b-hn (vt-slice bhh
                         :all `(,(* 2 hs) ,(* 3 hs))))
         (h (vt-zeros (list batch hs)))
         (output (vt-zeros (list batch seq-len hs)))
         (all-r '()) (all-z '()) (all-n '())
         (all-h '()) (all-x '()) (all-hn-gate '()))
    (dotimes (i seq-len)
      (let* ((x-t (vt-slice input :all i :all))
             (gates
               (vt-+ (vt-+
                      (vt-matmul x-t (vt-transpose wih))
                      (vt-matmul h (vt-transpose whh)))
                     (vt-+ bih bhh)))
             (r (vt-sigmoid
                 (vt-slice gates :all (list 0 hs))))
             (z (vt-sigmoid
                 (vt-slice gates :all `(,hs ,(* 2 hs)))))
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
        (setf (vt-slice output :all i :all) h-new)
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
         (w-in (vt-slice wih :all `(,(* 2 hs) ,(* 3 hs))))
         (w-hn (vt-slice whh :all `(,(* 2 hs) ,(* 3 hs))))
         (dwih-acc (vt-zeros (list (* 3 hs) is)))
         (dwhh-acc (vt-zeros (list (* 3 hs) hs)))
         (dbih-acc (vt-zeros (list (* 3 hs))))
         (dbhh-acc (vt-zeros (list (* 3 hs))))
         (dh-next (vt-zeros (list batch hs)))
         (grad-input (vt-zeros (list batch seq-len is)))
         (zero-h (vt-zeros (list batch hs))))
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
             (dh (vt-+ (vt-slice grad-output :all idx :all)
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
        (let ((full-dw (vt-zeros (list (* 3 hs) is))))
          (setf (vt-slice full-dw :all `(0 ,(* 2 hs)) :all)
                (vt-matmul (vt-transpose d-gates-rz) x-t))
          (setf (vt-slice full-dw
                          :all `(,(* 2 hs) ,(* 3 hs)) :all)
                (vt-matmul (vt-transpose dn-pre) x-t))
          (setf dwih-acc (vt-+ dwih-acc full-dw)))
        ;; 2. Whh
        (let ((full-dwh (vt-zeros (list (* 3 hs) hs))))
          (setf (vt-slice full-dwh :all `(0 ,(* 2 hs)) :all)
                (vt-matmul (vt-transpose d-gates-rz) h-prev))
          (setf (vt-slice full-dwh
                          :all `(,(* 2 hs) ,(* 3 hs)) :all)
                (vt-matmul (vt-transpose dhn-linear) h-prev))
          (setf dwhh-acc (vt-+ dwhh-acc full-dwh)))
        ;; 3. bih
        (let ((full-db (vt-zeros (list (* 3 hs)))))
          (setf (vt-slice full-db :all `(0 ,(* 2 hs)))
                (vt-sum d-gates-rz :axis 0))
          (setf (vt-slice full-db
                          :all `(,(* 2 hs) ,(* 3 hs)))
                (vt-sum dn-pre :axis 0))
          (setf dbih-acc (vt-+ dbih-acc full-db)))
        ;; 4. bhh
        (let ((full-dbh (vt-zeros (list (* 3 hs)))))
          (setf (vt-slice full-dbh :all `(0 ,(* 2 hs)))
                (vt-sum d-gates-rz :axis 0))
          (setf (vt-slice full-dbh
                          :all `(,(* 2 hs) ,(* 3 hs)))
                (vt-sum dhn-linear :axis 0))
          (setf dbhh-acc (vt-+ dbhh-acc full-dbh)))
        ;; 5. dX
        (let* ((wih-rz (vt-slice wih :all `(0 ,(* 2 hs))))
               (dx-t (vt-+ (vt-matmul d-gates-rz wih-rz)
                           (vt-matmul dn-pre w-in))))
          (setf (vt-slice grad-input :all idx :all) dx-t))
        ;; 6. dh_prev
        (let* ((whh-rz (vt-slice whh :all `(0 ,(* 2 hs))))
               (dh-from-rz (vt-matmul d-gates-rz whh-rz))
               (dh-from-n (vt-matmul dhn-linear w-hn))
               (dh-from-z (vt-* z-t dh)))
          (setf dh-next (vt-+ (vt-+ dh-from-rz dh-from-n)
                              dh-from-z)))))
    (setf (gru-dweight-ih l)
          (vt-+ (or (gru-dweight-ih l)
                    (vt-zeros (vt-shape dwih-acc)))
                dwih-acc))
    (setf (gru-dweight-hh l)
          (vt-+ (or (gru-dweight-hh l)
                    (vt-zeros (vt-shape dwhh-acc)))
                dwhh-acc))
    (setf (gru-dbias-ih l)
          (vt-+ (or (gru-dbias-ih l)
                    (vt-zeros (vt-shape dbih-acc)))
                dbih-acc))
    (setf (gru-dbias-hh l)
          (vt-+ (or (gru-dbias-hh l)
                    (vt-zeros (vt-shape dbhh-acc)))
                dbhh-acc))
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


































