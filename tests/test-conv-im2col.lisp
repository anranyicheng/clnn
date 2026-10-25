;;;; test-conv-im2col.lisp
;;;;
;;;; im2col / col2im / conv2d 的「与朴素参考实现逐元素对拍」测试。
;;;;
;;;; 为什么需要这个测试：im2col 与 col2im 是卷积的性能核心，为了提速做了
;;;; 「内部区快路径 + 跨边界慢路径」的分裂，以及基址预计算。这类改写最容易
;;;; 在边界（padding、stride != kernel、kernel 大于输入、非方形核）上出错，
;;;; 而梯度对拍只能发现「梯度不一致」，无法定位到具体是前向展开错了。
;;;; 因此这里用一个最直白、绝对正确的 4 重循环参考实现做逐元素比较。
;;;;
;;;; 用法:
;;;;   sbcl --dynamic-space-size 4096 --non-interactive \
;;;;        --eval '(require :asdf)' --eval '(asdf:load-system :clnn)' \
;;;;        --load tests/test-conv-im2col.lisp
;;;;
;;;; 加载即执行；存在失败项时以非 0 退出。

(require :asdf)
(asdf:load-system :clnn)

(in-package :nn)

(defparameter *pass* 0)
(defparameter *fail* 0)
(defparameter *fail-msgs* '())

(defmacro ck (name cond)
  `(if ,cond
       (progn (incf *pass*) (format t "~&  [PASS] ~a~%" ,name))
       (progn (incf *fail*)
              (push ,name *fail-msgs*)
              (format t "~&  [FAIL] ~a~%" ,name))))

;;; ---------------------------------------------------------------
;;; 参考实现：最直白的 6 重循环，逐元素带边界检查
;;; ---------------------------------------------------------------

(defun ref-im2col (x kh kw sh sw ph pw)
  "返回 (values rows cols getter)，GETTER 按 (row col) 取值。
   参考实现不追求速度，只求显然正确。"
  (let* ((shape (vt-shape x))
         (b (first shape)) (c (second shape))
         (h (third shape)) (w (fourth shape))
         (oh (1+ (floor (- (+ h (* 2 ph)) kh) sh)))
         (ow (1+ (floor (- (+ w (* 2 pw)) kw) sw)))
         (rows (* b oh ow))
         (cols (* c kh kw))
         (data (make-array (* rows cols) :element-type 'double-float
                                        :initial-element 0.0d0)))
    (dotimes (bb b)
      (dotimes (ii oh)
        (dotimes (jj ow)
          (let ((row (+ (* bb oh ow) (* ii ow) jj)))
            (dotimes (cc c)
              (dotimes (ki kh)
                (dotimes (kj kw)
                  (let ((ih (+ (* ii sh) ki (- ph)))
                        (iw (+ (* jj sw) kj (- pw))))
                    (when (and (>= ih 0) (< ih h) (>= iw 0) (< iw w))
                      (setf (aref data (+ (* row cols) (* cc kh kw) (* ki kw) kj))
                            (coerce (vt-ref x bb cc ih iw) 'double-float)))))))))))
    (values rows cols data)))

(defun ref-col2im (col kh kw sh sw ph pw in-shape)
  (let* ((b (first in-shape)) (c (second in-shape))
         (h (third in-shape)) (w (fourth in-shape))
         (oh (1+ (floor (- (+ h (* 2 ph)) kh) sh)))
         (ow (1+ (floor (- (+ w (* 2 pw)) kw) sw)))
         (cols (* c kh kw))
         (data (make-array (* b c h w) :element-type 'double-float
                                       :initial-element 0.0d0)))
    (dotimes (bb b)
      (dotimes (ii oh)
        (dotimes (jj ow)
          (let ((row (+ (* bb oh ow) (* ii ow) jj)))
            (dotimes (cc c)
              (dotimes (ki kh)
                (dotimes (kj kw)
                  (let ((ih (+ (* ii sh) ki (- ph)))
                        (iw (+ (* jj sw) kj (- pw))))
                    (when (and (>= ih 0) (< ih h) (>= iw 0) (< iw w))
                      (incf (aref data (+ (* bb c h w) (* cc h w) (* ih w) iw))
                            (coerce (vt-ref col row (+ (* cc kh kw) (* ki kw) kj))
                                    'double-float)))))))))))
    data))

(defun max-abs-diff-list (a b)
  (let ((m 0.0d0))
    (dotimes (i (length a) m)
      (setf m (max m (abs (- (aref a i) (aref b i))))))))

;;; ---------------------------------------------------------------
;;; 1. im2col 逐元素对拍
;;; ---------------------------------------------------------------

(format t "~%=== 1. im2col vs 朴素参考实现 ===~%")
;; (kh kw sh sw ph pw in-h in-w)
(dolist (cfg '((3 3 1 1 0 0 7 7)      ; 无 padding
               (3 3 1 1 1 1 7 7)      ; 同尺寸 + padding
               (3 3 2 2 1 1 8 8)      ; stride 2
               (2 2 2 2 0 0 6 6)      ; 池化型
               (1 1 1 1 0 0 5 5)      ; 1x1 卷积
               (5 5 1 1 2 2 7 7)      ; 大核 + 大 padding
               (3 2 2 1 1 0 6 8)      ; 非方形核 / 非方形 stride
               (4 4 3 3 0 0 9 9)      ; stride > 1, kernel 4
               (3 3 1 1 1 1 5 6)      ; 非方形输入
               (7 7 1 1 3 3 7 7)      ; kernel == 输入尺寸
               (2 2 1 1 0 0 3 3)))    ; 小图
  (destructuring-bind (kh kw sh sw ph pw h w) cfg
    (let* ((b 2) (c 3)
           (x (vt-random (list b c h w)))
           (col (im2col x kh kw sh sw ph pw))
           (actual (vt-to-list (vt-flatten col))))
      (multiple-value-bind (rows cols ref)
          (ref-im2col x kh kw sh sw ph pw)
        (ck (format nil "im2col k~ax~a s~a/~a p~a/~a in ~ax~a：形状 ~a×~a"
                    kh kw sh sw ph pw h w rows cols)
            (and (equal (vt-shape col) (list rows cols))
                 (< (max-abs-diff-list (coerce actual 'vector) ref) 1d-12)))))))

;;; ---------------------------------------------------------------
;;; 2. col2im 逐元素对拍（含重叠累加）
;;; ---------------------------------------------------------------

(format t "~%=== 2. col2im vs 朴素参考实现（重叠累加）===~%")
(dolist (cfg '((3 3 1 1 1 1 7 7)
               (3 3 2 2 1 1 8 8)      ; stride 2 有重叠
               (2 2 1 1 0 0 5 5)      ; 完全重叠（sum pooling 型）
               (4 4 2 2 2 2 9 9)
               (1 1 1 1 0 0 4 4)))
  (destructuring-bind (kh kw sh sw ph pw h w) cfg
    (let* ((b 2) (c 2)
           (in-shape (list b c h w))
           (oh (1+ (floor (- (+ h (* 2 ph)) kh) sh)))
           (ow (1+ (floor (- (+ w (* 2 pw)) kw) sw)))
           (col (vt-random (list (* b oh ow) (* c kh kw))))
           (dx (col2im col kh kw sh sw ph pw in-shape))
           (actual (coerce (vt-to-list (vt-flatten dx)) 'vector))
           (ref (ref-col2im col kh kw sh sw ph pw in-shape)))
      (ck (format nil "col2im k~ax~a s~a/~a p~a/~a in ~ax~a"
                  kh kw sh sw ph pw h w)
          (< (max-abs-diff-list actual ref) 1d-12)))))

;;; ---------------------------------------------------------------
;;; 3. col2im ∘ im2col 的伴随性质
;;;    <im2col(x), col> == <x, col2im(col)>   （内积伴随性）
;;; ---------------------------------------------------------------

(format t "~%=== 3. im2col / col2im 的伴随性 <Ax,c> = <x,A^T c> ===~%")
(dolist (cfg '((3 3 1 1 1 1 7 7)
               (3 3 2 2 1 1 8 8)
               (5 5 1 1 2 2 6 6)))
  (destructuring-bind (kh kw sh sw ph pw h w) cfg
    (let* ((x (vt-random (list 2 3 h w)))
           (col (im2col x kh kw sh sw ph pw))
           (r (vt-random (vt-shape col)))
           (lhs (vt-item (vt-sum (vt-* col r))))
           (rhs (vt-item (vt-sum (vt-* x (col2im r kh kw sh sw ph pw
                                               (vt-shape x)))))))
      (ck (format nil "伴随性 k~ax~a s~a p~a：~,10f vs ~,10f"
                  kh kw sh pw lhs rhs)
          (< (abs (- lhs rhs)) (* 1d-9 (max 1.0d0 (abs lhs))))))))

;;; ---------------------------------------------------------------
;;; 4. conv2d 前向 vs 直接卷积参考实现
;;; ---------------------------------------------------------------

(format t "~%=== 4. conv2d 前向 vs 直接卷积 ===~%")
(defun ref-conv2d-forward (x w bias kh kw sh sw ph pw)
  (let* ((shape (vt-shape x))
         (b (first shape)) (cin (second shape))
         (h (third shape)) (w-in (fourth shape))
         (cout (first (vt-shape w)))
         (oh (1+ (floor (- (+ h (* 2 ph)) kh) sh)))
         (ow (1+ (floor (- (+ w-in (* 2 pw)) kw) sw)))
         (out (make-array (list b cout oh ow) :element-type 'double-float
                                                 :initial-element 0.0d0)))
    (dotimes (bb b)
      (dotimes (oc cout)
        (dotimes (ii oh)
          (dotimes (jj ow)
            (let ((acc (if bias (coerce (vt-ref bias oc) 'double-float) 0.0d0)))
              (dotimes (ic cin)
                (dotimes (ki kh)
                  (dotimes (kj kw)
                    (let ((ih (+ (* ii sh) ki (- ph)))
                          (iw (+ (* jj sw) kj (- pw))))
                      (when (and (>= ih 0) (< ih h) (>= iw 0) (< iw w-in))
                        (incf acc (* (coerce (vt-ref x bb ic ih iw) 'double-float)
                                     (coerce (vt-ref w oc ic ki kj) 'double-float))))))))
              (setf (aref out bb oc ii jj) acc))))))
    out))

(dolist (cfg '((3 1 1 0) (3 1 1 1) (3 2 2 1) (1 1 1 0) (5 1 1 2)))
  (destructuring-bind (k s ph pw) cfg
    (let* ((c (make-conv2d 4 k :in-channels 3 :stride s :padding (list ph pw)))
           (x (vt-random (list 2 3 7 7))))
      (forward c x)                       ; 触发权重初始化
      (let* ((out (forward c x))
             (ref (ref-conv2d-forward x (conv-weights c) (conv-bias c)
                                      k k s s ph pw))
             (actual (coerce (vt-to-list (vt-flatten out)) 'vector))
             (refv (make-array (length actual) :element-type 'double-float)))
        (loop for i below (length refv)
              do (setf (aref refv i) (row-major-aref ref i)))
        (ck (format nil "conv2d k~a s~a p~a/~a 前向数值正确" k s ph pw)
            (< (max-abs-diff-list actual refv) 1d-10))))))

;;; ---------------------------------------------------------------
;;; 5. batch-norm 输出（非连续视图）喂给池化/卷积仍然正确
;;;     这是 max-pool2d 入口做 vt-contiguous 归一化的回归点
;;; ---------------------------------------------------------------

(format t "~%=== 5. 非连续输入（BN 输出）下的池化/卷积 ===~%")
(let* ((bn (make-batch-norm 4))
       (x (vt-random (list 2 4 6 6)))
       (pool (make-max-pool2d 2)))
  (set-global-training! t)
  (let* ((h (forward bn x))
         (out (forward pool h))
         (h-contig (vt-contiguous h))
         (out2 (forward pool h-contig)))
    (ck "max-pool 对非连续视图与连续拷贝结果一致"
        (equal (vt-to-list (vt-flatten out)) (vt-to-list (vt-flatten out2))))))

(let* ((bn (make-batch-norm 4))
       (x (vt-random (list 2 4 8 8)))
       (c (make-conv2d 3 3 :in-channels 4 :padding 1)))
  (set-global-training! t)
  (forward c x)                            ; 初始化卷积权重
  (let* ((h (forward bn x))
         (out (forward c h))
         (out2 (forward c (vt-contiguous h))))
    (ck "conv2d 对非连续视图与连续拷贝结果一致"
        (equal (vt-to-list (vt-flatten out)) (vt-to-list (vt-flatten out2))))))

;;; ---------------------------------------------------------------
;;; 6. dtype 守卫：这些核按 (simple-array double-float) 直读底层数组，
;;;    float32 必须被明确挡住而不是静默读错位模式
;;; ---------------------------------------------------------------

(format t "~%=== 6. 非 float64 输入必须被拒绝 ===~%")
(flet ((expect-error (name thunk)
         (ck name (handler-case (progn (funcall thunk) nil) (error () t)))))
  (let ((x32 (vt-astype (vt-random (list 1 2 8 8)) :float32)))
    (expect-error "im2col 拒绝 float32" (lambda () (im2col x32 3 3 1 1 1 1)))
    (expect-error "max-pool2d 拒绝 float32" (lambda () (forward (make-max-pool2d 2) x32)))
    (expect-error "avg-pool2d 拒绝 float32" (lambda () (forward (make-avg-pool2d 2) x32))))
  (expect-error "col2im 拒绝 float32"
                (lambda () (col2im (vt-astype (vt-random (list 4 8)) :float32)
                                   2 2 1 1 0 0 (list 1 2 3 3)))))

;;; ---------------------------------------------------------------
(format t "~%========================================~%")
(format t "  结果: ~a PASS / ~a FAIL~%" *pass* *fail*)
(when *fail-msgs*
  (format t "~%  失败项:~%")
  (dolist (m (reverse *fail-msgs*)) (format t "   - ~a~%" m)))
(format t "========================================~%")
(finish-output)
(when (plusp *fail*) (sb-ext:exit :code 1))
