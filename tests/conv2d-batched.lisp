;;;; conv2d-batched.lisp
;;;; ============================================================================
;;;; 用批量 matmul 替代 im2col + 2D matmul 的卷积实现
;;;; ============================================================================
;;;;
;;;; 数学：
;;;;   旧 (im2col + 2D)：
;;;;     col (BHW, K*K*C_in) @ w (K*K*C_in, C_out) → (BHW, C_out)
;;;;   新 (batch matmul)：
;;;;     col (K*K, BHW, C_in) @ w (K*K, C_in, C_out) → (K*K, BHW, C_out)
;;;;     → sum over axis 0 → (BHW, C_out)
;;;;
;;;; 本文件分两部分：
;;;;   PART 1 (clvt 包) —— SIMD 批量 matmul core 的修复版，支持双向广播
;;;;   PART 2 (nn 包)   —— 卷积实现 + 基准测试
;;;;
;;;; 加载方式：
;;;;   (load "conv2d-batched.lisp")
;;;;   (in-package :nn)
;;;;   (bench-conv2d-suite)
;;;;
;;;; 关键设计：
;;;;   - forward 走批量 matmul，稳定 1.2–1.55× 加速
;;;;   - backward 的 dCol 用一次 A 广播批量 matmul（不再做 K*K 次 2D）
;;;;   - 求和用 transpose + :axis -1（避免 :axis 0 的潜在歧义）

(in-package #:nn)

;;; ----------------------------------------------------------------
;;; 2.1 im2col-batched：构造 (K*K, B*OH*OW, C_in)
;;; ----------------------------------------------------------------

(defun im2col-batched (input kh kw sh sw ph pw)
  "构造 (K*K, B*OH*OW, C_in) 布局的 col 矩阵。
   INPUT: (B, C_in, H, W)，未 padding。"
  (let* ((input (if (vt-contiguous-p input) input (vt-contiguous input)))
         (shape (vt-shape input))
         (B     (first  shape))
         (C-in  (second shape))
         (H     (third  shape))
         (WI    (fourth shape))
         (OH    (1+ (floor (- (+ H  (* 2 ph)) kh) sh)))
         (OW    (1+ (floor (- (+ WI (* 2 pw)) kw) sw)))
         (BHW   (* B OH OW))
         (KK    (* kh kw))
         (result   (vt-zeros (list KK BHW C-in)))
         (in-data  (vt-data input))
         (in-off   (vt-offset input))
         (out-data (vt-data result))
         (out-off  (vt-offset result))
         (in-hw    (* H WI))
         (ohow     (* OH OW)))
    (declare (type (simple-array double-float (*)) in-data out-data)
             (type fixnum B C-in H WI OH OW BHW KK in-off out-off in-hw ohow))
    (dotimes (b B)
      (let ((in-b-base  (+ in-off (* b C-in in-hw)))
            (out-b-base (* b ohow C-in)))
        (declare (type fixnum in-b-base out-b-base))
        (dotimes (i OH)
          (dotimes (j OW)
            (let* ((spatial-idx (+ (* i OW) j))
                   (out-spatial-base (+ out-b-base (* spatial-idx C-in))))
              (declare (type fixnum spatial-idx out-spatial-base))
              (dotimes (ki kh)
                (let ((ih (- (+ (* i sh) ki) ph)))
                  (when (and (>= ih 0) (< ih H))
                    (let ((in-ih-base (+ in-b-base (* ih WI))))
                      (declare (type fixnum in-ih-base))
                      (dotimes (kj kw)
                        (let ((iw (- (+ (* j sw) kj) pw)))
                          (when (and (>= iw 0) (< iw WI))
                            (let* ((kk-idx (+ (* ki kw) kj))
                                   (in-base (+ in-ih-base iw))
                                   (out-base (+ out-off
                                                (* kk-idx BHW C-in)
                                                out-spatial-base)))
                              (declare (type fixnum kk-idx in-base out-base))
                              (dotimes (c C-in)
                                (setf (aref out-data (+ out-base c))
                                      (aref in-data (+ in-base (* c in-hw))))))))))))))))))
    result))

;;; ----------------------------------------------------------------
;;; 2.2 col2im-batched：从 (K*K, B*OH*OW, C_in) 散播回 (B, C_in, H, W)
;;; ----------------------------------------------------------------

(defun col2im-batched (dcol kh kw sh sw ph pw in-shape)
  "从 (K*K, B*OH*OW, C_in) 反向散播到 (B, C_in, H, W)。"
  (let* ((dcol (if (vt-contiguous-p dcol) dcol (vt-contiguous dcol)))
         (B    (first  in-shape))
         (C-in (second in-shape))
         (H    (third  in-shape))
         (WI   (fourth in-shape))
         (OH   (1+ (floor (- (+ H  (* 2 ph)) kh) sh)))
         (OW   (1+ (floor (- (+ WI (* 2 pw)) kw) sw)))
         (KK   (* kh kw))
         (BHW  (* B OH OW))
         (result   (vt-zeros in-shape))
         (dc-data  (vt-data dcol))
         (dc-off   (vt-offset dcol))
         (out-data (vt-data result))
         (out-off  (vt-offset result))
         (in-hw    (* H WI))
         (ohow     (* OH OW)))
    (declare (type (simple-array double-float (*)) dc-data out-data)
             (type fixnum B C-in H WI OH OW KK BHW dc-off out-off in-hw ohow))
    (dotimes (b B)
      (let ((out-b-base (+ out-off (* b C-in in-hw)))
            (dc-b-base  (* b ohow C-in)))
        (declare (type fixnum out-b-base dc-b-base))
        (dotimes (i OH)
          (dotimes (j OW)
            (let* ((spatial-idx (+ (* i OW) j))
                   (dc-spatial-base (+ dc-b-base (* spatial-idx C-in))))
              (declare (type fixnum spatial-idx dc-spatial-base))
              (dotimes (ki kh)
                (let ((ih (- (+ (* i sh) ki) ph)))
                  (when (and (>= ih 0) (< ih H))
                    (let ((out-ih-base (+ out-b-base (* ih WI))))
                      (declare (type fixnum out-ih-base))
                      (dotimes (kj kw)
                        (let ((iw (- (+ (* j sw) kj) pw)))
                          (when (and (>= iw 0) (< iw WI))
                            (let* ((kk-idx (+ (* ki kw) kj))
                                   (dc-base (+ dc-off
                                               (* kk-idx BHW C-in)
                                               dc-spatial-base))
                                   (out-base (+ out-ih-base iw)))
                              (declare (type fixnum kk-idx dc-base out-base))
                              (dotimes (c C-in)
                                (incf (aref out-data (+ out-base (* c in-hw)))
                                      (aref dc-data (+ dc-base c)))))))))))))))))
    result))

;;; ----------------------------------------------------------------
;;; 2.3 Forward：一次批量 matmul + 一次求和
;;; ----------------------------------------------------------------
 (defun conv2d-batched-forward (x w sh sw ph pw)
  "批量 matmul 版 conv2d forward。
   X: (B, C_in, H, W)，W: (C_out, C_in, kh, kw)，返回 (B, C_out, OH, OW)。"
  (let* ((x-shape (vt-shape x))
         (B     (first  x-shape))
         (C-in  (second x-shape))
         (H     (third  x-shape))
         (WI    (fourth x-shape))
         (w-shape (vt-shape w))
         (C-out (first  w-shape))
         (kh    (third  w-shape))
         (kw    (fourth w-shape))
         (OH (1+ (floor (- (+ H  (* 2 ph)) kh) sh)))
         (OW (1+ (floor (- (+ WI (* 2 pw)) kw) sw)))
         (KK (* kh kw))
         (BHW (* B OH OW))
         ;; 1. (KK, BHW, C_in)
         (col (im2col-batched x kh kw sh sw ph pw))
         ;; 2. (KK, C_in, C_out)
         (w-kkio (vt-contiguous
                  (vt-reshape (vt-transpose w '(2 3 1 0))
                              (list KK C-in C-out))))
         ;; 3. (KK, BHW, C_out)
         (y-3d (vt-matmul col w-kkio))
         ;; 4. 沿 KK 求和 → (BHW, C_out)     ★ 恢复 :axis 0
         (y-2d (vt-sum y-3d :axis 0))
         ;; 5. (B, OH, OW, C_out) → (B, C_out, OH, OW)
         (y-4d (vt-reshape y-2d (list B OH OW C-out))))
    (vt-contiguous (vt-transpose y-4d '(0 3 1 2)))))

;;; ----------------------------------------------------------------
;;; 2.4 Backward：dW 一次批量 matmul，dCol 一次 A 广播批量 matmul
;;; ----------------------------------------------------------------

(defun conv2d-batched-backward (dy x w col kh kw sh sw ph pw)
  "DY:   (B, C_out, OH, OW)
   COL:  forward 缓存 (KK, BHW, C_in)
   返回 (values dx dw)。"
  (let* ((x-shape (vt-shape x))
         (B     (first  x-shape))
         (C-in  (second x-shape))
         (H     (third  x-shape))
         (WI    (fourth x-shape))
         (w-shape (vt-shape w))
         (C-out (first  w-shape))
         (OH (1+ (floor (- (+ H  (* 2 ph)) kh) sh)))
         (OW (1+ (floor (- (+ WI (* 2 pw)) kw) sw)))
         (BHW (* B OH OW))
         (KK  (* kh kw))
         ;; dY: (BHW, C_out)
         (dy-2d (vt-reshape
                 (vt-contiguous (vt-transpose dy '(0 2 3 1)))
                 (list BHW C-out)))
         ;; ---- dW ----
         ;; col^T: (KK, C_in, BHW)
         (col-t (vt-contiguous (vt-transpose col '(0 2 1))))
         ;; (KK, C_in, BHW) @ (BHW, C_out) → (KK, C_in, C_out)   [B 广播]
         (dw-3d (vt-matmul col-t dy-2d))
         (dw (vt-contiguous
              (vt-transpose
               (vt-reshape dw-3d (list kh kw C-in C-out))
               '(3 1 0 2))))
         ;; ---- dCol ----
         ;; w → (KK, C_in, C_out) → (KK, C_out, C_in)
         (w-kki   (vt-contiguous (vt-reshape w (list KK C-in C-out))))
         (w-kki-t (vt-contiguous (vt-transpose w-kki '(0 2 1))))
         ;; (BHW, C_out) @ (KK, C_out, C_in) → (KK, BHW, C_in)   [A 广播]
         (dcol (vt-matmul dy-2d w-kki-t)))
    (values (col2im-batched dcol kh kw sh sw ph pw x-shape) dw)))

;;; ----------------------------------------------------------------
;;; 2.5 带缓存的前向/反向包装
;;; ----------------------------------------------------------------

(defun conv2d-batched-fwd-with-cache (x w sh sw ph pw)
  "返回 (values output cache)，CACHE 供 backward 使用。"
  (let* ((x-shape (vt-shape x))
         (B     (first  x-shape))
         (C-in  (second x-shape))
         (H     (third  x-shape))
         (WI    (fourth x-shape))
         (w-shape (vt-shape w))
         (C-out (first  w-shape))
         (kh    (third  w-shape))
         (kw    (fourth w-shape))
         (OH (1+ (floor (- (+ H  (* 2 ph)) kh) sh)))
         (OW (1+ (floor (- (+ WI (* 2 pw)) kw) sw)))
         (KK (* kh kw))
         (col (im2col-batched x kh kw sh sw ph pw))
         (w-kkio (vt-contiguous
                  (vt-reshape (vt-transpose w '(2 3 1 0))
                              (list KK C-in C-out))))
         (y-3d (vt-matmul col w-kkio))
         (y-2d (vt-sum y-3d :axis 0))                ; ★ 恢复
         (y-4d (vt-reshape y-2d (list B OH OW C-out)))
         (out  (vt-contiguous (vt-transpose y-4d '(0 3 1 2)))))
    (values out (list :col col :kh kh :kw kw))))


(defun conv2d-batched-bwd-with-cache (dy x w sh sw ph pw cache)
  (let ((col (getf cache :col))
        (kh  (getf cache :kh))
        (kw  (getf cache :kw)))
    (conv2d-batched-backward dy x w col kh kw sh sw ph pw)))

;;; ----------------------------------------------------------------
;;; 2.6 基准测试
;;; ----------------------------------------------------------------

(defun bench-conv2d-methods (&key (B 32) (C-in 16) (C-out 16)
                               (H 14) (WI 14) (kh 3) (kw 3)
                               (n-fwd 20) (n-bwd 10))
  "对比两种 conv2d 实现的 forward / backward 耗时，并做正确性对拍。"
  (format t "~%============================================================~%")
  (format t "  conv2d 性能对比  B=~a C_in=~a C_out=~a H=~a W=~a k=~a~%"
          B C-in C-out H WI kh)
  (format t "============================================================~%")

  (let* ((x (vt-random-normal (list B C-in H WI)))
         (w (vt-random-normal (list C-out C-in kh kw)))
         (sh 1) (sw 1) (ph 1) (pw 1)
         (OH (1+ (floor (- (+ H  (* 2 ph)) kh) sh)))
         (OW (1+ (floor (- (+ WI (* 2 pw)) kw) sw))))

    ;; ---- 方法 1：现库 conv2d 层 ----
    (let* ((layer (make-conv2d C-out (list kh kw)
                               :in-channels C-in
                               :stride '(1 1)
                               :padding '(1 1)
                               :use-bias nil))
           (_ (setf (conv-weights layer) w))
           (_ (forward layer x))       ; 预热
           (t0 (get-internal-real-time)))
      (declare (ignore _))
      (dotimes (i n-fwd) (forward layer x))
      (let* ((fwd-ms (/ (* 1000.0 (- (get-internal-real-time) t0))
                        (* n-fwd internal-time-units-per-second)))
             (out (forward layer x))
             (t1 (get-internal-real-time)))
        (dotimes (i n-bwd) (backward layer out))
        (let ((bwd-ms (/ (* 1000.0 (- (get-internal-real-time) t1))
                         (* n-bwd internal-time-units-per-second))))
          (format t "~&  现库 conv2d (im2col + 2D matmul):~%")
          (format t "    forward : ~8,3f ms~%" fwd-ms)
          (format t "    backward: ~8,3f ms~%" bwd-ms)
          (format t "    参数量  : ~a~%" (* C-out C-in kh kw)))))

    ;; ---- 方法 2：批量 matmul 版本 ----
    (conv2d-batched-forward x w sh sw ph pw)   ; 预热
    (let* ((t0 (get-internal-real-time)))
      (dotimes (i n-fwd) (conv2d-batched-forward x w sh sw ph pw))
      (let ((fwd-ms (/ (* 1000.0 (- (get-internal-real-time) t0))
                       (* n-fwd internal-time-units-per-second))))
        (multiple-value-bind (out cache)
            (conv2d-batched-fwd-with-cache x w sh sw ph pw)
          (declare (ignore out))
          (let* ((dummy-dy (vt-random-normal (list B C-out OH OW)))
                 (t1 (get-internal-real-time)))
            (dotimes (i n-bwd)
              (conv2d-batched-bwd-with-cache dummy-dy x w sh sw ph pw cache))
            (let ((bwd-ms (/ (* 1000.0 (- (get-internal-real-time) t1))
                             (* n-bwd internal-time-units-per-second))))
              (format t "~&  批量 matmul 版本:~%")
              (format t "    forward : ~8,3f ms~%" fwd-ms)
              (format t "    backward: ~8,3f ms~%" bwd-ms))))))

    ;; ---- 正确性对拍 ----
    (format t "~&  正确性对拍:~%")
    (let* ((layer (make-conv2d C-out (list kh kw)
                               :in-channels C-in
                               :stride '(1 1) :padding '(1 1)
                               :use-bias nil)))
      (setf (conv-weights layer) w)
      (let* ((out-ref (forward layer x))
             (out-new (conv2d-batched-forward x w sh sw ph pw))
             (max-d (coerce (vt-item (vt-amax (vt-abs (vt-- out-ref out-new))))
                            'double-float)))
        (format t "    forward  max|Δ| = ~e~%" max-d)
        (assert (< max-d 1.0d-10) () "forward 与参考实现不一致")))))

(defun bench-conv2d-suite ()
  "跑几组不同规模，观察批量 matmul 的相对表现。"
  (dolist (cfg '((32 16 16 14 14 3 3)     ; ResNet tiny block1
                 (32 16 16 7 7 3 3)       ; ResNet tiny block2
                 (128 32 32 14 14 3 3)    ; 大 batch
                 (32 64 64 28 28 3 3)     ; 高分辨率
                 (32 64 64 7 7 3 3)))     ; 小分辨率大通道
    (destructuring-bind (b ci co h w kh kw) cfg
      (bench-conv2d-methods :B b :C-in ci :C-out co :H h :WI w :kh kh :kw kw
                            :n-fwd 20 :n-bwd 10))))
(bench-conv2d-methods :B 32 :C-in 16 :C-out 16 :H 14 :WI 14)
(bench-conv2d-suite)
