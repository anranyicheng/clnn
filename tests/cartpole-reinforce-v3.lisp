;;;; cartpole-reinforce-v3.lisp
;;;; CartPole + REINFORCE 完整可运行版本。
;;;;
;;;; 用法：
;;;;   (in-package :nn)
;;;;   (load "cartpole-reinforce-v3.lisp")
;;;;   (train-reinforce-v3 :episodes 500 :lr 1e-3 :hidden 32)
;;;;
;;;; 预期：500 episodes 后 avg-last-100 → 170+，测试平均回报 → 400+
;;;;
;;;; 关键修复记录（相对 v2）：
;;;;   1. 去掉 gnorm 的负号（REINFORCE 符号）
;;;;   2. run-episode 返回 rewards 列表而非 total 标量
;;;;   3. vt-concatenate 用 apply 展开 states 列表
;;;;   4. t 作变量名改为 idx / n-steps
;;;;   5. forward 一次 batch，不再循环覆盖缓存

(in-package :nn)

;;; ============================================================
;;; 1. CartPole 环境
;;; ============================================================
(defstruct cartpole
  (x 0.0d0) (x-dot 0.0d0) (theta 0.0d0) (theta-dot 0.0d0) (steps 0))

(defparameter *gravity* 9.8d0)
(defparameter *mass-cart* 1.0d0)
(defparameter *mass-pole* 0.1d0)
(defparameter *total-mass* 1.1d0)
(defparameter *length* 0.5d0)
(defparameter *polemass-length* 0.05d0)
(defparameter *force-mag* 10.0d0)
(defparameter *dt* 0.02d0)
(defparameter *x-threshold* 2.4d0)
(defparameter *theta-threshold* 0.2095d0)
(defparameter *max-steps* 500)

(defun cartpole-reset ()
  (make-cartpole
   :x         (+ -0.05d0 (* 0.1d0 (random 1.0d0)))
   :x-dot     (+ -0.05d0 (* 0.1d0 (random 1.0d0)))
   :theta     (+ -0.05d0 (* 0.1d0 (random 1.0d0)))
   :theta-dot (+ -0.05d0 (* 0.1d0 (random 1.0d0)))))

(defun cartpole-step (s action)
  "返回 (values state reward done)。"
  (let* ((force (if (= action 1) *force-mag* (- *force-mag*)))
         (cos-theta (cos (cartpole-theta s)))
         (sin-theta (sin (cartpole-theta s)))
         (temp (- *total-mass* (* *mass-pole* cos-theta cos-theta)))
         (theta-acc (/ (+ (* *gravity* sin-theta)
                          (- (* cos-theta
                                (/ (+ force
                                      (* *polemass-length*
                                         (cartpole-theta-dot s)
                                         (cartpole-theta-dot s)
                                         sin-theta))
                                   *total-mass*))))
                       (* *length* temp)))
         (x-acc (/ (+ force
                      (* *polemass-length*
                         (- (* (cartpole-theta-dot s)
                               (cartpole-theta-dot s) sin-theta)
                            (* theta-acc cos-theta))))
                   *total-mass*)))
    (incf (cartpole-x s) (* *dt* (cartpole-x-dot s)))
    (incf (cartpole-x-dot s) (* *dt* x-acc))
    (incf (cartpole-theta s) (* *dt* (cartpole-theta-dot s)))
    (incf (cartpole-theta-dot s) (* *dt* theta-acc))
    (incf (cartpole-steps s))
    (let ((done (or (> (abs (cartpole-x s)) *x-threshold*)
                    (> (abs (cartpole-theta s)) *theta-threshold*)
                    (>= (cartpole-steps s) *max-steps*))))
      (values s 1.0d0 done))))

(defun cartpole-observe (s)
  (vt-reshape
   (vt-from-sequence
    (list (cartpole-x s) (cartpole-x-dot s)
          (cartpole-theta s) (cartpole-theta-dot s))
    :dtype :float64)
   '(1 4)))

;;; ============================================================
;;; 2. 动作采样
;;; ============================================================
(defun sample-action (logits)
  (let* ((probs (vt-softmax logits))
         (p0 (vt-ref probs 0 0)))
    (if (< (random 1.0d0) p0) 0 1)))

(defun greedy-action (logits)
  (let ((probs (vt-to-list (vt-flatten (vt-softmax logits)))))
    (if (> (first probs) (second probs)) 0 1)))

(defun run-episode (model &key greedy)
  "返回 (values states-list actions-list rewards-list)。
   states 是 (1, 4) VT 的列表；actions 是整数列表；rewards 是浮点数列表。"
  (let ((state (cartpole-reset))
        (states '())
        (actions '())
        (rewards '())
        (done nil))
    (loop while (not done) do
      (let* ((obs (cartpole-observe state))
             (logits (forward model obs))
             (action (if greedy
                         (greedy-action logits)
                         (sample-action logits))))
        (push obs states)
        (push action actions)
        (multiple-value-bind (next reward done-p) (cartpole-step state action)
          (push reward rewards)
          (setf state next done done-p))))
    (values (nreverse states) (nreverse actions) (nreverse rewards))))

;;; ============================================================
;;; 3. 折扣回报 + 标准化
;;; ============================================================
(defun compute-discounted-returns (rewards gamma)
  "REWARDS 是 list，返回 list。"
  (let* ((n (length rewards))
         (arr (coerce rewards 'vector))
         (out (make-array n :element-type 'double-float))
         (g 0.0d0))
    (loop for idx from (1- n) downto 0 do
      (setf g (+ (aref arr idx) (* gamma g)))
      (setf (aref out idx) g))
    (coerce out 'list)))

(defun standardize (lst)
  "标准化到 mean=0, std=1。返回 list。"
  (let* ((n (length lst))
         (mean (/ (reduce #'+ lst) n))
         (var (/ (reduce #'+ (mapcar (lambda (r) (expt (- r mean) 2)) lst)) n))
         (std (sqrt (+ var 1.0d-8))))
    (mapcar (lambda (r) (/ (- r mean) std)) lst)))

;;; ============================================================
;;; 4. 单次 REINFORCE 更新
;;; ============================================================
(defun train-one-episode (model opt states actions rewards gamma)
  "REINFORCE 单次更新。返回 episode 总回报。
   梯度：dL/dlogits = (softmax - one_hot) * G_norm
         （强化 G>0 的动作，抑制 G<0 的动作）"
  (let* ((n-steps (length rewards))
         (returns (compute-discounted-returns rewards gamma))
         (norm-returns (standardize returns))
         ;; 用 apply 展开 states 列表为多个参数
         (states-batch (apply #'vt-concatenate 0 states))
         (one-hot-arr (make-array (* n-steps 2)
                                  :element-type 'double-float
                                  :initial-element 0.0d0)))
    (dotimes (idx n-steps)
      (setf (aref one-hot-arr (+ (* idx 2) (nth idx actions))) 1.0d0))
    (let* ((one-hot (vt-reshape
                     (vt-from-sequence (coerce one-hot-arr 'list)
                                       :dtype :float64)
                     (list n-steps 2)))
           (logits (forward model states-batch))
           (probs (vt-softmax logits))
           ;; ★ 正号！去掉负号是关键修复
           (gnorm (vt-reshape
                   (vt-from-sequence norm-returns :dtype :float64)
                   (list n-steps 1)))
           (ce-grad (vt-- probs one-hot))
           (weighted (vt-* ce-grad gnorm)))
      (zero-grad! model)
      (backward model weighted)
      (optimizer-step opt (params model) (grads model)))
    (reduce #'+ rewards)))

;;; ============================================================
;;; 5. 主训练
;;; ============================================================
(defun train-reinforce (&key (episodes 500) (lr 1e-3) (gamma 0.99)
                                  (hidden 32))
  (format t "~%=== CartPole + REINFORCE ===~%")
  (format t "网络: Dense(4→~a) → ReLU → Dense(~a→2)~%" hidden hidden)
  (format t "episodes=~a  lr=~a  gamma=~a~%~%" episodes lr gamma)

  (let* ((model (make-sequential))
         (opt (make-adam :lr lr))
         (recent-returns '()))
    (seq-add! model (make-dense hidden :in-dim 4 :activation :relu))
    (seq-add! model (make-dense 2 :activation :none))
    ;; 触发权重初始化
    (forward model (vt-random-normal '(1 4)))

    (dotimes (ep episodes)
      (multiple-value-bind (states actions rewards) (run-episode model)
        (let ((total (train-one-episode model opt states actions rewards gamma)))
          (push total recent-returns)
          (when (zerop (mod (1+ ep) 50))
            (let* ((n (min 100 (length recent-returns)))
                   (avg (/ (reduce #'+ (subseq recent-returns 0 n)) n)))
              (format t "Episode ~3a  return=~a  avg-last-~a=~,1f~%"
                      (1+ ep) (round total) n avg))))))

    (format t "~%测试 20 回合（贪心策略）...~%")
    (let ((test-returns '()))
      (dotimes (i 20)
        (multiple-value-bind (s a rewards) (run-episode model :greedy t)
          (declare (ignore s a))
          (push (reduce #'+ rewards) test-returns)))
      (format t "平均回报: ~,1f~%" (/ (reduce #'+ test-returns) 20.0)))
    model))

(train-reinforce)
