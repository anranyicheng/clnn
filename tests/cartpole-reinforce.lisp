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

;;;; cartpole-a2c.lisp
;;;; CartPole + A2C (Advantage Actor-Critic)
;;;;
;;;; 相比 REINFORCE 的改进：
;;;;   1. 加一个 critic 网络估计 V(s)
;;;;   2. 策略梯度用 advantage A_t = G_t - V(s_t) 加权，而非原始 G_t
;;;;   3. advantage 当作常数处理（等价于 stop-gradient）
;;;;   4. 方差降低 3-5 倍，收敛更稳
;;;;
;;;; 前置：cartpole-reinforce-v3.lisp 已加载（复用环境 + 工具函数）
;;;; 用法：
;;;;   (in-package :nn)
;;;;   (load "cartpole-a2c.lisp")
;;;;   (train-a2c :episodes 500 :actor-lr 1e-3 :critic-lr 5e-3 :hidden 32)

(in-package :nn)

;;; ============================================================
;;; 模型构造器：两个独立网络（避免分叉复杂度）
;;; ============================================================
(defun make-actor (hidden)
  "策略网络：Dense(4→hidden) → ReLU → Dense(hidden→2)"
  (let ((m (make-sequential)))
    (seq-add! m (make-dense hidden :in-dim 4 :activation :relu))
    (seq-add! m (make-dense 2 :activation :none))
    m))

(defun make-critic (hidden)
  "价值网络：Dense(4→hidden) → ReLU → Dense(hidden→1)"
  (let ((m (make-sequential)))
    (seq-add! m (make-dense hidden :in-dim 4 :activation :relu))
    (seq-add! m (make-dense 1 :activation :none))
    m))

;;; ============================================================
;;; 单次 A2C 更新
;;; ============================================================
(defun train-one-episode-a2c (actor critic actor-opt critic-opt
                              states actions rewards gamma)
  "对一条轨迹做一次 A2C 更新，返回 episode 总回报。
   actor 梯度：dL/dlogits = (softmax - one_hot) * A_t
     A_t = G_norm_t - V(s_t)，当作常数（stop-gradient）
   critic 梯度：dMSE/dV = 2*(V - G_norm) / N"
  (let* ((n-steps (length rewards))
         ;; ---- 折扣回报 + 标准化 ----
         (returns (compute-discounted-returns rewards gamma))
         (returns-norm (standardize returns))
         ;; ---- 状态打包成 (T, 4) ----
         (states-batch (apply #'vt-concatenate 0 states))
         ;; ---- 一次 forward 两个网络 ----
         (logits (forward actor states-batch))     ; (T, 2)
         (values (forward critic states-batch))    ; (T, 1)
         ;; ---- 目标张量 (T, 1) ----
         (targets (vt-reshape
                   (vt-from-sequence returns-norm :dtype :float64)
                   (list n-steps 1)))
         ;; ---- Advantage = G - V （注意：此处 values 当作常数） ----
         (advantages (vt-- targets values))
         ;; ---- one-hot (T, 2) ----
         (one-hot-arr (make-array (* n-steps 2)
                                  :element-type 'double-float
                                  :initial-element 0.0d0)))
    (dotimes (idx n-steps)
      (setf (aref one-hot-arr (+ (* idx 2) (nth idx actions))) 1.0d0))
    (let* ((one-hot (vt-reshape
                     (vt-from-sequence (coerce one-hot-arr 'list)
                                       :dtype :float64)
                     (list n-steps 2)))
           (probs (vt-softmax logits))
           ;; ---- actor 梯度： (probs - one_hot) * A_t ----
           (policy-grad (vt-- probs one-hot))
           (weighted-policy (vt-* policy-grad advantages))
           ;; ---- critic 梯度： 2*(V - G)/N ----
           (value-grad (vt-scale (vt-- values targets)
                                 (/ 2.0d0 n-steps))))
      ;; ---- 更新 actor ----
      (zero-grad! actor)
      (backward actor weighted-policy)
      (optimizer-step actor-opt (params actor) (grads actor))
      ;; ---- 更新 critic ----
      (zero-grad! critic)
      (backward critic value-grad)
      (optimizer-step critic-opt (params critic) (grads critic)))
    (reduce #'+ rewards)))

;;; ============================================================
;;; 主训练
;;; ============================================================
(defun train-a2c (&key (episodes 500) (actor-lr 1e-3) (critic-lr 5e-3)
                    (gamma 0.99) (hidden 32))
  (format t "~%=== CartPole + A2C ===~%")
  (format t "actor:  Dense(4→~a) → ReLU → Dense(~a→2)~%" hidden hidden)
  (format t "critic: Dense(4→~a) → ReLU → Dense(~a→1)~%" hidden hidden)
  (format t "episodes=~a  actor-lr=~a  critic-lr=~a  gamma=~a~%~%"
          episodes actor-lr critic-lr gamma)

  (let* ((actor (make-actor hidden))
         (critic (make-critic hidden))
         (actor-opt (make-adam :lr actor-lr))
         (critic-opt (make-adam :lr critic-lr))
         (recent-returns '()))
    ;; 触发权重初始化
    (forward actor (vt-random-normal '(1 4)))
    (forward critic (vt-random-normal '(1 4)))

    (dotimes (ep episodes)
      (multiple-value-bind (states actions rewards) (run-episode actor)
        (let ((total (train-one-episode-a2c actor critic actor-opt critic-opt
                                            states actions rewards gamma)))
          (push total recent-returns)
          (when (zerop (mod (1+ ep) 50))
            (let* ((n (min 100 (length recent-returns)))
                   (avg (/ (reduce #'+ (subseq recent-returns 0 n)) n)))
              (format t "Episode ~3a  return=~a  avg-last-~a=~,1f~%"
                      (1+ ep) (round total) n avg))))))

    (format t "~%测试 20 回合（贪心策略）...~%")
    (let ((test-returns '()))
      (dotimes (i 20)
        (multiple-value-bind (s a rewards) (run-episode actor :greedy t)
          (declare (ignore s a))
          (push (reduce #'+ rewards) test-returns)))
      (format t "平均回报: ~,1f~%" (/ (reduce #'+ test-returns) 20.0)))
    actor))

;; (train-a2c :episodes 500 :actor-lr 1e-3 :critic-lr 5e-3 :hidden 32)





;;;; cartpole-dqn.lisp
;;;; CartPole + DQN，修正超参后的版本。
;;;;
;;;; 前置：cartpole-reinforce-v3.lisp 已加载（复用环境）
;;;; 用法：
;;;;   (in-package :nn)
;;;;   (load "cartpole-dqn.lisp")
;;;;   (train-dqn)

(in-package :nn)

;;; ============================================================
;;; 1. Replay Buffer（保持原样，实现正确）
;;; ============================================================
(defstruct (replay-buffer (:constructor %make-replay-buffer))
  (capacity 0 :type fixnum)
  (size 0 :type fixnum)
  (pos 0 :type fixnum)
  (states nil)
  (actions nil)
  (rewards nil)
  (next-states nil)
  (dones nil))

(defun make-replay-buffer (capacity &key (state-dim 4))
  (%make-replay-buffer
   :capacity capacity :size 0 :pos 0
   :states (make-array (* capacity state-dim)
                       :element-type 'double-float :initial-element 0.0d0)
   :actions (make-array capacity :element-type 'fixnum :initial-element 0)
   :rewards (make-array capacity :element-type 'double-float :initial-element 0.0d0)
   :next-states (make-array (* capacity state-dim)
                            :element-type 'double-float :initial-element 0.0d0)
   :dones (make-array capacity :element-type 'fixnum :initial-element 0)))

(defun buffer-add! (buf state action reward next-state done &key (state-dim 4))
  (let* ((pos (replay-buffer-pos buf))
         (state-list (if (vt-p state) (vt-to-list (vt-flatten state)) state))
         (next-list (if (vt-p next-state) (vt-to-list (vt-flatten next-state)) next-state)))
    (dotimes (i state-dim)
      (setf (aref (replay-buffer-states buf) (+ (* pos state-dim) i))
            (coerce (nth i state-list) 'double-float))
      (setf (aref (replay-buffer-next-states buf) (+ (* pos state-dim) i))
            (coerce (nth i next-list) 'double-float)))
    (setf (aref (replay-buffer-actions buf) pos) action)
    (setf (aref (replay-buffer-rewards buf) pos) (coerce reward 'double-float))
    (setf (aref (replay-buffer-dones buf) pos) (if done 1 0))
    (setf (replay-buffer-pos buf) (mod (1+ pos) (replay-buffer-capacity buf)))
    (setf (replay-buffer-size buf)
          (min (1+ (replay-buffer-size buf)) (replay-buffer-capacity buf)))))

(defun buffer-sample (buf n &key (state-dim 4))
  (let* ((size (replay-buffer-size buf))
         (state-arr (make-array (* n state-dim) :element-type 'double-float))
         (next-arr (make-array (* n state-dim) :element-type 'double-float))
         (actions (make-array n :element-type 'fixnum))
         (rewards (make-array n :element-type 'double-float))
         (dones (make-array n :element-type 'fixnum)))
    (dotimes (k n)
      (let ((idx (random size)))
        (dotimes (i state-dim)
          (setf (aref state-arr (+ (* k state-dim) i))
                (aref (replay-buffer-states buf) (+ (* idx state-dim) i)))
          (setf (aref next-arr (+ (* k state-dim) i))
                (aref (replay-buffer-next-states buf) (+ (* idx state-dim) i))))
        (setf (aref actions k) (aref (replay-buffer-actions buf) idx))
        (setf (aref rewards k) (aref (replay-buffer-rewards buf) idx))
        (setf (aref dones k) (aref (replay-buffer-dones buf) idx))))
    (values
     (vt-reshape (vt-from-sequence (coerce state-arr 'list) :dtype :float64)
                 (list n state-dim))
     actions rewards
     (vt-reshape (vt-from-sequence (coerce next-arr 'list) :dtype :float64)
                 (list n state-dim))
     dones)))

;;; ============================================================
;;; 2. Q 网络
;;; ============================================================
(defun make-q-net (hidden)
  (let ((m (make-sequential)))
    (seq-add! m (make-dense hidden :in-dim 4 :activation :relu))
    (seq-add! m (make-dense 2 :activation :none))
    m))

(defun q-greedy-action (q-net state-obs)
  (let* ((q (forward q-net state-obs))
         (q-list (vt-to-list (vt-flatten q))))
    (if (> (first q-list) (second q-list)) 0 1)))

(defun q-epsilon-greedy (q-net state-obs epsilon)
  (if (< (random 1.0d0) epsilon)
      (random 2)
      (q-greedy-action q-net state-obs)))

;;; ============================================================
;;; 3. 单次 DQN 更新（返回 batch loss 供监控）
;;; ============================================================
(defun train-one-batch-dqn (q-net target-net q-opt buf batch-size gamma
                            &key (state-dim 4))
  (multiple-value-bind (states actions rewards next-states dones)
      (buffer-sample buf batch-size :state-dim state-dim)
    ;; ---- target: r + γ * max_a Q_target(s') * (1 - done) ----
    (let* ((q-next (forward target-net next-states))
           (q-next-data (vt-data q-next))
           (q-next-off (vt-offset q-next))
           (q-next-rs (first (vt-strides q-next)))
           (q-next-cs (second (vt-strides q-next)))
           (target-arr (make-array batch-size :element-type 'double-float)))
      (dotimes (k batch-size)
        (let* ((base (+ q-next-off (* k q-next-rs)))
               (q0 (aref q-next-data base))
               (q1 (aref q-next-data (+ base q-next-cs)))
               (max-q (max q0 q1)))
          (setf (aref target-arr k)
                (+ (aref rewards k)
                   (if (= (aref dones k) 1) 0.0d0 (* gamma max-q))))))
      ;; ---- Q(s, ·)，构造 grad-output（只对 action 位置有梯度）----
      (let* ((q-current (forward q-net states))
             (q-data (vt-data q-current))
             (q-off (vt-offset q-current))
             (q-rs (first (vt-strides q-current)))
             (q-cs (second (vt-strides q-current)))
             (grad-arr (make-array (* batch-size 2)
                                   :element-type 'double-float
                                   :initial-element 0.0d0))
             (loss-sum 0.0d0))
        (dotimes (k batch-size)
          (let* ((base (+ q-off (* k q-rs)))
                 (action (aref actions k))
                 (q-pred (aref q-data (+ base (* action q-cs))))
                 (target (aref target-arr k))
                 (err (- q-pred target)))
            (incf loss-sum (* err err))
            (setf (aref grad-arr (+ (* k 2) action))
                  (/ (* 2.0d0 err) batch-size))))
        (let ((grad-output (vt-reshape
                            (vt-from-sequence (coerce grad-arr 'list) :dtype :float64)
                            (list batch-size 2))))
          (zero-grad! q-net)
          (backward q-net grad-output)
          (optimizer-step q-opt (params q-net) (grads q-net)))
        (/ loss-sum batch-size)))))

;;; ============================================================
;;; 4. 主训练
;;; ============================================================
(defun train-dqn (&key (episodes 2000) (lr 5e-4) (gamma 0.99)
                       (hidden 64) (buffer-capacity 50000)
                       (batch-size 64) (warmup-steps 200)
                       (target-update-freq 500)
                       (learn-every 4)
                       (epsilon-start 1.0d0)
                       (epsilon-end 0.05d0)
                       (epsilon-decay-steps 30000))
  (format t "~%=== CartPole + DQN ===~%")
  (format t "网络: Dense(4→~a) → ReLU → Dense(~a→2)~%" hidden hidden)
  (format t "episodes=~a  lr=~a  gamma=~a  batch=~a~%" episodes lr gamma batch-size)
  (format t "buffer=~a  warmup=~a  target-sync=~a~%" buffer-capacity warmup-steps target-update-freq)
  (format t "epsilon: ~a → ~a over ~a steps~%~%" epsilon-start epsilon-end epsilon-decay-steps)

  (let* ((q-net (make-q-net hidden))
         (target-net (make-q-net hidden))
         (q-opt (make-adam :lr lr))
         (buf (make-replay-buffer buffer-capacity :state-dim 4))
         (total-steps 0)
         (learn-steps 0)
         (recent-returns '()))
    (forward q-net (vt-random-normal '(1 4)))
    (forward target-net (vt-random-normal '(1 4)))
    (setf target-net (copy-network q-net))

    (dotimes (ep episodes)
      (let ((state (cartpole-reset))
            (ep-return 0.0d0)
            (done nil))
        (loop while (not done) do
          ;; ---- 选动作 ----
          (let* ((obs (cartpole-observe state))
                 (progress (min 1.0d0 (/ total-steps (coerce epsilon-decay-steps 'double-float))))
                 (epsilon (+ epsilon-start (* progress (- epsilon-end epsilon-start))))
                 (action (q-epsilon-greedy q-net obs epsilon)))
            (multiple-value-bind (next-state reward done-p)
                (cartpole-step state action)
              (buffer-add! buf obs action reward (cartpole-observe next-state) done-p)
              (incf ep-return reward)
              (incf total-steps)
              (setf state next-state done done-p)))

          ;; ---- 学习 ----
          (when (and (>= (replay-buffer-size buf) warmup-steps)
                     (zerop (mod total-steps learn-every)))
            (train-one-batch-dqn q-net target-net q-opt buf batch-size gamma)
            (incf learn-steps))

          ;; ---- target 同步 ----
          (when (and (>= (replay-buffer-size buf) warmup-steps)
                     (zerop (mod total-steps target-update-freq)))
            (setf target-net (copy-network q-net))))

        (push ep-return recent-returns)
        (when (zerop (mod (1+ ep) 100))
          (let* ((n (min 100 (length recent-returns)))
                 (avg (/ (reduce #'+ (subseq recent-returns 0 n)) n)))
            (format t "Episode ~4a  return=~3a  avg-last-~a=~,1f  steps=~a  learns=~a~%"
                    (1+ ep) (round ep-return) n avg total-steps learn-steps)))))

    (format t "~%测试 20 回合（贪心）...~%")
    (let ((test-returns '()))
      (dotimes (i 20)
        (let ((state (cartpole-reset))
              (done nil)
              (total 0.0d0))
          (loop while (not done) do
            (let* ((obs (cartpole-observe state))
                   (action (q-greedy-action q-net obs)))
              (multiple-value-bind (next reward done-p) (cartpole-step state action)
                (incf total reward)
                (setf state next done done-p))))
          (push total test-returns)))
      (format t "平均回报: ~,1f~%" (/ (reduce #'+ test-returns) 20.0)))
    q-net))


#|
 (train-dqn)

=== CartPole + DQN ===
网络: Dense(4→64) → ReLU → Dense(64→2)
episodes=2000  lr=5.0e-4  gamma=0.99  batch=64
buffer=50000  warmup=200  target-sync=500
epsilon: 1.0 → 0.05 over 30000 steps

Episode 100   return=13   avg-last-100=17.8  steps=1776  learns=395
Episode 200   return=13   avg-last-100=18.0  steps=3576  learns=845
Episode 300   return=31   avg-last-100=17.5  steps=5327  learns=1282
Episode 400   return=11   avg-last-100=19.3  steps=7256  learns=1765
Episode 500   return=17   avg-last-100=28.5  steps=10104  learns=2477
Episode 600   return=19   avg-last-100=42.5  steps=14358  learns=3540
Episode 700   return=163  avg-last-100=74.8  steps=21836  learns=5410
Episode 800   return=500  avg-last-100=313.0  steps=53140  learns=13236
Episode 900   return=365  avg-last-100=380.3  steps=91167  learns=22742
Episode 1000  return=500  avg-last-100=448.6  steps=136032  learns=33959
Episode 1100  return=281  avg-last-100=490.3  steps=185066  learns=46217
Episode 1200  return=500  avg-last-100=494.6  steps=234529  learns=58583
Episode 1300  return=500  avg-last-100=494.1  steps=283941  learns=70936
Episode 1400  return=189  avg-last-100=340.4  steps=317984  learns=79447
Episode 1500  return=500  avg-last-100=442.0  steps=362185  learns=90497
Episode 1600  return=500  avg-last-100=477.8  steps=409963  learns=102441
Episode 1700  return=500  avg-last-100=500.0  steps=459963  learns=114941
Episode 1800  return=500  avg-last-100=472.4  steps=507198  learns=126750
Episode 1900  return=500  avg-last-100=492.4  steps=556442  learns=139061
Episode 2000  return=500  avg-last-100=500.0  steps=606442  learns=151561

测试 20 回合（贪心）...
平均回报: 475.6
|#

;;;; mountaincar-dqn.lisp
;;;; MountainCar + DQN
;;;;
;;;; 与 CartPole 的关键差异：
;;;;   - 状态 (2,)：位置 ∈ [-1.2, 0.6]，速度 ∈ [-0.07, 0.07]
;;;;   - 动作：3 个（0=左推, 1=不推, 2=右推）
;;;;   - 奖励：每步 -1，到达目标（位置 ≥ 0.5）才终止
;;;;   - 稀疏奖励：随机策略永远到不了目标，必须真正探索
;;;;
;;;; 前置：cartpole-reinforce-v3.lisp 已加载
;;;; 用法：(in-package :nn) (load "mountaincar-dqn.lisp") (train-mountaincar-dqn)

(in-package :nn)

;;; ============================================================
;;; 1. MountainCar 环境
;;; ============================================================
(defparameter *mc-min-pos* -1.2d0)
(defparameter *mc-max-pos* 0.6d0)
(defparameter *mc-max-speed* 0.07d0)
(defparameter *mc-goal-pos* 0.5d0)
(defparameter *mc-force* 0.001d0)
(defparameter *mc-gravity* 0.0025d0)
(defparameter *mc-max-steps* 200)

(defstruct mountaincar (pos 0.0d0) (vel 0.0d0) (steps 0))

(defun mc-reset ()
  (make-mountaincar
   :pos (+ -0.6d0 (* 0.2d0 (random 1.0d0)))   ; 位置 ∈ [-0.6, -0.4]
   :vel 0.0d0))

(defun mc-step (s action)
  "ACTION ∈ {0, 1, 2}。返回 (values state reward done)。"
  (let* ((force (- action 1)))                      ; 0→-1, 1→0, 2→+1
    (incf (mountaincar-vel s)
          (+ (* force *mc-force*)
             (* (- (cos (* 3.0d0 (mountaincar-pos s)))) *mc-gravity*)))
    ;; 速度限幅
    (setf (mountaincar-vel s)
          (max (- *mc-max-speed*) (min *mc-max-speed* (mountaincar-vel s))))
    (incf (mountaincar-pos s) (mountaincar-vel s))
    ;; 位置限幅（撞墙速度归零）
    (when (< (mountaincar-pos s) *mc-min-pos*)
      (setf (mountaincar-pos s) *mc-min-pos*)
      (setf (mountaincar-vel s) 0.0d0))
    (when (> (mountaincar-pos s) *mc-max-pos*)
      (setf (mountaincar-pos s) *mc-max-pos*)
      (setf (mountaincar-vel s) 0.0d0))
    (incf (mountaincar-steps s))
    (let ((done (or (>= (mountaincar-pos s) *mc-goal-pos*)
                    (>= (mountaincar-steps s) *mc-max-steps*))))
      (values s -1.0d0 done))))

(defun mc-observe (s)
  (vt-reshape
   (vt-from-sequence
    (list (mountaincar-pos s) (mountaincar-vel s))
    :dtype :float64)
   '(1 2)))

;;; ============================================================
;;; 2. Replay Buffer（复用，但状态维 = 2）
;;; ============================================================
(defstruct (mc-buffer (:constructor %make-mc-buffer))
  (capacity 0 :type fixnum)
  (size 0 :type fixnum)
  (pos 0 :type fixnum)
  (states nil) (actions nil) (rewards nil) (next-states nil) (dones nil))

(defun make-mc-buffer (capacity &key (state-dim 2))
  (%make-mc-buffer
   :capacity capacity :size 0 :pos 0
   :states (make-array (* capacity state-dim)
                       :element-type 'double-float :initial-element 0.0d0)
   :actions (make-array capacity :element-type 'fixnum :initial-element 0)
   :rewards (make-array capacity :element-type 'double-float :initial-element 0.0d0)
   :next-states (make-array (* capacity state-dim)
                            :element-type 'double-float :initial-element 0.0d0)
   :dones (make-array capacity :element-type 'fixnum :initial-element 0)))

(defun mc-buffer-add! (buf state action reward next-state done &key (state-dim 2))
  (let* ((pos (mc-buffer-pos buf))
         (s-list (if (vt-p state) (vt-to-list (vt-flatten state)) state))
         (n-list (if (vt-p next-state) (vt-to-list (vt-flatten next-state)) next-state)))
    (dotimes (i state-dim)
      (setf (aref (mc-buffer-states buf) (+ (* pos state-dim) i))
            (coerce (nth i s-list) 'double-float))
      (setf (aref (mc-buffer-next-states buf) (+ (* pos state-dim) i))
            (coerce (nth i n-list) 'double-float)))
    (setf (aref (mc-buffer-actions buf) pos) action)
    (setf (aref (mc-buffer-rewards buf) pos) (coerce reward 'double-float))
    (setf (aref (mc-buffer-dones buf) pos) (if done 1 0))
    (setf (mc-buffer-pos buf) (mod (1+ pos) (mc-buffer-capacity buf)))
    (setf (mc-buffer-size buf)
          (min (1+ (mc-buffer-size buf)) (mc-buffer-capacity buf)))))

(defun mc-buffer-sample (buf n &key (state-dim 2))
  (let* ((size (mc-buffer-size buf))
         (s-arr (make-array (* n state-dim) :element-type 'double-float))
         (n-arr (make-array (* n state-dim) :element-type 'double-float))
         (actions (make-array n :element-type 'fixnum))
         (rewards (make-array n :element-type 'double-float))
         (dones (make-array n :element-type 'fixnum)))
    (dotimes (k n)
      (let ((idx (random size)))
        (dotimes (i state-dim)
          (setf (aref s-arr (+ (* k state-dim) i))
                (aref (mc-buffer-states buf) (+ (* idx state-dim) i)))
          (setf (aref n-arr (+ (* k state-dim) i))
                (aref (mc-buffer-next-states buf) (+ (* idx state-dim) i))))
        (setf (aref actions k) (aref (mc-buffer-actions buf) idx))
        (setf (aref rewards k) (aref (mc-buffer-rewards buf) idx))
        (setf (aref dones k) (aref (mc-buffer-dones buf) idx))))
    (values
     (vt-reshape (vt-from-sequence (coerce s-arr 'list) :dtype :float64)
                 (list n state-dim))
     actions rewards
     (vt-reshape (vt-from-sequence (coerce n-arr 'list) :dtype :float64)
                 (list n state-dim))
     dones)))

;;; ============================================================
;;; 3. Q 网络（3 个动作）
;;; ============================================================
(defun make-mc-q-net (hidden)
  "Dense(2→hidden) → ReLU → Dense(hidden→3)"
  (let ((m (make-sequential)))
    (seq-add! m (make-dense hidden :in-dim 2 :activation :relu))
    (seq-add! m (make-dense 3 :activation :none))
    m))

(defun mc-argmax (q-flat)
  "Q-FLAT 是 3 元素列表。返回 argmax 索引。"
  (let ((best 0) (best-val (first q-flat)))
    (loop for v in (rest q-flat)
          for i from 1
          do (when (> v best-val) (setf best-val v best i)))
    best))

(defun mc-greedy-action (q-net state-obs)
  (mc-argmax (vt-to-list (vt-flatten (forward q-net state-obs)))))

(defun mc-epsilon-greedy (q-net state-obs epsilon)
  (if (< (random 1.0d0) epsilon) (random 3) (mc-greedy-action q-net state-obs)))

;;; ============================================================
;;; 4. 单次 DQN 更新（3 动作版）
;;; ============================================================
(defun train-one-batch-mc-dqn (q-net target-net q-opt buf batch-size gamma
                                &key (state-dim 2) (n-actions 3))
  (multiple-value-bind (states actions rewards next-states dones)
      (mc-buffer-sample buf batch-size :state-dim state-dim)
    ;; ---- target: r + γ * max_a Q_target(s') * (1 - done) ----
    (let* ((q-next (forward target-net next-states))
           (qn-data (vt-data q-next))
           (qn-off (vt-offset q-next))
           (qn-rs (first (vt-strides q-next)))
           (qn-cs (second (vt-strides q-next)))
           (target-arr (make-array batch-size :element-type 'double-float)))
      (dotimes (k batch-size)
        (let* ((base (+ qn-off (* k qn-rs)))
               (q0 (aref qn-data base))
               (q1 (aref qn-data (+ base qn-cs)))
               (q2 (aref qn-data (+ base (* 2 qn-cs))))
               (max-q (max q0 q1 q2)))
          (setf (aref target-arr k)
                (+ (aref rewards k)
                   (if (= (aref dones k) 1) 0.0d0 (* gamma max-q))))))
      ;; ---- Q(s, ·)，构造 grad-output ----
      (let* ((q-cur (forward q-net states))
             (q-data (vt-data q-cur))
             (q-off (vt-offset q-cur))
             (q-rs (first (vt-strides q-cur)))
             (q-cs (second (vt-strides q-cur)))
             (grad-arr (make-array (* batch-size n-actions)
                                   :element-type 'double-float
                                   :initial-element 0.0d0))
             (loss-sum 0.0d0))
        (dotimes (k batch-size)
          (let* ((base (+ q-off (* k q-rs)))
                 (action (aref actions k))
                 (q-pred (aref q-data (+ base (* action q-cs))))
                 (target (aref target-arr k))
                 (err (- q-pred target)))
            (incf loss-sum (* err err))
            (setf (aref grad-arr (+ (* k n-actions) action))
                  (/ (* 2.0d0 err) batch-size))))
        (let ((grad-output (vt-reshape
                            (vt-from-sequence (coerce grad-arr 'list) :dtype :float64)
                            (list batch-size n-actions))))
          (zero-grad! q-net)
          (backward q-net grad-output)
          (optimizer-step q-opt (params q-net) (grads q-net)))
        (/ loss-sum batch-size)))))

;;; ============================================================
;;; 5. 主训练
;;; ============================================================
(defun train-mountaincar-dqn (&key (episodes 1500) (lr 5e-4) (gamma 0.99)
                                    (hidden 64) (buffer-capacity 50000)
                                    (batch-size 64) (warmup-steps 500)
                                    (target-update-freq 500)
                                    (learn-every 4)
                                    (epsilon-start 1.0d0)
                                    (epsilon-end 0.05d0)
                                    (epsilon-decay-steps 50000))
  (format t "~%=== MountainCar + DQN ===~%")
  (format t "网络: Dense(2→~a) → ReLU → Dense(~a→3)~%" hidden hidden)
  (format t "episodes=~a  lr=~a  gamma=~a  batch=~a~%" episodes lr gamma batch-size)
  (format t "buffer=~a  warmup=~a  target-sync=~a~%" buffer-capacity warmup-steps target-update-freq)
  (format t "epsilon: ~a → ~a over ~a steps~%~%" epsilon-start epsilon-end epsilon-decay-steps)

  (let* ((q-net (make-mc-q-net hidden))
         (target-net (make-mc-q-net hidden))
         (q-opt (make-adam :lr lr))
         (buf (make-mc-buffer buffer-capacity :state-dim 2))
         (total-steps 0)
         (learn-steps 0)
         (recent-returns '())
         (best-return -200.0d0))
    (forward q-net (vt-random-normal '(1 2)))
    (setf target-net (copy-network q-net))

    (dotimes (ep episodes)
      (let ((state (mc-reset))
            (ep-return 0.0d0)
            (done nil))
        (loop while (not done) do
          (let* ((obs (mc-observe state))
                 (progress (min 1.0d0 (/ total-steps
                                         (coerce epsilon-decay-steps 'double-float))))
                 (epsilon (+ epsilon-start (* progress (- epsilon-end epsilon-start))))
                 (action (mc-epsilon-greedy q-net obs epsilon)))
            (multiple-value-bind (next reward done-p) (mc-step state action)
              (mc-buffer-add! buf obs action reward (mc-observe next) done-p)
              (incf ep-return reward)
              (incf total-steps)
              (setf state next done done-p)))

          (when (and (>= (mc-buffer-size buf) warmup-steps)
                     (zerop (mod total-steps learn-every)))
            (train-one-batch-mc-dqn q-net target-net q-opt buf batch-size gamma)
            (incf learn-steps))

          (when (and (>= (mc-buffer-size buf) warmup-steps)
                     (zerop (mod total-steps target-update-freq)))
            (setf target-net (copy-network q-net))))

        (when (> ep-return best-return)
          (setf best-return ep-return))
        (push ep-return recent-returns)
        (when (zerop (mod (1+ ep) 100))
          (let* ((n (min 100 (length recent-returns)))
                 (avg (/ (reduce #'+ (subseq recent-returns 0 n)) n)))
            (format t "Episode ~4a  return=~4a  avg-last-~a=~,1f  best=~a  steps=~a~%"
                    (1+ ep) (round ep-return) n avg (round best-return) total-steps)))))

    (format t "~%测试 20 回合（贪心）...~%")
    (let ((test-returns '()))
      (dotimes (i 20)
        (let ((state (mc-reset))
              (done nil)
              (total 0.0d0))
          (loop while (not done) do
            (let* ((obs (mc-observe state))
                   (action (mc-greedy-action q-net obs)))
              (multiple-value-bind (next reward done-p) (mc-step state action)
                (incf total reward)
                (setf state next done done-p))))
          (push total test-returns)))
      (format t "平均回报: ~,1f~%（-200 = 随机/失败；接近 -100 表示成功）~%"
              (/ (reduce #'+ test-returns) 20.0)))
    q-net))















































;;;; lunarlander-dqn.lisp
;;;; LunarLander (简化版) + DQN
;;;;
;;;; 8 维状态, 4 动作, 稠密奖励
;;;; 与 CartPole/MountainCar 的关键差异：
;;;;   - 状态更多 (8 vs 4/2)
;;;;   - 动作更多 (4 vs 2/3)
;;;;   - 物理更复杂 (推力/重力/旋转/着陆判定)
;;;;
;;;; 前置：cartpole-reinforce-v3.lisp 已加载
;;;; 用法：(in-package :nn) (load "lunarlander-dqn.lisp") (train-lunarlander-dqn)

(in-package :nn)

;;; ============================================================
;;; 1. LunarLander 简化环境
;;; ============================================================
;;; 物理参数（取自 gym LunarLander 的简化版本）
(defparameter *ll-fps* 50)
(defparameter *ll-dt* (/ 1.0d0 *ll-fps*))        ; 0.02
(defparameter *ll-gravity* -10.0d0)
(defparameter *ll-main-thrust* 13.0d0)
(defparameter *ll-side-thrust* 0.6d0)
(defparameter *ll-main-fuel-cost* 0.30d0)
(defparameter *ll-side-fuel-cost* 0.03d0)
(defparameter *ll-leg-pos* 0.05d0)               ; 着陆腿相对坐标
(defparameter *ll-leg-spring* 4.0d0)             ; 弹簧系数（简化用）
(defparameter *ll-max-time* 1000)                ; 最大步数（20 秒）
(defparameter *ll-pad-x* 0.0d0)                  ; 着陆平台中心
(defparameter *ll-pad-width* 0.4d0)              ; 平台宽度

;;; 状态：
;;;   x        - 水平位置 [-1.5, 1.5]
;;;   y        - 高度     [0, ∞)
;;;   vx       - 水平速度 [-5, 5]
;;;   vy       - 垂直速度 [-5, 5]
;;;   theta    - 角度     [-π, π]
;;;   omega    - 角速度   [-5, 5]
;;;   leg-left - 左腿接触 (0/1)
;;;   leg-right- 右腿接触 (0/1)

(defstruct lunarlander
  (x 0.0d0) (y 0.0d0)
  (vx 0.0d0) (vy 0.0d0)
  (theta 0.0d0) (omega 0.0d0)
  (leg-left 0) (leg-right 0)
  (steps 0) (prev-leg-contact nil))

(defun ll-reset ()
  (make-lunarlander
   :x (+ -0.1d0 (* 0.2d0 (random 1.0d0)))
   :y (+ 1.2d0 (* 0.2d0 (random 1.0d0)))
   :vx (+ -0.1d0 (* 0.2d0 (random 1.0d0)))
   :vy 0.0d0
   :theta (+ -0.05d0 (* 0.1d0 (random 1.0d0)))
   :omega 0.0d0
   :leg-left 0 :leg-right 0
   :steps 0))

(defun ll-step (s action)
  "ACTION ∈ {0: noop, 1: left engine, 2: main engine, 3: right engine}。
   返回 (values state reward done)。"
  (let ((reward 0.0d0)
        (done nil))
    ;; ---- 1. 施力 ----
    (let ((ox (- (sin (lunarlander-theta s))))
          (oy (cos (lunarlander-theta s))))
      ;; 主引擎：沿飞船朝向的推力
      (when (= action 2)
        ;; 主引擎有概率失效（简化版去掉）
        (incf (lunarlander-vx s) (* ox *ll-main-thrust* *ll-dt*))
        (incf (lunarlander-vy s) (* oy *ll-main-thrust* *ll-dt*))
        ;; 主引擎引起角速度变化（小的推力偏心）
        (incf (lunarlander-omega s) (* 0.0d0 *ll-dt*))
        (decf reward *ll-main-fuel-cost*))
      ;; 侧引擎：旋转力矩
      (when (= action 1)
        (decf (lunarlander-omega s) (* *ll-side-thrust* *ll-dt*))
        (decf reward *ll-side-fuel-cost*))
      (when (= action 3)
        (incf (lunarlander-omega s) (* *ll-side-thrust* *ll-dt*))
        (decf reward *ll-side-fuel-cost*)))

    ;; ---- 2. 重力 + 阻尼 ----
    (incf (lunarlander-vy s) (* *ll-gravity* *ll-dt*))
    (setf (lunarlander-theta s) (+ (lunarlander-theta s)
                                    (* (lunarlander-omega s) *ll-dt*)))
    ;; 位置更新
    (incf (lunarlander-x s) (* (lunarlander-vx s) *ll-dt*))
    (incf (lunarlander-y s) (* (lunarlander-vy s) *ll-dt*))

    ;; ---- 3. 速度限幅 ----
    (setf (lunarlander-vx s)
          (max -5.0d0 (min 5.0d0 (lunarlander-vx s))))
    (setf (lunarlander-vy s)
          (max -5.0d0 (min 5.0d0 (lunarlander-vy s))))
    (setf (lunarlander-omega s)
          (max -5.0d0 (min 5.0d0 (lunarlander-omega s))))

    ;; ---- 4. 着地检测 ----
    (let ((leg-l-x (+ (lunarlander-x s)
                      (- (* *ll-leg-pos* (cos (lunarlander-theta s))))
                      (* *ll-leg-pos* (sin (lunarlander-theta s)))))
          (leg-r-x (+ (lunarlander-x s)
                      (* *ll-leg-pos* (cos (lunarlander-theta s)))
                      (* *ll-leg-pos* (sin (lunarlander-theta s)))))
          (leg-y (- (lunarlander-y s)
                    (* *ll-leg-pos* (cos (lunarlander-theta s))))))
      ;; 简化：两条腿同高，只要 y 低到一定高度就算接触
      (let ((contact (> leg-y 0.0d0)))
        (declare (ignore contact))
        (setf (lunarlander-leg-left s) (if (<= (lunarlander-y s) 0.10d0) 1 0))
        (setf (lunarlander-leg-right s) (if (<= (lunarlander-y s) 0.10d0) 1 0))
      ;;  (declare (ignore leg-l-x leg-r-x)))
      ))

    ;; ---- 5. 着陆 / 坠毁判定 ----
    (when (<= (lunarlander-y s) 0.0d0)
      (setf (lunarlander-y s) 0.0d0)
      (let* ((on-pad (and (< (abs (- (lunarlander-x s) *ll-pad-x*))
                            (/ *ll-pad-width* 2.0d0))
                          (< (abs (lunarlander-theta s)) (/ pi 4.0d0))
                          (< (abs (lunarlander-vx s)) 1.0d0)
                          (< (abs (lunarlander-vy s)) 1.5d0))))
        (if on-pad
            (progn
              (incf reward 100.0d0)
              (setf done t))
            (progn
              (incf reward -100.0d0)
              (setf done t)))))

    ;; ---- 6. 越界 / 超时 ----
    (when (or (> (abs (lunarlander-x s)) 1.5d0)
              (> (lunarlander-y s) 2.0d0))
      (incf reward -100.0d0)
      (setf done t))
    (when (>= (lunarlander-steps s) *ll-max-time*)
      (setf done t))

    ;; ---- 7. 每步燃料消耗 ----
    (decf reward 0.3d0)
    (incf (lunarlander-steps s))
    (values s reward done)))

(defun ll-observe (s)
  "返回 (1, 8) VT。"
  (vt-reshape
   (vt-from-sequence
    (list (lunarlander-x s)
          (lunarlander-y s)
          (lunarlander-vx s)
          (lunarlander-vy s)
          (lunarlander-theta s)
          (lunarlander-omega s)
          (coerce (lunarlander-leg-left s) 'double-float)
          (coerce (lunarlander-leg-right s) 'double-float))
    :dtype :float64)
   '(1 8)))

;;; ============================================================
;;; 2. Replay Buffer (8 维状态)
;;; ============================================================
(defstruct (ll-buffer (:constructor %make-ll-buffer))
  (capacity 0 :type fixnum)
  (size 0 :type fixnum)
  (pos 0 :type fixnum)
  (states nil) (actions nil) (rewards nil) (next-states nil) (dones nil))

(defun make-ll-buffer (capacity &key (state-dim 8))
  (%make-ll-buffer
   :capacity capacity :size 0 :pos 0
   :states (make-array (* capacity state-dim)
                       :element-type 'double-float :initial-element 0.0d0)
   :actions (make-array capacity :element-type 'fixnum :initial-element 0)
   :rewards (make-array capacity :element-type 'double-float :initial-element 0.0d0)
   :next-states (make-array (* capacity state-dim)
                            :element-type 'double-float :initial-element 0.0d0)
   :dones (make-array capacity :element-type 'fixnum :initial-element 0)))

(defun ll-buffer-add! (buf state action reward next-state done &key (state-dim 8))
  (let* ((pos (ll-buffer-pos buf))
         (s-list (if (vt-p state) (vt-to-list (vt-flatten state)) state))
         (n-list (if (vt-p next-state) (vt-to-list (vt-flatten next-state)) next-state)))
    (dotimes (i state-dim)
      (setf (aref (ll-buffer-states buf) (+ (* pos state-dim) i))
            (coerce (nth i s-list) 'double-float))
      (setf (aref (ll-buffer-next-states buf) (+ (* pos state-dim) i))
            (coerce (nth i n-list) 'double-float)))
    (setf (aref (ll-buffer-actions buf) pos) action)
    (setf (aref (ll-buffer-rewards buf) pos) (coerce reward 'double-float))
    (setf (aref (ll-buffer-dones buf) pos) (if done 1 0))
    (setf (ll-buffer-pos buf) (mod (1+ pos) (ll-buffer-capacity buf)))
    (setf (ll-buffer-size buf)
          (min (1+ (ll-buffer-size buf)) (ll-buffer-capacity buf)))))

(defun ll-buffer-sample (buf n &key (state-dim 8))
  (let* ((size (ll-buffer-size buf))
         (s-arr (make-array (* n state-dim) :element-type 'double-float))
         (n-arr (make-array (* n state-dim) :element-type 'double-float))
         (actions (make-array n :element-type 'fixnum))
         (rewards (make-array n :element-type 'double-float))
         (dones (make-array n :element-type 'fixnum)))
    (dotimes (k n)
      (let ((idx (random size)))
        (dotimes (i state-dim)
          (setf (aref s-arr (+ (* k state-dim) i))
                (aref (ll-buffer-states buf) (+ (* idx state-dim) i)))
          (setf (aref n-arr (+ (* k state-dim) i))
                (aref (ll-buffer-next-states buf) (+ (* idx state-dim) i))))
        (setf (aref actions k) (aref (ll-buffer-actions buf) idx))
        (setf (aref rewards k) (aref (ll-buffer-rewards buf) idx))
        (setf (aref dones k) (aref (ll-buffer-dones buf) idx))))
    (values
     (vt-reshape (vt-from-sequence (coerce s-arr 'list) :dtype :float64)
                 (list n state-dim))
     actions rewards
     (vt-reshape (vt-from-sequence (coerce n-arr 'list) :dtype :float64)
                 (list n state-dim))
     dones)))

;;; ============================================================
;;; 3. Q 网络 (8→128→4)
;;; ============================================================
(defun make-ll-q-net (hidden)
  (let ((m (make-sequential)))
    (seq-add! m (make-dense hidden :in-dim 8 :activation :relu))
    (seq-add! m (make-dense 4 :activation :none))
    m))

(defun ll-argmax (q-flat)
  (let ((best 0) (best-val (first q-flat)))
    (loop for v in (rest q-flat) for i from 1
          do (when (> v best-val) (setf best-val v best i)))
    best))

(defun ll-greedy-action (q-net state-obs)
  (ll-argmax (vt-to-list (vt-flatten (forward q-net state-obs)))))

(defun ll-epsilon-greedy (q-net state-obs epsilon)
  (if (< (random 1.0d0) epsilon) (random 4) (ll-greedy-action q-net state-obs)))

;;; ============================================================
;;; 4. 单次 DQN 更新（4 动作）
;;; ============================================================
(defun train-one-batch-ll-dqn (q-net target-net q-opt buf batch-size gamma
                                &key (state-dim 8) (n-actions 4))
  (multiple-value-bind (states actions rewards next-states dones)
      (ll-buffer-sample buf batch-size :state-dim state-dim)
    (let* ((q-next (forward target-net next-states))
           (qn-data (vt-data q-next))
           (qn-off (vt-offset q-next))
           (qn-rs (first (vt-strides q-next)))
           (qn-cs (second (vt-strides q-next)))
           (target-arr (make-array batch-size :element-type 'double-float)))
      (dotimes (k batch-size)
        (let* ((base (+ qn-off (* k qn-rs)))
               (q0 (aref qn-data base))
               (q1 (aref qn-data (+ base qn-cs)))
               (q2 (aref qn-data (+ base (* 2 qn-cs))))
               (q3 (aref qn-data (+ base (* 3 qn-cs))))
               (max-q (max q0 q1 q2 q3)))
          (setf (aref target-arr k)
                (+ (aref rewards k)
                   (if (= (aref dones k) 1) 0.0d0 (* gamma max-q))))))
      (let* ((q-cur (forward q-net states))
             (q-data (vt-data q-cur))
             (q-off (vt-offset q-cur))
             (q-rs (first (vt-strides q-cur)))
             (q-cs (second (vt-strides q-cur)))
             (grad-arr (make-array (* batch-size n-actions)
                                   :element-type 'double-float
                                   :initial-element 0.0d0))
             (loss-sum 0.0d0))
        (dotimes (k batch-size)
          (let* ((base (+ q-off (* k q-rs)))
                 (action (aref actions k))
                 (q-pred (aref q-data (+ base (* action q-cs))))
                 (target (aref target-arr k))
                 (err (- q-pred target)))
            (incf loss-sum (* err err))
            (setf (aref grad-arr (+ (* k n-actions) action))
                  (/ (* 2.0d0 err) batch-size))))
        (let ((grad-output (vt-reshape
                            (vt-from-sequence (coerce grad-arr 'list) :dtype :float64)
                            (list batch-size n-actions))))
          (zero-grad! q-net)
          (backward q-net grad-output)
          (optimizer-step q-opt (params q-net) (grads q-net)))
        (/ loss-sum batch-size)))))

;;; ============================================================
;;; 5. 主训练
;;; ============================================================
(defun train-lunarlander-dqn (&key (episodes 2000) (lr 5e-4) (gamma 0.99)
                                    (hidden 128) (buffer-capacity 100000)
                                    (batch-size 128) (warmup-steps 2000)
                                    (target-update-freq 1000)
                                    (learn-every 4)
                                    (epsilon-start 1.0d0)
                                    (epsilon-end 0.05d0)
                                    (epsilon-decay-steps 100000))
  (format t "~%=== LunarLander + DQN ===~%")
  (format t "网络: Dense(8→~a) → ReLU → Dense(~a→4)~%" hidden hidden)
  (format t "episodes=~a  lr=~a  gamma=~a  batch=~a~%" episodes lr gamma batch-size)
  (format t "buffer=~a  warmup=~a  target-sync=~a~%" buffer-capacity warmup-steps target-update-freq)
  (format t "epsilon: ~a → ~a over ~a steps~%~%" epsilon-start epsilon-end epsilon-decay-steps)

  (let* ((q-net (make-ll-q-net hidden))
         (target-net (make-ll-q-net hidden))
         (q-opt (make-adam :lr lr))
         (buf (make-ll-buffer buffer-capacity :state-dim 8))
         (total-steps 0)
         (learn-steps 0)
         (recent-returns '())
         (best-return -500.0d0))
    (forward q-net (vt-random-normal '(1 8)))
    (setf target-net (copy-network q-net))

    (dotimes (ep episodes)
      (let ((state (ll-reset))
            (ep-return 0.0d0)
            (done nil))
        (loop while (not done) do
          (let* ((obs (ll-observe state))
                 (progress (min 1.0d0 (/ total-steps
                                         (coerce epsilon-decay-steps 'double-float))))
                 (epsilon (+ epsilon-start (* progress (- epsilon-end epsilon-start))))
                 (action (ll-epsilon-greedy q-net obs epsilon)))
            (multiple-value-bind (next reward done-p) (ll-step state action)
              (ll-buffer-add! buf obs action reward (ll-observe next) done-p)
              (incf ep-return reward)
              (incf total-steps)
              (setf state next done done-p)))

          (when (and (>= (ll-buffer-size buf) warmup-steps)
                     (zerop (mod total-steps learn-every)))
            (train-one-batch-ll-dqn q-net target-net q-opt buf batch-size gamma)
            (incf learn-steps))

          (when (and (>= (ll-buffer-size buf) warmup-steps)
                     (zerop (mod total-steps target-update-freq)))
            (setf target-net (copy-network q-net))))

        (when (> ep-return best-return) (setf best-return ep-return))
        (push ep-return recent-returns)
        (when (zerop (mod (1+ ep) 100))
          (let* ((n (min 100 (length recent-returns)))
                 (avg (/ (reduce #'+ (subseq recent-returns 0 n)) n)))
            (format t "Episode ~4a  return=~5,0f  avg-last-~a=~6,1f  best=~5,0f  steps=~a~%"
                    (1+ ep) ep-return n avg best-return total-steps)))))

    (format t "~%测试 20 回合（贪心）...~%")
    (let ((test-returns '()))
      (dotimes (i 20)
        (let ((state (ll-reset))
              (done nil)
              (total 0.0d0))
          (loop while (not done) do
            (let* ((obs (ll-observe state))
                   (action (ll-greedy-action q-net obs)))
              (multiple-value-bind (next reward done-p) (ll-step state action)
                (incf total reward)
                (setf state next done done-p))))
          (push total test-returns)))
      (format t "平均回报: ~,1f~%（200+ = 良好着陆；-100 以下 = 坠毁）~%"
              (/ (reduce #'+ test-returns) 20.0)))
    q-net))




















;;;; lunarlander-dqn-v2.lisp
;;;; LunarLander + DQN（修正版）
;;;;
;;;; 相比 v1 的 4 类改动：
;;;;   1. 环境物理：主引擎推力 13→20，初始高度 1.2→1.6，时间窗口更充裕
;;;;   2. 奖励：成功着陆 +100→+200，燃料 -0.3→-0.1，信号更强
;;;;   3. 着陆判定：速度/角度容忍度放宽，平台加宽
;;;;   4. 超参：epsilon 衰减 100k→250k，episodes 2000→3000
;;;;
;;;; 前置：cartpole-reinforce-v3.lisp 已加载
;;;; 用法：(in-package :nn) (load "lunarlander-dqn-v2.lisp") (train-lunarlander-dqn-v2)

(in-package :nn)

;;; ============================================================
;;; 1. LunarLander 环境（修正物理）
;;; ============================================================
(defparameter *ll-dt* 0.02d0)
(defparameter *ll-gravity* -10.0d0)
(defparameter *ll-main-thrust* 20.0d0)      ; ← 13→20，更可控
(defparameter *ll-side-thrust* 1.5d0)       ; ← 0.6→1.5，旋转响应更快
(defparameter *ll-main-fuel-cost* 0.10d0)   ; ← 0.30→0.10
(defparameter *ll-side-fuel-cost* 0.01d0)   ; ← 0.03→0.01
(defparameter *ll-max-time* 1000)
(defparameter *ll-pad-x* 0.0d0)
(defparameter *ll-pad-width* 0.6d0)         ; ← 0.4→0.6，平台更宽

(defstruct lunarlander
  (x 0.0d0) (y 0.0d0)
  (vx 0.0d0) (vy 0.0d0)
  (theta 0.0d0) (omega 0.0d0)
  (leg-left 0) (leg-right 0)
  (steps 0))

(defun ll-reset ()
  (make-lunarlander
   :x (+ -0.05d0 (* 0.1d0 (random 1.0d0)))    ; x ∈ [-0.05, 0.05]
   :y (+ 1.6d0 (* 0.2d0 (random 1.0d0)))      ; y ∈ [1.6, 1.8]
   :vx (+ -0.05d0 (* 0.1d0 (random 1.0d0)))   ; vx 小
   :vy 0.0d0
   :theta (+ -0.02d0 (* 0.04d0 (random 1.0d0))) ; 角度很小
   :omega 0.0d0
   :leg-left 0 :leg-right 0
   :steps 0))

(defun ll-step (s action)
  "ACTION ∈ {0: noop, 1: left, 2: main, 3: right}。"
  (let ((reward 0.0d0)
        (done nil))
    ;; ---- 1. 施力 ----
    (let ((ox (- (sin (lunarlander-theta s))))
          (oy (cos (lunarlander-theta s))))
      (when (= action 2)
        (incf (lunarlander-vx s) (* ox *ll-main-thrust* *ll-dt*))
        (incf (lunarlander-vy s) (* oy *ll-main-thrust* *ll-dt*))
        (decf reward *ll-main-fuel-cost*))
      (when (= action 1)
        (decf (lunarlander-omega s) (* *ll-side-thrust* *ll-dt*))
        (decf reward *ll-side-fuel-cost*))
      (when (= action 3)
        (incf (lunarlander-omega s) (* *ll-side-thrust* *ll-dt*))
        (decf reward *ll-side-fuel-cost*)))

    ;; ---- 2. 重力 + 运动 ----
    (incf (lunarlander-vy s) (* *ll-gravity* *ll-dt*))
    (setf (lunarlander-theta s) (+ (lunarlander-theta s)
                                    (* (lunarlander-omega s) *ll-dt*)))
    (incf (lunarlander-x s) (* (lunarlander-vx s) *ll-dt*))
    (incf (lunarlander-y s) (* (lunarlander-vy s) *ll-dt*))

    ;; ---- 3. 速度限幅 ----
    (setf (lunarlander-vx s) (max -5.0d0 (min 5.0d0 (lunarlander-vx s))))
    (setf (lunarlander-vy s) (max -5.0d0 (min 5.0d0 (lunarlander-vy s))))
    (setf (lunarlander-omega s) (max -5.0d0 (min 5.0d0 (lunarlander-omega s))))

    ;; ---- 4. 腿接触状态（简化：低高度就算接触）----
    (setf (lunarlander-leg-left s)  (if (<= (lunarlander-y s) 0.10d0) 1 0))
    (setf (lunarlander-leg-right s) (if (<= (lunarlander-y s) 0.10d0) 1 0))

    ;; ---- 5. 着陆 / 坠毁 ----
    (when (<= (lunarlander-y s) 0.0d0)
      (setf (lunarlander-y s) 0.0d0)
      (let* ((on-pad (and (< (abs (- (lunarlander-x s) *ll-pad-x*))
                            (/ *ll-pad-width* 2.0d0))
                          (< (abs (lunarlander-theta s)) (/ pi 3.0d0))   ; ← 放宽
                          (< (abs (lunarlander-vx s)) 2.0d0)              ; ← 放宽
                          (< (abs (lunarlander-vy s)) 2.5d0))))           ; ← 放宽
        (if on-pad
            (progn
              (incf reward 200.0d0)   ; ← +100→+200
              (setf done t))
            (progn
              (incf reward -100.0d0)
              (setf done t)))))

    ;; ---- 6. 越界 ----
    (when (or (> (abs (lunarlander-x s)) 1.5d0)
              (> (lunarlander-y s) 2.5d0))    ; ← 天花板提高
      (incf reward -100.0d0)
      (setf done t))
    (when (>= (lunarlander-steps s) *ll-max-time*)
      (setf done t))

    ;; ---- 7. 每步基础惩罚 ----
    (decf reward 0.05d0)                      ; ← -0.30→-0.05
    (incf (lunarlander-steps s))
    (values s reward done)))

(defun ll-observe (s)
  (vt-reshape
   (vt-from-sequence
    (list (lunarlander-x s) (lunarlander-y s)
          (lunarlander-vx s) (lunarlander-vy s)
          (lunarlander-theta s) (lunarlander-omega s)
          (coerce (lunarlander-leg-left s) 'double-float)
          (coerce (lunarlander-leg-right s) 'double-float))
    :dtype :float64)
   '(1 8)))

;;; ============================================================
;;; 2. Replay Buffer (8 维状态，4 动作)
;;; ============================================================
(defstruct (ll-buffer (:constructor %make-ll-buffer))
  (capacity 0 :type fixnum) (size 0 :type fixnum) (pos 0 :type fixnum)
  (states nil) (actions nil) (rewards nil) (next-states nil) (dones nil))

(defun make-ll-buffer (capacity &key (state-dim 8))
  (%make-ll-buffer
   :capacity capacity :size 0 :pos 0
   :states (make-array (* capacity state-dim)
                       :element-type 'double-float :initial-element 0.0d0)
   :actions (make-array capacity :element-type 'fixnum :initial-element 0)
   :rewards (make-array capacity :element-type 'double-float :initial-element 0.0d0)
   :next-states (make-array (* capacity state-dim)
                            :element-type 'double-float :initial-element 0.0d0)
   :dones (make-array capacity :element-type 'fixnum :initial-element 0)))

(defun ll-buffer-add! (buf state action reward next-state done &key (state-dim 8))
  (let* ((pos (ll-buffer-pos buf))
         (s-list (if (vt-p state) (vt-to-list (vt-flatten state)) state))
         (n-list (if (vt-p next-state) (vt-to-list (vt-flatten next-state)) next-state)))
    (dotimes (i state-dim)
      (setf (aref (ll-buffer-states buf) (+ (* pos state-dim) i))
            (coerce (nth i s-list) 'double-float))
      (setf (aref (ll-buffer-next-states buf) (+ (* pos state-dim) i))
            (coerce (nth i n-list) 'double-float)))
    (setf (aref (ll-buffer-actions buf) pos) action)
    (setf (aref (ll-buffer-rewards buf) pos) (coerce reward 'double-float))
    (setf (aref (ll-buffer-dones buf) pos) (if done 1 0))
    (setf (ll-buffer-pos buf) (mod (1+ pos) (ll-buffer-capacity buf)))
    (setf (ll-buffer-size buf)
          (min (1+ (ll-buffer-size buf)) (ll-buffer-capacity buf)))))

(defun ll-buffer-sample (buf n &key (state-dim 8))
  (let* ((size (ll-buffer-size buf))
         (s-arr (make-array (* n state-dim) :element-type 'double-float))
         (n-arr (make-array (* n state-dim) :element-type 'double-float))
         (actions (make-array n :element-type 'fixnum))
         (rewards (make-array n :element-type 'double-float))
         (dones (make-array n :element-type 'fixnum)))
    (dotimes (k n)
      (let ((idx (random size)))
        (dotimes (i state-dim)
          (setf (aref s-arr (+ (* k state-dim) i))
                (aref (ll-buffer-states buf) (+ (* idx state-dim) i)))
          (setf (aref n-arr (+ (* k state-dim) i))
                (aref (ll-buffer-next-states buf) (+ (* idx state-dim) i))))
        (setf (aref actions k) (aref (ll-buffer-actions buf) idx))
        (setf (aref rewards k) (aref (ll-buffer-rewards buf) idx))
        (setf (aref dones k) (aref (ll-buffer-dones buf) idx))))
    (values
     (vt-reshape (vt-from-sequence (coerce s-arr 'list) :dtype :float64)
                 (list n state-dim))
     actions rewards
     (vt-reshape (vt-from-sequence (coerce n-arr 'list) :dtype :float64)
                 (list n state-dim))
     dones)))

;;; ============================================================
;;; 3. Q 网络
;;; ============================================================
(defun make-ll-q-net (hidden)
  (let ((m (make-sequential)))
    (seq-add! m (make-dense hidden :in-dim 8 :activation :relu))
    (seq-add! m (make-dense 4 :activation :none))
    m))

(defun ll-argmax (q-flat)
  (let ((best 0) (best-val (first q-flat)))
    (loop for v in (rest q-flat) for i from 1
          do (when (> v best-val) (setf best-val v best i)))
    best))

(defun ll-greedy-action (q-net state-obs)
  (ll-argmax (vt-to-list (vt-flatten (forward q-net state-obs)))))

(defun ll-epsilon-greedy (q-net state-obs epsilon)
  (if (< (random 1.0d0) epsilon) (random 4) (ll-greedy-action q-net state-obs)))

;;; ============================================================
;;; 4. DQN 更新
;;; ============================================================
(defun train-one-batch-ll-dqn (q-net target-net q-opt buf batch-size gamma
                                &key (state-dim 8) (n-actions 4))
  (multiple-value-bind (states actions rewards next-states dones)
      (ll-buffer-sample buf batch-size :state-dim state-dim)
    (let* ((q-next (forward target-net next-states))
           (qn-data (vt-data q-next)) (qn-off (vt-offset q-next))
           (qn-rs (first (vt-strides q-next))) (qn-cs (second (vt-strides q-next)))
           (target-arr (make-array batch-size :element-type 'double-float)))
      (dotimes (k batch-size)
        (let* ((base (+ qn-off (* k qn-rs)))
               (max-q (max (aref qn-data base)
                           (aref qn-data (+ base qn-cs))
                           (aref qn-data (+ base (* 2 qn-cs)))
                           (aref qn-data (+ base (* 3 qn-cs))))))
          (setf (aref target-arr k)
                (+ (aref rewards k)
                   (if (= (aref dones k) 1) 0.0d0 (* gamma max-q))))))
      (let* ((q-cur (forward q-net states))
             (q-data (vt-data q-cur)) (q-off (vt-offset q-cur))
             (q-rs (first (vt-strides q-cur))) (q-cs (second (vt-strides q-cur)))
             (grad-arr (make-array (* batch-size n-actions)
                                   :element-type 'double-float :initial-element 0.0d0))
             (loss-sum 0.0d0))
        (dotimes (k batch-size)
          (let* ((base (+ q-off (* k q-rs)))
                 (action (aref actions k))
                 (q-pred (aref q-data (+ base (* action q-cs))))
                 (target (aref target-arr k))
                 (err (- q-pred target)))
            (incf loss-sum (* err err))
            (setf (aref grad-arr (+ (* k n-actions) action))
                  (/ (* 2.0d0 err) batch-size))))
        (let ((grad-output (vt-reshape
                            (vt-from-sequence (coerce grad-arr 'list) :dtype :float64)
                            (list batch-size n-actions))))
          (zero-grad! q-net)
          (backward q-net grad-output)
          (optimizer-step q-opt (params q-net) (grads q-net)))
        (/ loss-sum batch-size)))))

;;; ============================================================
;;; 5. 主训练
;;; ============================================================
(defun train-lunarlander-dqn-v2 (&key (episodes 3000) (lr 5e-4) (gamma 0.99)
                                       (hidden 128) (buffer-capacity 100000)
                                       (batch-size 128) (warmup-steps 2000)
                                       (target-update-freq 1000)
                                       (learn-every 4)
                                       (epsilon-start 1.0d0)
                                       (epsilon-end 0.05d0)
                                       (epsilon-decay-steps 250000))
  (format t "~%=== LunarLander + DQN v2 ===~%")
  (format t "网络: Dense(8→~a) → ReLU → Dense(~a→4)~%" hidden hidden)
  (format t "episodes=~a  lr=~a  gamma=~a  batch=~a~%" episodes lr gamma batch-size)
  (format t "buffer=~a  warmup=~a  target-sync=~a~%" buffer-capacity warmup-steps target-update-freq)
  (format t "epsilon: ~a → ~a over ~a steps~%~%" epsilon-start epsilon-end epsilon-decay-steps)

  (let* ((q-net (make-ll-q-net hidden))
         (target-net (make-ll-q-net hidden))
         (q-opt (make-adam :lr lr))
         (buf (make-ll-buffer buffer-capacity :state-dim 8))
         (total-steps 0) (learn-steps 0)
         (recent-returns '())
         (best-return -500.0d0))
    (forward q-net (vt-random-normal '(1 8)))
    (setf target-net (copy-network q-net))

    (dotimes (ep episodes)
      (let ((state (ll-reset))
            (ep-return 0.0d0)
            (done nil))
        (loop while (not done) do
          (let* ((obs (ll-observe state))
                 (progress (min 1.0d0 (/ total-steps
                                         (coerce epsilon-decay-steps 'double-float))))
                 (epsilon (+ epsilon-start (* progress (- epsilon-end epsilon-start))))
                 (action (ll-epsilon-greedy q-net obs epsilon)))
            (multiple-value-bind (next reward done-p) (ll-step state action)
              (ll-buffer-add! buf obs action reward (ll-observe next) done-p)
              (incf ep-return reward)
              (incf total-steps)
              (setf state next done done-p)))

          (when (and (>= (ll-buffer-size buf) warmup-steps)
                     (zerop (mod total-steps learn-every)))
            (train-one-batch-ll-dqn q-net target-net q-opt buf batch-size gamma)
            (incf learn-steps))

          (when (and (>= (ll-buffer-size buf) warmup-steps)
                     (zerop (mod total-steps target-update-freq)))
            (setf target-net (copy-network q-net))))

        (when (> ep-return best-return) (setf best-return ep-return))
        (push ep-return recent-returns)
        (when (zerop (mod (1+ ep) 100))
          (let* ((n (min 100 (length recent-returns)))
                 (avg (/ (reduce #'+ (subseq recent-returns 0 n)) n)))
            (format t "Episode ~4a  return=~6,0f  avg-last-~a=~7,1f  best=~6,0f  eps=~,3f  steps=~a~%"
                    (1+ ep) ep-return n avg best-return
                    (max epsilon-end
                         (- epsilon-start (* (- epsilon-start epsilon-end)
                                             (min 1.0d0 (/ total-steps
                                                           (coerce epsilon-decay-steps 'double-float))))))
                    total-steps)))))

    (format t "~%测试 20 回合（贪心）...~%")
    (let ((test-returns '()))
      (dotimes (i 20)
        (let ((state (ll-reset))
              (done nil)
              (total 0.0d0))
          (loop while (not done) do
            (let* ((obs (ll-observe state))
                   (action (ll-greedy-action q-net obs)))
              (multiple-value-bind (next reward done-p) (ll-step state action)
                (incf total reward)
                (setf state next done done-p))))
          (push total test-returns)))
      (format t "平均回报: ~,1f~%（200+ = 良好着陆；-100 以下 = 坠毁）~%"
              (/ (reduce #'+ test-returns) 20.0)))
    q-net))



















;;;; cartpole-ppo.lisp
;;;; CartPole + PPO (Proximal Policy Optimization)
;;;;
;;;; 设计：
;;;;   - policy-net: Dense(4→hidden) → ReLU → Dense(hidden→2)
;;;;   - value-net:  Dense(4→hidden) → ReLU → Dense(hidden→1)
;;;;   - 两个网络独立 forward/backward/optimizer-step
;;;;   - 每个 iteration：收集 1 个 episode，计算 GAE，做 K 个 epoch 更新
;;;;
;;;; 前置：cartpole-reinforce-v3.lisp 已加载
;;;; 用法：(in-package :nn) (load "cartpole-ppo.lisp") (train-ppo)

(in-package :nn)

;;; ============================================================
;;; 1. 网络
;;; ============================================================
(defun make-policy-net (hidden)
  (let ((m (make-sequential)))
    (seq-add! m (make-dense hidden :in-dim 4 :activation :relu))
    (seq-add! m (make-dense 2 :activation :none))
    m))

(defun make-value-net (hidden)
  (let ((m (make-sequential)))
    (seq-add! m (make-dense hidden :in-dim 4 :activation :relu))
    (seq-add! m (make-dense 1 :activation :none))
    m))

;;; ============================================================
;;; 2. 采集一个 episode
;;; ============================================================
(defun ppo-collect-episode (policy-net)
  "返回 (values states-list actions-list rewards-list dones-list log-probs-list)。
   states 是 (1, 4) VT 列表；其他都是标量列表。"
  (let ((state (cartpole-reset))
        (states '()) (actions '()) (rewards '()) (dones '()) (logps '())
        (done nil))
    (loop while (not done) do
      (let* ((obs (cartpole-observe state))
             (logits (forward policy-net obs))
             (probs (vt-softmax logits))
             (p0 (vt-ref probs 0 0))
             (action (if (< (random 1.0d0) p0) 0 1))
             (logp (log (if (= action 0) p0 (- 1.0d0 p0)))))
        (push obs states)
        (push action actions)
        (push logp logps)
        (multiple-value-bind (next reward done-p) (cartpole-step state action)
          (push reward rewards)
          (push (if done-p 1 0) dones)
          (setf state next done done-p))))
    (values (nreverse states) (nreverse actions) (nreverse rewards)
            (nreverse dones) (nreverse logps))))

;;; ============================================================
;;; 3. GAE
;;; ============================================================
(defun ppo-compute-gae (value-net states-list rewards dones gamma lambda)
  "返回 (values advs-list rets-list)。"
  (let* ((n-steps (length rewards))
         (states-batch (apply #'vt-concatenate 0 states-list))
         (values (forward value-net states-batch))
         (v-data (vt-data values))
         (v-off (vt-offset values))
         (v-rs (first (vt-strides values)))
         (advs (make-array n-steps :element-type 'double-float))
         (rets (make-array n-steps :element-type 'double-float))
         (last-gae 0.0d0))
    (loop for idx from (1- n-steps) downto 0 do
      (let* ((v-cur (aref v-data (+ v-off (* idx v-rs))))
             (v-next (if (= idx (1- n-steps))
                         0.0d0
                         (aref v-data (+ v-off (* (1+ idx) v-rs)))))
             (r (nth idx rewards))
             (done (nth idx dones))
             (next-mask (if (= done 1) 0.0d0 1.0d0))
             (delta (+ r (* gamma v-next next-mask) (- v-cur)))
             (gae (+ delta (* gamma lambda next-mask last-gae))))
        (setf (aref advs idx) gae)
        (setf (aref rets idx) (+ gae v-cur))
        (setf last-gae gae)))
    ;; 标准化 advantage
    (let* ((mean (/ (reduce #'+ advs) n-steps))
           (var (/ (reduce #'+ (map 'list (lambda (x) (expt (- x mean) 2)) advs))
                   n-steps))
           (std (sqrt (+ var 1.0d-8))))
      (dotimes (idx n-steps)
        (setf (aref advs idx) (/ (- (aref advs idx) mean) std))))
    (values (coerce advs 'list) (coerce rets 'list))))

;;; ============================================================
;;; 4. 单次 PPO 更新（一个 epoch）
;;; ============================================================
(defun ppo-update-epoch (policy-net value-net policy-opt value-opt
                          states-list actions old-logps advs rets
                          clip-eps entropy-coef value-coef)
  "对整批数据做一次更新。"
  (let* ((n (length actions))
         (states-batch (apply #'vt-concatenate 0 states-list))
         ;; ---- policy 前向 ----
         (logits (forward policy-net states-batch))
         (probs (vt-softmax logits))
         (probs-data (vt-data probs))
         (probs-off (vt-offset probs))
         (probs-rs (first (vt-strides probs)))
         (probs-cs (second (vt-strides probs)))
         ;; ---- value 前向 ----
         (values (forward value-net states-batch))
         (v-data (vt-data values))
         (v-off (vt-offset values))
         (v-rs (first (vt-strides values)))
         ;; ---- 梯度张量 ----
         (policy-grad-arr (make-array (* n 2)
                                      :element-type 'double-float
                                      :initial-element 0.0d0))
         (value-grad-arr (make-array n
                                     :element-type 'double-float
                                     :initial-element 0.0d0)))
    (dotimes (i n)
      (let* ((base-p (+ probs-off (* i probs-rs)))
             (p0 (aref probs-data base-p))
             (p1 (aref probs-data (+ base-p probs-cs)))
             (action (nth i actions))
             (old-logp (nth i old-logps))
             (adv (nth i advs))
             (ret (nth i rets))
             (new-logp (log (if (= action 0) p0 p1)))
             (ratio (exp (- new-logp old-logp)))
             (unclipped (* ratio adv))
             (clipped-ratio (max (- 1.0d0 clip-eps)
                                 (min (+ 1.0d0 clip-eps) ratio)))
             (clipped (* clipped-ratio adv))
             (use-clipped (< clipped unclipped))
             (in-range (and (>= ratio (- 1.0d0 clip-eps))
                            (<= ratio (+ 1.0d0 clip-eps))))
             ;; dL/dlogp_new
             (dL-dlogp (cond
                         ;; clip 生效且 ratio 超范围 → 梯度 0
                         ((and use-clipped (not in-range)) 0.0d0)
                         ;; 其他情况 → -ratio * adv
                         (t (- (* ratio adv)))))
             ;; d(log π(a))/dlogits_j = δ_{aj} - softmax_j
             (delta-0 (if (= action 0) 1.0d0 0.0d0))
             (delta-1 (if (= action 1) 1.0d0 0.0d0))
             (g0 (* dL-dlogp (- delta-0 p0)))
             (g1 (* dL-dlogp (- delta-1 p1)))
             ;; entropy bonus: H = -Σ p log p
             (H (+ (* -1.0d0 p0 (log (+ p0 1.0d-8)))
                   (* -1.0d0 p1 (log (+ p1 1.0d-8)))))
             (ent-g0 (* -1.0d0 p0 (+ (log (+ p0 1.0d-8)) H)))
             (ent-g1 (* -1.0d0 p1 (+ (log (+ p1 1.0d-8)) H)))
             ;; value 梯度：d(0.5*(V-R)²)/dV = V - R
             (v-cur (aref v-data (+ v-off (* i v-rs)))))
        (setf (aref policy-grad-arr (+ (* i 2) 0))
              (/ (+ g0 (* entropy-coef ent-g0)) n))
        (setf (aref policy-grad-arr (+ (* i 2) 1))
              (/ (+ g1 (* entropy-coef ent-g1)) n))
        (setf (aref value-grad-arr i)
              (/ (* value-coef (- v-cur ret)) n))))
    ;; ---- 更新 policy ----
    (let ((pg (vt-reshape
               (vt-from-sequence (coerce policy-grad-arr 'list) :dtype :float64)
               (list n 2))))
      (zero-grad! policy-net)
      (backward policy-net pg)
      (optimizer-step policy-opt (params policy-net) (grads policy-net)))
    ;; ---- 更新 value ----
    (let ((vg (vt-reshape
               (vt-from-sequence (coerce value-grad-arr 'list) :dtype :float64)
               (list n 1))))
      (zero-grad! value-net)
      (backward value-net vg)
      (optimizer-step value-opt (params value-net) (grads value-net)))))

;;; ============================================================
;;; 5. 主训练
;;; ============================================================
(defun train-ppo (&key (iterations 300) (lr 3e-4) (gamma 0.99) (lambda 0.95)
                        (hidden 64) (epochs-per-iter 10)
                        (clip-eps 0.2d0) (entropy-coef 0.01d0)
                        (value-coef 0.5d0))
  (format t "~%=== CartPole + PPO ===~%")
  (format t "policy: Dense(4→~a, ReLU) → Dense(~a→2)~%" hidden hidden)
  (format t "value:  Dense(4→~a, ReLU) → Dense(~a→1)~%" hidden hidden)
  (format t "iterations=~a  lr=~a  gamma=~a  lambda=~a~%" iterations lr gamma lambda)
  (format t "ppo-epochs=~a  clip-eps=~a  entropy=~a  vf-coef=~a~%~%"
          epochs-per-iter clip-eps entropy-coef value-coef)

  (let* ((policy-net (make-policy-net hidden))
         (value-net  (make-value-net  hidden))
         (policy-opt (make-adam :lr lr))
         (value-opt  (make-adam :lr lr))
         (recent-returns '()))
    ;; 触发初始化
    (forward policy-net (vt-random-normal '(1 4)))
    (forward value-net  (vt-random-normal '(1 4)))

    (dotimes (iter iterations)
      ;; ---- 收集 episode ----
      (multiple-value-bind (states-list actions rewards dones old-logps)
          (ppo-collect-episode policy-net)
        (let ((ep-return (reduce #'+ rewards)))
          ;; ---- 计算 GAE ----
          (multiple-value-bind (advs rets)
              (ppo-compute-gae value-net states-list rewards dones gamma lambda)
            ;; ---- K 个 epoch ----
            (dotimes (epoch epochs-per-iter)
              (ppo-update-epoch policy-net value-net policy-opt value-opt
                                states-list actions old-logps advs rets
                                clip-eps entropy-coef value-coef)))
          ;; ---- 记录 ----
          (push ep-return recent-returns)
          (when (zerop (mod (1+ iter) 20))
            (let* ((n (min 50 (length recent-returns)))
                   (avg (/ (reduce #'+ (subseq recent-returns 0 n)) n)))
              (format t "Iter ~3a  return=~a  avg-last-~a=~,1f~%"
                      (1+ iter) (round ep-return) n avg))))))

    ;; ---- 测试 ----
    (format t "~%测试 20 回合（贪心）...~%")
    (let ((test-returns '()))
      (dotimes (i 20)
        (let ((state (cartpole-reset))
              (done nil)
              (total 0.0d0))
          (loop while (not done) do
            (let* ((obs (cartpole-observe state))
                   (logits (forward policy-net obs))
                   (q-list (vt-to-list (vt-flatten logits)))
                   (action (if (> (first q-list) (second q-list)) 0 1)))
              (multiple-value-bind (next reward done-p) (cartpole-step state action)
                (incf total reward)
                (setf state next done done-p))))
          (push total test-returns)))
      (format t "平均回报: ~,1f~%" (/ (reduce #'+ test-returns) 20.0)))
    (values policy-net value-net)))

(defun train-ppo-v2 (&key (iterations 100) (lr 3e-4) (gamma 0.99) (lambda 0.95)
                            (hidden 64) (epochs-per-iter 10)
                            (episodes-per-iter 20)
                            (clip-eps 0.2d0) (entropy-coef 0.01d0)
                            (value-coef 0.5d0))
  (format t "~%=== CartPole + PPO v2 ===~%")
  (format t "policy: Dense(4→~a, ReLU) → Dense(~a→2)~%" hidden hidden)
  (format t "value:  Dense(4→~a, ReLU) → Dense(~a→1)~%" hidden hidden)
  (format t "iterations=~a  episodes/iter=~a  epochs/iter=~a~%"
          iterations episodes-per-iter epochs-per-iter)
  (format t "lr=~a  gamma=~a  lambda=~a  clip-eps=~a  entropy=~a  vf-coef=~a~%~%"
          lr gamma lambda clip-eps entropy-coef value-coef)

  (let* ((policy-net (make-policy-net hidden))
         (value-net  (make-value-net  hidden))
         (policy-opt (make-adam :lr lr))
         (value-opt  (make-adam :lr lr))
         (recent-returns '()))
    ;; 触发初始化
    (forward policy-net (vt-random-normal '(1 4)))
    (forward value-net  (vt-random-normal '(1 4)))

    (dotimes (iter iterations)
      ;; ---- 收集 N 个 episode，合并数据 ----
      (let ((all-states '()) (all-actions '()) (all-rewards '())
            (all-dones '()) (all-logps '())
            (ep-returns '()))
        (dotimes (ep episodes-per-iter)
          (multiple-value-bind (states actions rewards dones logps)
              (ppo-collect-episode policy-net)
            (setf all-states (append all-states states))
            (setf all-actions (append all-actions actions))
            (setf all-rewards (append all-rewards rewards))
            (setf all-dones (append all-dones dones))
            (setf all-logps (append all-logps logps))
            (push (reduce #'+ rewards) ep-returns)))

        ;; ---- 计算 GAE ----
        (multiple-value-bind (advs rets)
            (ppo-compute-gae value-net all-states all-rewards all-dones gamma lambda)
          ;; ---- K 个 epoch ----
          (dotimes (epoch epochs-per-iter)
            (ppo-update-epoch policy-net value-net policy-opt value-opt
                              all-states all-actions all-logps advs rets
                              clip-eps entropy-coef value-coef)))

        ;; ---- 记录 ----
        (let ((avg-ep (/ (reduce #'+ ep-returns) episodes-per-iter)))
          (push avg-ep recent-returns)
          (when (zerop (mod (1+ iter) 10))
            (let* ((n (min 20 (length recent-returns)))
                   (avg (/ (reduce #'+ (subseq recent-returns 0 n)) n))
                   (steps-this-iter (length all-actions)))
              (format t "Iter ~3a  avg-ep-return=~,1f  avg-last-~a=~,1f  steps=~a~%"
                      (1+ iter) avg-ep n avg steps-this-iter))))))

    ;; ---- 测试 ----
    (format t "~%测试 20 回合（贪心）...~%")
    (let ((test-returns '()))
      (dotimes (i 20)
        (let ((state (cartpole-reset))
              (done nil)
              (total 0.0d0))
          (loop while (not done) do
            (let* ((obs (cartpole-observe state))
                   (logits (forward policy-net obs))
                   (q-list (vt-to-list (vt-flatten logits)))
                   (action (if (> (first q-list) (second q-list)) 0 1)))
              (multiple-value-bind (next reward done-p) (cartpole-step state action)
                (incf total reward)
                (setf state next done done-p))))
          (push total test-returns)))
      (format t "平均回报: ~,1f~%" (/ (reduce #'+ test-returns) 20.0)))
    (values policy-net value-net)))




;;;; cartpole-ppo-v3.lisp
;;;; CartPole + PPO v3（KL 早停 + 修正超参）
;;;;
;;;; 相比 v2 的改动：
;;;;   1. ppo-update-epoch 返回 approx_kl
;;;;   2. 主训练里 epoch 循环加 KL 早停（超阈值提前 break）
;;;;   3. 默认参数：episodes/iter 20→50, epochs/iter 10→4, lr 3e-4→1e-4
;;;;
;;;; 前置：cartpole-reinforce-v3.lisp 已加载
;;;; 用法：(in-package :nn) (load "cartpole-ppo-v3.lisp") (train-ppo-v3)

(in-package :nn)

;;; ============================================================
;;; 1. 网络
;;; ============================================================
(defun make-policy-net (hidden)
  (let ((m (make-sequential)))
    (seq-add! m (make-dense hidden :in-dim 4 :activation :relu))
    (seq-add! m (make-dense 2 :activation :none))
    m))

(defun make-value-net (hidden)
  (let ((m (make-sequential)))
    (seq-add! m (make-dense hidden :in-dim 4 :activation :relu))
    (seq-add! m (make-dense 1 :activation :none))
    m))

;;; ============================================================
;;; 2. 采集一个 episode
;;; ============================================================
(defun ppo-collect-episode (policy-net)
  (let ((state (cartpole-reset))
        (states '()) (actions '()) (rewards '()) (dones '()) (logps '())
        (done nil))
    (loop while (not done) do
      (let* ((obs (cartpole-observe state))
             (logits (forward policy-net obs))
             (probs (vt-softmax logits))
             (p0 (vt-ref probs 0 0))
             (action (if (< (random 1.0d0) p0) 0 1))
             (logp (log (if (= action 0) p0 (- 1.0d0 p0)))))
        (push obs states)
        (push action actions)
        (push logp logps)
        (multiple-value-bind (next reward done-p) (cartpole-step state action)
          (push reward rewards)
          (push (if done-p 1 0) dones)
          (setf state next done done-p))))
    (values (nreverse states) (nreverse actions) (nreverse rewards)
            (nreverse dones) (nreverse logps))))

;;; ============================================================
;;; 3. GAE
;;; ============================================================
(defun ppo-compute-gae (value-net states-list rewards dones gamma lambda)
  (let* ((n-steps (length rewards))
         (states-batch (apply #'vt-concatenate 0 states-list))
         (values (forward value-net states-batch))
         (v-data (vt-data values))
         (v-off (vt-offset values))
         (v-rs (first (vt-strides values)))
         (advs (make-array n-steps :element-type 'double-float))
         (rets (make-array n-steps :element-type 'double-float))
         (last-gae 0.0d0))
    (loop for idx from (1- n-steps) downto 0 do
      (let* ((v-cur (aref v-data (+ v-off (* idx v-rs))))
             (v-next (if (= idx (1- n-steps))
                         0.0d0
                         (aref v-data (+ v-off (* (1+ idx) v-rs)))))
             (r (nth idx rewards))
             (done (nth idx dones))
             (next-mask (if (= done 1) 0.0d0 1.0d0))
             (delta (+ r (* gamma v-next next-mask) (- v-cur)))
             (gae (+ delta (* gamma lambda next-mask last-gae))))
        (setf (aref advs idx) gae)
        (setf (aref rets idx) (+ gae v-cur))
        (setf last-gae gae)))
    ;; 标准化 advantage
    (let* ((mean (/ (reduce #'+ advs) n-steps))
           (var (/ (reduce #'+ (map 'list (lambda (x) (expt (- x mean) 2)) advs))
                   n-steps))
           (std (sqrt (+ var 1.0d-8))))
      (dotimes (idx n-steps)
        (setf (aref advs idx) (/ (- (aref advs idx) mean) std))))
    (values (coerce advs 'list) (coerce rets 'list))))

;;; ============================================================
;;; 4. 单次 PPO 更新（返回 approx_kl）
;;; ============================================================
(defun ppo-update-epoch (policy-net value-net policy-opt value-opt
                          states-list actions old-logps advs rets
                          clip-eps entropy-coef value-coef)
  "对整批数据做一次更新。返回 approx_kl。"
  (let* ((n (length actions))
         (states-batch (apply #'vt-concatenate 0 states-list))
         ;; ---- policy 前向 ----
         (logits (forward policy-net states-batch))
         (probs (vt-softmax logits))
         (probs-data (vt-data probs))
         (probs-off (vt-offset probs))
         (probs-rs (first (vt-strides probs)))
         (probs-cs (second (vt-strides probs)))
         ;; ---- value 前向 ----
         (values (forward value-net states-batch))
         (v-data (vt-data values))
         (v-off (vt-offset values))
         (v-rs (first (vt-strides values)))
         ;; ---- 梯度张量 ----
         (policy-grad-arr (make-array (* n 2)
                                      :element-type 'double-float
                                      :initial-element 0.0d0))
         (value-grad-arr (make-array n
                                     :element-type 'double-float
                                     :initial-element 0.0d0))
         (kl-sum 0.0d0))
    (dotimes (i n)
      (let* ((base-p (+ probs-off (* i probs-rs)))
             (p0 (aref probs-data base-p))
             (p1 (aref probs-data (+ base-p probs-cs)))
             (action (nth i actions))
             (old-logp (nth i old-logps))
             (adv (nth i advs))
             (ret (nth i rets))
             (new-logp (log (if (= action 0) p0 p1)))
             (ratio (exp (- new-logp old-logp)))
             (unclipped (* ratio adv))
             (clipped-ratio (max (- 1.0d0 clip-eps)
                                 (min (+ 1.0d0 clip-eps) ratio)))
             (clipped (* clipped-ratio adv))
             (use-clipped (< clipped unclipped))
             (in-range (and (>= ratio (- 1.0d0 clip-eps))
                            (<= ratio (+ 1.0d0 clip-eps))))
             (dL-dlogp (cond
                         ((and use-clipped (not in-range)) 0.0d0)
                         (t (- (* ratio adv)))))
             (delta-0 (if (= action 0) 1.0d0 0.0d0))
             (delta-1 (if (= action 1) 1.0d0 0.0d0))
             (g0 (* dL-dlogp (- delta-0 p0)))
             (g1 (* dL-dlogp (- delta-1 p1)))
             (H (+ (* -1.0d0 p0 (log (+ p0 1.0d-8)))
                   (* -1.0d0 p1 (log (+ p1 1.0d-8)))))
             (ent-g0 (* -1.0d0 p0 (+ (log (+ p0 1.0d-8)) H)))
             (ent-g1 (* -1.0d0 p1 (+ (log (+ p1 1.0d-8)) H)))
             (v-cur (aref v-data (+ v-off (* i v-rs)))))
        ;; KL 近似：old_logp - new_logp
        (incf kl-sum (- old-logp new-logp))
        (setf (aref policy-grad-arr (+ (* i 2) 0))
              (/ (+ g0 (* entropy-coef ent-g0)) n))
        (setf (aref policy-grad-arr (+ (* i 2) 1))
              (/ (+ g1 (* entropy-coef ent-g1)) n))
        (setf (aref value-grad-arr i)
              (/ (* value-coef (- v-cur ret)) n))))
    ;; ---- 更新 policy ----
    (let ((pg (vt-reshape
               (vt-from-sequence (coerce policy-grad-arr 'list) :dtype :float64)
               (list n 2))))
      (zero-grad! policy-net)
      (backward policy-net pg)
      (optimizer-step policy-opt (params policy-net) (grads policy-net)))
    ;; ---- 更新 value ----
    (let ((vg (vt-reshape
               (vt-from-sequence (coerce value-grad-arr 'list) :dtype :float64)
               (list n 1))))
      (zero-grad! value-net)
      (backward value-net vg)
      (optimizer-step value-opt (params value-net) (grads value-net)))
    ;; 返回平均 KL
    (/ kl-sum n)))

;;; ============================================================
;;; 5. 主训练（带 KL 早停）
;;; ============================================================
(defun train-ppo-v3 (&key (iterations 200) (lr 1e-4) (gamma 0.99) (lambda 0.95)
                            (hidden 64) (epochs-per-iter 4)
                            (episodes-per-iter 50)
                            (clip-eps 0.2d0) (entropy-coef 0.01d0)
                            (value-coef 0.5d0)
                            (target-kl 0.02d0))
  (format t "~%=== CartPole + PPO v3（KL 早停）===~%")
  (format t "policy: Dense(4→~a, ReLU) → Dense(~a→2)~%" hidden hidden)
  (format t "value:  Dense(4→~a, ReLU) → Dense(~a→1)~%" hidden hidden)
  (format t "iterations=~a  episodes/iter=~a  epochs/iter=~a~%"
          iterations episodes-per-iter epochs-per-iter)
  (format t "lr=~a  gamma=~a  lambda=~a  clip-eps=~a~%" lr gamma lambda clip-eps)
  (format t "entropy=~a  vf-coef=~a  target-kl=~a~%~%"
          entropy-coef value-coef target-kl)

  (let* ((policy-net (make-policy-net hidden))
         (value-net  (make-value-net  hidden))
         (policy-opt (make-adam :lr lr))
         (value-opt  (make-adam :lr lr))
         (recent-returns '())
         (total-kl-early-stops 0))
    (forward policy-net (vt-random-normal '(1 4)))
    (forward value-net  (vt-random-normal '(1 4)))

    (dotimes (iter iterations)
      (let ((all-states '()) (all-actions '()) (all-rewards '())
            (all-dones '()) (all-logps '())
            (ep-returns '()))
        ;; ---- 收集 N 个 episode ----
        (dotimes (ep episodes-per-iter)
          (multiple-value-bind (states actions rewards dones logps)
              (ppo-collect-episode policy-net)
            (setf all-states (append all-states states))
            (setf all-actions (append all-actions actions))
            (setf all-rewards (append all-rewards rewards))
            (setf all-dones (append all-dones dones))
            (setf all-logps (append all-logps logps))
            (push (reduce #'+ rewards) ep-returns)))

        ;; ---- 计算 GAE ----
        (multiple-value-bind (advs rets)
            (ppo-compute-gae value-net all-states all-rewards all-dones gamma lambda)
          ;; ---- K 个 epoch，带 KL 早停 ----
          (let ((early-stopped nil))
            (dotimes (epoch epochs-per-iter)
              (unless early-stopped
                (let ((kl (ppo-update-epoch policy-net value-net
                                             policy-opt value-opt
                                             all-states all-actions all-logps
                                             advs rets
                                             clip-eps entropy-coef value-coef)))
                  ;; KL 超过 1.5 × target 就停
                  (when (> kl (* 1.5d0 target-kl))
                    (setf early-stopped t)
                    (incf total-kl-early-stops)
                    (when (zerop (mod (1+ iter) 20))
                      (format t "  [iter ~a epoch ~a] KL 早停 kl=~,4f~%"
                              (1+ iter) (1+ epoch) kl))))))))

        ;; ---- 记录 ----
        (let ((avg-ep (/ (reduce #'+ ep-returns) episodes-per-iter)))
          (push avg-ep recent-returns)
          (when (zerop (mod (1+ iter) 20))
            (let* ((n (min 20 (length recent-returns)))
                   (avg (/ (reduce #'+ (subseq recent-returns 0 n)) n)))
              (format t "Iter ~3a  avg-ep-return=~,1f  avg-last-~a=~,1f  steps=~a~%"
                      (1+ iter) avg-ep n avg (length all-actions)))))))

    (format t "~%KL 早停总次数: ~a~%" total-kl-early-stops)

    ;; ---- 测试 ----
    (format t "~%测试 20 回合（贪心）...~%")
    (let ((test-returns '()))
      (dotimes (i 20)
        (let ((state (cartpole-reset))
              (done nil)
              (total 0.0d0))
          (loop while (not done) do
            (let* ((obs (cartpole-observe state))
                   (logits (forward policy-net obs))
                   (q-list (vt-to-list (vt-flatten logits)))
                   (action (if (> (first q-list) (second q-list)) 0 1)))
              (multiple-value-bind (next reward done-p) (cartpole-step state action)
                (incf total reward)
                (setf state next done done-p))))
          (push total test-returns)))
      (format t "平均回报: ~,1f~%" (/ (reduce #'+ test-returns) 20.0)))
    (values policy-net value-net)))


#|

 (train-ppo-v3 :iterations 300
              :episodes-per-iter 50   ; 保持
              :epochs-per-iter 6      ; 4→6
              :lr 2e-4                ; 1e-4→2e-4
              :target-kl 0.03)        ; 0.02→0.03，放宽 KL 阈值

=== CartPole + PPO v3（KL 早停）===
policy: Dense(4→64, ReLU) → Dense(64→2)
value:  Dense(4→64, ReLU) → Dense(64→1)
iterations=300  episodes/iter=50  epochs/iter=6
lr=2.0e-4  gamma=0.99  lambda=0.95  clip-eps=0.2
entropy=0.01  vf-coef=0.5  target-kl=0.03

Iter 20   avg-ep-return=19.8  avg-last-20=15.5  steps=988
Iter 40   avg-ep-return=50.1  avg-last-20=33.7  steps=2506
Iter 60   avg-ep-return=81.7  avg-last-20=61.7  steps=4083
Iter 80   avg-ep-return=111.7  avg-last-20=99.5  steps=5584
Iter 100  avg-ep-return=194.4  avg-last-20=147.4  steps=9720
Iter 120  avg-ep-return=218.4  avg-last-20=196.0  steps=10921
Iter 140  avg-ep-return=252.5  avg-last-20=239.5  steps=12624
Iter 160  avg-ep-return=254.9  avg-last-20=255.0  steps=12744
Iter 180  avg-ep-return=277.7  avg-last-20=286.5  steps=13883
Iter 200  avg-ep-return=309.7  avg-last-20=303.0  steps=15484
|#
