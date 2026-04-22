;;;; nn-package.lisp — 工业级神经网络库包定义
;;;; 设计目标: 覆盖 ML/DL/RL 中常见网络架构
(in-package #:cl-user)

(defpackage #:nn
  (:use #:cl)
  (:nicknames #:neural-net #:clnn)
  (:import-from #:clvt
		;; ---- 张量核心 ----
   :vt :vt-p :vt-data :vt-shape :vt-element-type
       :vt-zeros :vt-ones :vt-ones-like :vt-zeros-like
       :vt-const :vt-arange :vt-random :vt-random-normal
       :vt-transpose :vt-reshape :vt-squeeze :vt-split
       :vt-copy :vt-contiguous
       :vt-ref :vt-slice :vt-do-each :vt-map :vt-reduce
       :vt-amax
   :vt-+ :vt-- :vt-* :vt-/ :vt-scale :vt-=
       :vt-matmul :vt-einsum :vt-dot :vt-outer
       :vt-sum :vt-mean :vt-std :vt-var
       :vt-amax :vt-amin :vt-argmax :vt-argmin
       :vt-softmax :vt-log-softmax
       :vt-sigmoid :vt-relu :vt-leaky-relu :vt-swish
       :vt-softplus :vt-gelu :vt-mish
   :vt-hard-tanh :vt-hard-sigmoid :vt-tanh
       :vt-exp :vt-log :vt-log2 :vt-log10
       :vt-sqrt :vt-abs :vt-expt :vt-square
       :vt-clip :vt-concatenate :vt-norm
   :vt-mean-squared-error :vt-binary-cross-entropy
   :vt-cross-entropy
       :vt-to-2d-array :vt-from-2d-array
       :vt-flatten-sequence :vt-from-sequence :vt-data->list
   :vt-inv :vt-det :vt-solve :vt-trace
   :vt-take
   :vt-copy-into :vt-flatten)
  (:export
   ;; ============= 协议 =============
   :forward :backward :params :grads :update!
   :set-training! :training-p

   ;; ============= 层 =============
   ;; 基类
   :layer :make-layer :layer-name :layer-trainable-p
   ;; 全连接
   :dense :make-dense
   ;; 激活
   :activation-layer :make-activation-layer
   ;; Dropout
   :dropout :make-dropout
   ;; 批归一化
   :batch-norm :make-batch-norm
   ;; 层归一化
   :layer-norm :make-layer-norm
   ;; 卷积
   :conv2d :make-conv2d
   ;; 池化
   :max-pool2d :make-max-pool2d
   :avg-pool2d :make-avg-pool2d
   ;; 全局池化
   :global-avg-pool2d :make-global-avg-pool2d
   ;; 展平
   :flatten :make-flatten
   ;; 嵌入
   :embedding :make-embedding
   ;; RNN 系列
   :rnn-cell :make-rnn-cell
   :lstm :make-lstm
   :gru :make-gru
   ;; 注意力
   :scaled-dot-product-attention :make-scaled-dot-product-attention
   :multi-head-attention :make-multi-head-attention
   :transformer-block :make-transformer-block
   ;; 残差连接
   :residual :make-residual

   ;; ============= 模型容器 =============
   :sequential :make-sequential :seq-add! :seq-insert!
   :model-forward :model-backward :model-update!
   :zero-grad!
   ;; ============= 损失函数 =============
   :loss :make-loss
   :mse-loss :make-mse-loss
   :bce-loss :make-bce-loss
   :ce-loss :make-ce-loss
   :huber-loss :make-huber-loss
   :kl-divergence-loss :make-kl-divergence-loss
   :smooth-l1-loss :make-smooth-l1-loss
   :cosine-similarity-loss :make-cosine-similarity-loss

   ;; ============= 优化器 =============
   :optimizer :make-optimizer
   :sgd :make-sgd
   :sgd-momentum :make-sgd-momentum
   :adam :make-adam
   :adamw :make-adamw
   :rmsprop :make-rmsprop
   :adagrad :make-adagrad

   ;; ============= 学习率调度 =============
   :lr-scheduler :make-lr-scheduler
   :step-lr :make-step-lr
   :exponential-lr :make-exponential-lr
   :cosine-annealing-lr :make-cosine-annealing-lr
   :reduce-on-plateau :make-reduce-on-plateau
   :warmup-cosine-lr :make-warmup-cosine-lr
   :one-cycle-lr :make-one-cycle-lr
   :scheduler-step! :scheduler-get-lr

   ;; ============= 初始化器 =============
   :initializer :make-initializer
   :he-normal :make-he-normal
   :he-uniform :make-he-uniform
   :xavier-normal :make-xavier-normal
   :xavier-uniform :make-xavier-uniform
   :orthogonal-init :make-orthogonal-init
   :zeros-init :make-zeros-init
   :ones-init :make-ones-init
   :constant-init :make-constant-init
   :kaiming-normal :make-kaiming-normal
   :truncated-normal-init :make-truncated-normal-init
   :init-weight :init-bias

   ;; ============= 正则化 =============
   :l1-regularizer :make-l1-regularizer
   :l2-regularizer :make-l2-regularizer
   :elastic-regularizer :make-elastic-regularizer
   :regularizer-penalty

   ;; ============= 回调 =============
   :callback :make-callback
   :early-stopping :make-early-stopping
   :model-checkpoint :make-model-checkpoint
   :tensorboard-callback :make-tensorboard-callback
   :lr-monitor :make-lr-monitor
   :gradient-clip-callback :make-gradient-clip-callback
   :callback-on-epoch-begin :callback-on-epoch-end
   :callback-on-batch-begin :callback-on-batch-end
   :callback-on-train-begin :callback-on-train-end

   ;; ============= 训练器 =============
   :trainer :make-trainer
   :trainer-fit! :trainer-evaluate :trainer-predict

   ;; ============= 序列化 =============
   :save-model :load-model
   :model->plist :plist->model

   ;; ============= 工具 =============
   :tensor-ensure-2d :tensor-unsqueeze :tensor-softmax
   :tensor-log-softmax :tensor-one-hot
   :tensor-masked-fill :tensor-top-k :tensor-where
   :tensor-repeat :tensor-tile :tensor-pad
   :tensor-gather :tensor-scatter
   :compute-grad-norm
   :clipped-gradient-update!
   :param-count :flops-estimate

   ;; ============= 激活函数名称枚举 =============
   :+activation-names+))
