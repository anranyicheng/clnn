(asdf:defsystem :clnn
  :description "common lisp 神经网络库"
  :author "xizang123321@gmail.com>"
  :license  "MIT"
  :version "0.0.1"
  :serial t
  :depends-on (:clvt :closer-mop)
  :components ((:file "nn-package")
               ;; 协议层（基类与泛型）必须最先加载，
               ;; 后续所有层/损失/优化器/初始化器都依赖它
               (:file "nn-protocol")
               (:file "nn-layers-embedding")
               (:file "nn-optimizer")
               (:file "nn-initializer")
               (:file "nn-layers-norm")
               (:file "nn-layers-attention")
               (:file "nn-layers-rnn")
               (:file "nn-layers-conv")
               (:file "nn-loss")
               (:file "nn-layers-dense")
               (:file "nn-model")
               (:file "nn-scheduler")))
