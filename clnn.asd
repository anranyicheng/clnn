(asdf::defsystem :clnn
  :description "common lisp 神经网络库"
  :author "xizang123321@gmail.com>"
  :license  "MIT"
  :version "0.0.1"
  :serial t
  :depends-on (:clvt :closer-mop)
  :components ((:file "nn-package")
	       (:file "nn-layers-embedding")
	       (:file "nn-optimizer")
	       (:file "nn-initializer")
	       (:file "nn-layers-norm")	     
	       (:file "nn-layers-attention")
	       (:file "nn-layers-rnn")
	       (:file "nn-layers-conv")
	       (:file "nn-loss")
	       (:file "nn-protocol")
	       (:file "nn-layers-dense")
	       (:file "nn-model")
	       (:file "nn-scheduler")))
