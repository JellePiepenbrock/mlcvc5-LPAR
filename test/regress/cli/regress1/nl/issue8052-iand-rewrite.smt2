(set-logic ALL)
(set-info :status unsat)
(declare-fun v () Int)
(assert (exists ((V Int)) (and (= 2 v) (= v ((_ iand 1) v v)))))
(check-sat)
