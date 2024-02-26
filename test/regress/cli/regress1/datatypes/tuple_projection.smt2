(set-logic ALL)
(set-info :status sat)
(declare-fun t () (Tuple String String String String))
(declare-fun u () (Tuple String String))
(declare-fun v () Tuple)
(declare-fun x () String)
(assert (= t (tuple "a" "b" "c" "d")))
(assert (= x ((_ tuple.select 0) t)))
(assert (= u ((_ tuple.project 2 3) t)))
(assert (= v (tuple.project t)))
(check-sat)