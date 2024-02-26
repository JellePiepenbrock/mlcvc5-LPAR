; EXPECT: unsat
(set-option :incremental false)
(set-logic ALL)

(declare-fun x () (Relation Int Int))
(declare-fun y () (Relation Int Int))
(declare-fun r () (Relation Int Int))
(declare-fun f () Int)
(declare-fun d () (Tuple Int Int))
(assert (= d (tuple f 3)))
(assert (set.member d y))
(declare-fun e () (Tuple Int Int))
(assert (= e (tuple 4 f)))
(assert (set.member e x))
(declare-fun a () (Tuple Int Int))
(assert (= a (tuple 4 3)))
(assert (= r (rel.join x y)))
(assert (not (set.member a r)))
(check-sat)
