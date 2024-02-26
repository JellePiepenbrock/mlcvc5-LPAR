(set-info :smt-lib-version 2.6)
(set-logic UF)

(declare-sort Any 0)

(declare-fun a () Any)
(declare-fun b () Any)
(declare-fun c () Any)

(declare-fun p (Any) Bool)

(assert (not (= a b)))
(assert (not (= b c)))
(assert (not (= a c)))

(assert (forall ((?x Any)) (p ?x)))

(check-sat)
(exit)

