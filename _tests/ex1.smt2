(set-info :smt-lib-version 2.6)
(set-logic UF)

(declare-sort Any 0)

(declare-fun a () Any)
(declare-fun b () Any)
(declare-fun c () Any)

(declare-fun p (Any) Bool)
(declare-fun r (Any) Bool)

(assert (not (p a)))
(assert (not (p b)))
(assert (p c))
(assert (not (r b)))

(assert (forall ((?x Any)) (or (p ?x) (r ?x))))

(check-sat)
(exit)

