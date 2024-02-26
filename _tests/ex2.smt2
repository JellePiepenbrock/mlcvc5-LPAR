(set-info :smt-lib-version 2.6)
(set-logic UF)

(declare-sort Any 0)

(declare-fun a () Any)
(declare-fun b () Any)
(declare-fun c () Any)

(declare-fun p (Any) Bool)
(declare-fun r (Any) Bool)
(declare-fun s (Any) Bool)

(assert (not (p a)))
(assert (r b))
(assert (s c))

(assert (forall ((?x Any)) (or (r ?x) (s ?x))))
(assert (forall ((?x Any)) (or (not (r ?x)) (p ?x))))
(assert (forall ((?x Any)) (or (not (s ?x)) (p ?x))))

(check-sat)
(exit)

