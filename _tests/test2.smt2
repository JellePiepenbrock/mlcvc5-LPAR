(set-info :smt-lib-version 2.6)
(set-logic UF)

(declare-sort Any 0)

(declare-fun a1 () Any)
(declare-fun a2 () Any)
(declare-fun a3 () Any)
(declare-fun a4 () Any)
(declare-fun a5 () Any)
(declare-fun a6 () Any)
(declare-fun a7 () Any)
(declare-fun a8 () Any)
(declare-fun a9 () Any)

(declare-fun p (Any Any) Bool)
(declare-fun r (Any Any) Bool)

(assert (p a1 a2))
(assert (p a2 a3))
(assert (p a3 a4))
(assert (p a4 a5))
(assert (p a5 a6))
(assert (p a6 a7))
(assert (p a7 a8))
(assert (p a8 a9))

(assert (forall ((?a Any) (?b Any) (?c Any)) 
   (=> (and (p ?a ?b) (p ?b ?c)) 
       (p ?a ?c))
))

(assert (forall ((a Any) (b Any)) (or (r a b) (r b a))))

(assert (not (p a1 a9)))

(check-sat)
(exit)

