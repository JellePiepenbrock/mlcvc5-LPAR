(set-info :smt-lib-version 2.6)
(set-logic UF)

(declare-sort Any 0)

(declare-fun a1 () Any)
(declare-fun a2 () Any)
(declare-fun a3 () Any)
(declare-fun a4 () Any)

(declare-fun p (Any Any) Bool)

(assert (p a1 a2))
(assert (p a2 a3))
(assert (p a3 a4))

(assert (forall ((?a Any) (?b Any) (?c Any)) 
   (=> (and (p ?a ?b) (p ?b ?c)) 
       (p ?a ?c))
))

(assert (not (p a1 a4)))

(check-sat)
(exit)

