(set-info :smt-lib-version 2.6)
(set-logic UFNIA)

(declare-fun p (Int Int) Bool)

(assert (forall ((?n Int))
   (p ?n (+ ?n 1))
))

;(assert (forall ((?a Int) (?b Int) (?c Int)) 
;   (=> (and (p ?a ?b) (p ?b ?c)) 
;       (p ?a ?c))
;))

(assert (not 
   (forall ((?n Int))
      (exists ((?m Int)) 
         (p ?m ?n)
      )
   )
))

(check-sat)
(exit)

