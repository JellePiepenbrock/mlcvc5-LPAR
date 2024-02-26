(set-info :smt-lib-version 2.6)
(set-logic UFLIA)

(declare-fun p (Int Int) Bool)

(assert (forall ((?n Int))
   (p ?n (+ ?n 1))
))

(assert (forall ((?a Int) (?b Int) (?c Int)) 
   (=> (and (p ?a ?b) (p ?b ?c)) 
       (p ?a ?c))
))

(assert (not 
   (forall ((?m Int))
      (exists ((?n Int)) 
         (and (> ?n (+ ?m 5))
              (p ?m ?n)
         )
      )
   )
))

(check-sat)
(exit)

