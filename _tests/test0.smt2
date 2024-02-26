(set-info :smt-lib-version 2.6)
(set-logic UF)



;(declare-sort List 0)
;(declare-sort Elem 0)
;(declare-fun cons (Elem List) List)
;(declare-fun nil () List)
;(declare-fun car (List) Elem)
;(declare-fun cdr (List) List)
;(assert (forall ((?x Elem) (?y List)) (= (car (cons ?x ?y)) ?x)))
;(assert (forall ((?x Elem) (?y List)) (= (cdr (cons ?x ?y)) ?y)))
;(assert (not (forall ((?x1 Elem) (?y1 List) (?x2 Elem) (?y2 List)) (=> (= (cons ?x1 ?y1) (cons ?x2 ?y2)) (and (= ?x1 ?x2) (= ?y1 ?y2))))))


(declare-sort Any 0)
(declare-fun p (Any) Bool)

(assert (not (forall ((?x Any)) (=> (p ?x) (p ?x)))))

(check-sat)
(exit)
