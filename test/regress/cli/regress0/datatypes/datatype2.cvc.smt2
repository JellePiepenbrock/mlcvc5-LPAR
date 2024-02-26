; EXPECT: unsat
(set-logic ALL)
(set-option :incremental false)
(declare-datatypes ((nat 0)(list 0)(tree 0)) (((succ (pred nat)) (zero))((cons (car tree) (cdr list)) (null))((node (data nat) (children list)) (leaf))))
(declare-fun x () nat)
(declare-fun y () list)
(declare-fun z () tree)
(assert (= x (succ zero)))
(assert (= z (ite ((_ is cons) y) (car y) (node x null))))
(check-sat-assuming ( (not (=> (not ((_ is cons) y)) (= (pred (data z)) zero))) ))