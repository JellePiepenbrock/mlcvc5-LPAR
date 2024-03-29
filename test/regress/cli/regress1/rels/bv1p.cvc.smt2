; COMMAND-LINE: --jh-rlv-order
; EXPECT: unsat
(set-option :incremental false)
(set-logic ALL)

(declare-fun x () (Relation (_ BitVec 1) (_ BitVec 1)))
(declare-fun y () (Relation (_ BitVec 1) (_ BitVec 1)))
(declare-fun a () (Tuple (_ BitVec 1) (_ BitVec 1)))
(declare-fun b () (Tuple (_ BitVec 1) (_ BitVec 1)))
(declare-fun c () (Tuple (_ BitVec 1) (_ BitVec 1)))
(declare-fun d () (Tuple (_ BitVec 1) (_ BitVec 1)))
(declare-fun e () (Tuple (_ BitVec 1) (_ BitVec 1)))
(assert (distinct a b))
(assert (distinct c d e))
(assert (set.member a x))
(assert (set.member b x))
(assert (set.member a y))
(assert (set.member b y))
(assert (let ((_let_1 (rel.join x y))) (and (and (not (set.member c _let_1)) (not (set.member d _let_1))) (not (set.member e _let_1)))))
(check-sat)
