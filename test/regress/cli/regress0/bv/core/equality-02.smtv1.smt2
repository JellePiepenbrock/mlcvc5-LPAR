(set-option :incremental false)
(set-info :status unsat)
(set-logic QF_BV)
(declare-fun x0 () (_ BitVec 32))
(declare-fun x1 () (_ BitVec 32))
(declare-fun x2 () (_ BitVec 32))
(declare-fun x3 () (_ BitVec 32))
(declare-fun y0 () (_ BitVec 32))
(declare-fun y1 () (_ BitVec 32))
(declare-fun y2 () (_ BitVec 32))
(declare-fun y3 () (_ BitVec 32))
(assert (= x0 x1))
(assert (= x1 x2))
(assert (= x2 x3))
(assert (= y0 y1))
(assert (= y1 y2))
(assert (= y2 y3))
(assert (= x0 y0))
(check-sat-assuming ( (not (= x3 y3)) ))
