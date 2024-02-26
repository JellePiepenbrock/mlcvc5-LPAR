(set-option :incremental false)
(set-info :source "Generating minimum transitivity constraints in P-time for deciding Equality Logic,
Ofer Strichman and Mirron Rozanov,
SMT Workshop 2005.

Translator: Leonardo de Moura.")
(set-info :status unsat)
(set-info :category "crafted")
(set-info :difficulty "0")
(set-logic QF_BV)
(declare-fun x0 () (_ BitVec 32))
(declare-fun y0 () (_ BitVec 32))
(declare-fun z0 () (_ BitVec 32))
(declare-fun x1 () (_ BitVec 32))
(declare-fun y1 () (_ BitVec 32))
(declare-fun z1 () (_ BitVec 32))
(declare-fun x2 () (_ BitVec 32))
(declare-fun y2 () (_ BitVec 32))
(declare-fun z2 () (_ BitVec 32))
(declare-fun x3 () (_ BitVec 32))
(declare-fun y3 () (_ BitVec 32))
(declare-fun z3 () (_ BitVec 32))
(declare-fun x4 () (_ BitVec 32))
(declare-fun y4 () (_ BitVec 32))
(declare-fun z4 () (_ BitVec 32))
(declare-fun x5 () (_ BitVec 32))
(declare-fun y5 () (_ BitVec 32))
(declare-fun z5 () (_ BitVec 32))
(declare-fun x6 () (_ BitVec 32))
(declare-fun y6 () (_ BitVec 32))
(declare-fun z6 () (_ BitVec 32))
(declare-fun x7 () (_ BitVec 32))
(declare-fun y7 () (_ BitVec 32))
(declare-fun z7 () (_ BitVec 32))
(declare-fun x8 () (_ BitVec 32))
(declare-fun y8 () (_ BitVec 32))
(declare-fun z8 () (_ BitVec 32))
(declare-fun x9 () (_ BitVec 32))
(declare-fun y9 () (_ BitVec 32))
(declare-fun z9 () (_ BitVec 32))
(declare-fun x10 () (_ BitVec 32))
(declare-fun y10 () (_ BitVec 32))
(declare-fun z10 () (_ BitVec 32))
(declare-fun x11 () (_ BitVec 32))
(declare-fun y11 () (_ BitVec 32))
(declare-fun z11 () (_ BitVec 32))
(check-sat-assuming ( (and (or (and (= x0 y0) (= y0 x1)) (and (= x0 z0) (= z0 x1))) (or (and (= x1 y1) (= y1 x2)) (and (= x1 z1) (= z1 x2))) (or (and (= x2 y2) (= y2 x3)) (and (= x2 z2) (= z2 x3))) (or (and (= x3 y3) (= y3 x4)) (and (= x3 z3) (= z3 x4))) (or (and (= x4 y4) (= y4 x5)) (and (= x4 z4) (= z4 x5))) (or (and (= x5 y5) (= y5 x6)) (and (= x5 z5) (= z5 x6))) (or (and (= x6 y6) (= y6 x7)) (and (= x6 z6) (= z6 x7))) (or (and (= x7 y7) (= y7 x8)) (and (= x7 z7) (= z7 x8))) (or (and (= x8 y8) (= y8 x9)) (and (= x8 z8) (= z8 x9))) (or (and (= x9 y9) (= y9 x10)) (and (= x9 z9) (= z9 x10))) (or (and (= x10 y10) (= y10 x11)) (and (= x10 z10) (= z10 x11))) (not (= x0 x11))) ))
