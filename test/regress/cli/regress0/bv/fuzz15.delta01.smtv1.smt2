; DISABLE-TESTER: lfsc
(set-option :incremental false)
(set-info :status unsat)
(set-logic QF_BV)
(declare-fun v11 () (_ BitVec 8))
(declare-fun v12 () (_ BitVec 8))
(declare-fun v10 () (_ BitVec 12))
(declare-fun v2 () (_ BitVec 10))
(declare-fun v8 () (_ BitVec 11))
(declare-fun v17 () (_ BitVec 8))
(declare-fun v5 () (_ BitVec 13))
(declare-fun v0 () (_ BitVec 15))
(declare-fun v14 () (_ BitVec 14))
(declare-fun v19 () (_ BitVec 10))
(check-sat-assuming ( (let ((_let_0 ((_ sign_extend 2) v11))) (let ((_let_1 ((_ zero_extend 2) v14))) (let ((_let_2 (ite (bvslt (_ bv0 12) v10) (_ bv1 1) (_ bv0 1)))) (let ((_let_3 (bvxnor ((_ sign_extend 4) ((_ extract 9 0) v0)) (_ bv6240 14)))) (let ((_let_4 ((_ zero_extend 2) v8))) (let ((_let_5 ((_ sign_extend 2) v17))) (let ((_let_6 ((_ sign_extend 6) v12))) (and (not (bvult ((_ sign_extend 6) _let_0) _let_1)) (not (bvugt (bvsub ((_ sign_extend 15) _let_2) ((_ zero_extend 2) (ite (= (_ bv1 1) (ite (bvslt (_ bv0 10) v19) (_ bv1 1) (_ bv0 1))) (bvxnor v14 (_ bv0 14)) _let_3))) (_ bv0 16))) (or false (not (bvugt (bvshl _let_4 ((_ zero_extend 12) (ite (bvsle v0 (_ bv0 15)) (_ bv1 1) (_ bv0 1)))) ((_ zero_extend 12) (ite (bvsge (_ bv0 16) _let_1) (_ bv1 1) (_ bv0 1))))) (bvsge ((_ sign_extend 14) (ite (distinct _let_1 (_ bv0 16)) (_ bv1 1) (_ bv0 1))) (_ bv0 15))) (or false (bvult ((_ sign_extend 14) (ite (distinct (_ bv0 13) ((_ zero_extend 3) _let_5)) (_ bv1 1) (_ bv0 1))) (bvnot (_ bv0 15))) (not (bvuge (_ bv0 16) ((_ sign_extend 2) v14)))) (bvugt (_ bv1 1) (ite (bvsgt _let_6 _let_3) (_ bv1 1) (_ bv0 1))) (not (bvule (bvmul (_ bv6240 14) ((_ zero_extend 1) _let_4)) ((_ zero_extend 13) (ite (bvsgt (_ bv0 13) v5) (_ bv1 1) (_ bv0 1))))) (= (bvxnor (_ bv0 14) ((_ sign_extend 4) v2)) ((_ zero_extend 2) ((_ zero_extend 2) _let_5))) (or false (bvsle (_ bv0 16) ((_ sign_extend 12) ((_ extract 6 3) _let_4))) (= (_ bv0 14) ((_ zero_extend 1) ((_ zero_extend 12) (ite (bvult (ite (= (_ bv1 1) ((_ extract 3 3) (bvxnor v2 ((_ sign_extend 9) _let_2)))) (_ bv0 14) _let_6) ((_ zero_extend 13) (bvnot (ite (bvugt _let_0 (_ bv0 10)) (_ bv1 1) (_ bv0 1))))) (_ bv1 1) (_ bv0 1)))))))))))))) ))
