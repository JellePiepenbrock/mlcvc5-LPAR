(set-option :incremental false)
(set-info :status sat)
(set-logic QF_BV)
(declare-fun v0 () (_ BitVec 4))
(declare-fun v1 () (_ BitVec 4))
(declare-fun v2 () (_ BitVec 4))
(declare-fun v3 () (_ BitVec 4))
(declare-fun v4 () (_ BitVec 4))
(check-sat-assuming ( (let ((_let_0 (bvnor v2 v2))) (let ((_let_1 (bvnor v2 v0))) (let ((_let_2 (bvnor v1 (bvmul (_ bv15 4) (_ bv15 4))))) (let ((_let_3 (bvxnor (_ bv13 4) v4))) (let ((_let_4 (ite (bvsge v2 (bvmul v0 _let_2)) (_ bv1 1) (_ bv0 1)))) (let ((_let_5 (bvnot _let_1))) (let ((_let_6 (bvadd (_ bv13 4) (bvadd _let_2 v0)))) (let ((_let_7 (ite (bvuge ((_ sign_extend 3) (ite (bvslt (bvadd _let_2 v0) _let_1) (_ bv1 1) (_ bv0 1))) (_ bv15 4)) (_ bv1 1) (_ bv0 1)))) (let ((_let_8 (bvnand (bvmul (_ bv15 4) (_ bv15 4)) _let_3))) (let ((_let_9 (ite (= v2 v0) (_ bv1 1) (_ bv0 1)))) (let ((_let_10 ((_ zero_extend 0) (bvashr _let_0 v0)))) (let ((_let_11 ((_ repeat 1) (ite (= (_ bv1 1) ((_ extract 0 0) v4)) _let_2 ((_ sign_extend 3) (ite (= ((_ sign_extend 3) (ite (bvuge (bvshl v3 v3) v0) (_ bv1 1) (_ bv0 1))) v2) (_ bv1 1) (_ bv0 1))))))) (let ((_let_12 ((_ sign_extend 3) (ite (bvuge v3 (bvmul v0 _let_2)) (_ bv1 1) (_ bv0 1))))) (let ((_let_13 ((_ zero_extend 0) (ite (= (_ bv1 1) ((_ extract 0 0) v4)) _let_2 ((_ sign_extend 3) (ite (= ((_ sign_extend 3) (ite (bvuge (bvshl v3 v3) v0) (_ bv1 1) (_ bv0 1))) v2) (_ bv1 1) (_ bv0 1))))))) (let ((_let_14 (bvxnor (bvxnor (_ bv13 4) (bvshl v3 v3)) (ite (= (_ bv1 1) ((_ extract 0 0) v4)) _let_2 ((_ sign_extend 3) (ite (= ((_ sign_extend 3) (ite (bvuge (bvshl v3 v3) v0) (_ bv1 1) (_ bv0 1))) v2) (_ bv1 1) (_ bv0 1))))))) (let ((_let_15 (bvashr ((_ sign_extend 3) (ite (bvugt (ite (= ((_ sign_extend 3) (ite (bvuge (bvshl v3 v3) v0) (_ bv1 1) (_ bv0 1))) v2) (_ bv1 1) (_ bv0 1)) (ite (= ((_ sign_extend 3) _let_4) _let_0) (_ bv1 1) (_ bv0 1))) (_ bv1 1) (_ bv0 1))) (bvashr _let_0 v0)))) (let ((_let_16 ((_ rotate_left 0) (ite (= ((_ sign_extend 3) (ite (bvuge (bvshl v3 v3) v0) (_ bv1 1) (_ bv0 1))) v2) (_ bv1 1) (_ bv0 1))))) (let ((_let_17 (bvadd _let_1 ((_ zero_extend 3) _let_4)))) (let ((_let_18 (bvadd ((_ sign_extend 3) (ite (bvslt (bvadd _let_2 v0) _let_1) (_ bv1 1) (_ bv0 1))) (bvmul (bvashr (_ bv11 4) ((_ sign_extend 3) (ite (bvugt (_ bv13 4) (_ bv15 4)) (_ bv1 1) (_ bv0 1)))) (bvmul v0 _let_2))))) (let ((_let_19 (bvnor ((_ sign_extend 3) (ite (bvule (bvshl v3 v3) v4) (_ bv1 1) (_ bv0 1))) ((_ sign_extend 0) _let_10)))) (let ((_let_20 (bvmul v0 v3))) (let ((_let_21 (bvor _let_15 (_ bv15 4)))) (let ((_let_22 ((_ zero_extend 0) (bvmul (bvashr (_ bv11 4) ((_ sign_extend 3) (ite (bvugt (_ bv13 4) (_ bv15 4)) (_ bv1 1) (_ bv0 1)))) (bvmul v0 _let_2))))) (let ((_let_23 ((_ repeat 1) (_ bv15 4)))) (let ((_let_24 (bvxor _let_18 v3))) (let ((_let_25 (bvcomp _let_23 _let_3))) (let ((_let_26 (ite (bvugt (ite (= (_ bv1 1) ((_ extract 0 0) v4)) _let_2 ((_ sign_extend 3) (ite (= ((_ sign_extend 3) (ite (bvuge (bvshl v3 v3) v0) (_ bv1 1) (_ bv0 1))) v2) (_ bv1 1) (_ bv0 1)))) _let_22) (_ bv1 1) (_ bv0 1)))) (let ((_let_27 (bvxnor _let_21 _let_20))) (let ((_let_28 (ite (bvsle _let_18 (ite (= (_ bv1 1) ((_ extract 0 0) v4)) _let_2 ((_ sign_extend 3) (ite (= ((_ sign_extend 3) (ite (bvuge (bvshl v3 v3) v0) (_ bv1 1) (_ bv0 1))) v2) (_ bv1 1) (_ bv0 1))))) (_ bv1 1) (_ bv0 1)))) (let ((_let_29 (bvshl _let_1 _let_24))) (let ((_let_30 (ite (bvult (bvmul (_ bv15 4) (_ bv15 4)) _let_13) (_ bv1 1) (_ bv0 1)))) (let ((_let_31 (bvor _let_22 ((_ zero_extend 3) (ite (bvsle _let_15 _let_5) (_ bv1 1) (_ bv0 1)))))) (let ((_let_32 (bvule ((_ zero_extend 3) _let_30) _let_2))) (let ((_let_33 (bvsle _let_15 (bvmul v0 _let_2)))) (let ((_let_34 ((_ zero_extend 3) (ite (distinct (bvmul (_ bv15 4) (_ bv15 4)) _let_12) (_ bv1 1) (_ bv0 1))))) (let ((_let_35 (bvuge _let_3 _let_34))) (let ((_let_36 (bvslt _let_0 _let_10))) (let ((_let_37 (bvsge _let_11 ((_ zero_extend 3) (ite (bvslt (bvadd _let_2 v0) _let_1) (_ bv1 1) (_ bv0 1)))))) (let ((_let_38 (distinct _let_24 (ite (= (_ bv1 1) ((_ extract 2 2) v4)) v1 v4)))) (let ((_let_39 ((_ zero_extend 3) (ite (= ((_ sign_extend 3) (ite (bvuge (bvshl v3 v3) v0) (_ bv1 1) (_ bv0 1))) v2) (_ bv1 1) (_ bv0 1))))) (let ((_let_40 (bvslt (bvashr _let_0 v0) _let_6))) (let ((_let_41 ((_ sign_extend 3) (ite (distinct (bvmul (_ bv15 4) (_ bv15 4)) _let_12) (_ bv1 1) (_ bv0 1))))) (let ((_let_42 (bvuge (_ bv11 4) _let_24))) (let ((_let_43 (bvsgt _let_31 ((_ zero_extend 3) (ite (bvugt (ite (= ((_ sign_extend 3) (ite (bvuge (bvshl v3 v3) v0) (_ bv1 1) (_ bv0 1))) v2) (_ bv1 1) (_ bv0 1)) (ite (= ((_ sign_extend 3) _let_4) _let_0) (_ bv1 1) (_ bv0 1))) (_ bv1 1) (_ bv0 1)))))) (let ((_let_44 (bvsle (_ bv13 4) (bvmul v0 _let_2)))) (let ((_let_45 ((_ zero_extend 3) _let_25))) (let ((_let_46 (bvult _let_3 _let_21))) (let ((_let_47 ((_ zero_extend 3) (ite (bvugt (_ bv13 4) (_ bv15 4)) (_ bv1 1) (_ bv0 1))))) (let ((_let_48 ((_ zero_extend 3) _let_26))) (let ((_let_49 (distinct _let_22 _let_48))) (let ((_let_50 (bvsgt ((_ sign_extend 3) (ite (= ((_ sign_extend 3) (ite (bvuge (bvshl v3 v3) v0) (_ bv1 1) (_ bv0 1))) v2) (_ bv1 1) (_ bv0 1))) _let_19))) (let ((_let_51 (bvule (ite (bvugt (ite (= ((_ sign_extend 3) (ite (bvuge (bvshl v3 v3) v0) (_ bv1 1) (_ bv0 1))) v2) (_ bv1 1) (_ bv0 1)) (ite (= ((_ sign_extend 3) _let_4) _let_0) (_ bv1 1) (_ bv0 1))) (_ bv1 1) (_ bv0 1)) (ite (bvuge v3 (bvmul v0 _let_2)) (_ bv1 1) (_ bv0 1))))) (let ((_let_52 (bvsge _let_5 (bvmul v0 _let_2)))) (let ((_let_53 (bvsle ((_ sign_extend 3) _let_16) _let_8))) (let ((_let_54 (= _let_24 ((_ sign_extend 3) (ite (bvugt (ite (= ((_ sign_extend 3) (ite (bvuge (bvshl v3 v3) v0) (_ bv1 1) (_ bv0 1))) v2) (_ bv1 1) (_ bv0 1)) (ite (= ((_ sign_extend 3) _let_4) _let_0) (_ bv1 1) (_ bv0 1))) (_ bv1 1) (_ bv0 1)))))) (let ((_let_55 (bvule _let_29 ((_ zero_extend 3) (ite (bvuge (bvshl v3 v3) v0) (_ bv1 1) (_ bv0 1)))))) (let ((_let_56 (bvuge _let_1 (_ bv13 4)))) (let ((_let_57 (distinct _let_19 (bvadd _let_2 v0)))) (let ((_let_58 (bvsgt _let_6 (bvnor v0 _let_12)))) (let ((_let_59 (bvugt (ite (= (_ bv1 1) ((_ extract 0 0) v4)) _let_2 ((_ sign_extend 3) (ite (= ((_ sign_extend 3) (ite (bvuge (bvshl v3 v3) v0) (_ bv1 1) (_ bv0 1))) v2) (_ bv1 1) (_ bv0 1)))) _let_3))) (let ((_let_60 ((_ zero_extend 3) _let_16))) (let ((_let_61 (bvuge (_ bv13 4) _let_2))) (let ((_let_62 (bvsgt _let_0 ((_ sign_extend 3) _let_9)))) (let ((_let_63 (bvuge (bvashr _let_0 v0) (ite (= (_ bv1 1) ((_ extract 2 2) v4)) v1 v4)))) (let ((_let_64 (bvsle ((_ zero_extend 3) _let_28) _let_29))) (let ((_let_65 (distinct _let_1 ((_ sign_extend 3) _let_28)))) (let ((_let_66 (bvule (_ bv11 4) v0))) (let ((_let_67 (bvule ((_ sign_extend 3) (ite (bvule ((_ sign_extend 1) _let_4) ((_ sign_extend 1) (ite (bvuge (bvshl v3 v3) v0) (_ bv1 1) (_ bv0 1)))) (_ bv1 1) (_ bv0 1))) v1))) (let ((_let_68 (bvsle _let_24 (bvadd _let_2 v0)))) (let ((_let_69 (bvule (ite (= (_ bv1 1) ((_ extract 2 2) v4)) v1 v4) _let_3))) (let ((_let_70 (bvult ((_ zero_extend 3) (ite (bvuge (bvshl v3 v3) v0) (_ bv1 1) (_ bv0 1))) (bvashr _let_0 v0)))) (let ((_let_71 (bvule ((_ zero_extend 3) (ite (bvule (bvshl v3 v3) v4) (_ bv1 1) (_ bv0 1))) _let_8))) (let ((_let_72 (bvsge (bvshl v3 v3) _let_0))) (let ((_let_73 (bvugt _let_20 ((_ repeat 1) _let_1)))) (let ((_let_74 (bvsle v0 _let_60))) (let ((_let_75 (bvugt (bvadd _let_2 v0) ((_ sign_extend 3) (ite (bvsle _let_15 _let_6) (_ bv1 1) (_ bv0 1)))))) (let ((_let_76 (not (bvuge _let_13 _let_6)))) (let ((_let_77 (not _let_68))) (let ((_let_78 (not (bvugt (bvshl v3 v3) _let_20)))) (let ((_let_79 (not (bvslt (ite (bvuge v3 (bvmul v0 _let_2)) (_ bv1 1) (_ bv0 1)) _let_7)))) (let ((_let_80 (not _let_57))) (let ((_let_81 (not _let_69))) (let ((_let_82 (not (bvsge _let_18 v2)))) (let ((_let_83 (not (bvugt (ite (= (_ bv1 1) ((_ extract 0 0) v4)) _let_2 ((_ sign_extend 3) (ite (= ((_ sign_extend 3) (ite (bvuge (bvshl v3 v3) v0) (_ bv1 1) (_ bv0 1))) v2) (_ bv1 1) (_ bv0 1)))) ((_ sign_extend 3) (ite (bvuge (bvshl v3 v3) v0) (_ bv1 1) (_ bv0 1))))))) (let ((_let_84 (not (bvult _let_22 _let_17)))) (let ((_let_85 (not _let_66))) (let ((_let_86 (not _let_56))) (let ((_let_87 (not (bvsge (bvmul (bvashr (_ bv11 4) ((_ sign_extend 3) (ite (bvugt (_ bv13 4) (_ bv15 4)) (_ bv1 1) (_ bv0 1)))) (bvmul v0 _let_2)) _let_27)))) (let ((_let_88 (not (bvsge (ite (= _let_7 (ite (= ((_ sign_extend 3) (ite (bvuge (bvshl v3 v3) v0) (_ bv1 1) (_ bv0 1))) v2) (_ bv1 1) (_ bv0 1))) (_ bv1 1) (_ bv0 1)) (ite (bvule ((_ sign_extend 1) _let_4) ((_ sign_extend 1) (ite (bvuge (bvshl v3 v3) v0) (_ bv1 1) (_ bv0 1)))) (_ bv1 1) (_ bv0 1)))))) (let ((_let_89 (not _let_50))) (let ((_let_90 (not (bvsge _let_28 (ite (= _let_10 (_ bv11 4)) (_ bv1 1) (_ bv0 1)))))) (let ((_let_91 (not (bvult _let_39 ((_ sign_extend 0) _let_10))))) (let ((_let_92 (not _let_54))) (let ((_let_93 (not (bvult _let_41 _let_24)))) (let ((_let_94 (not (bvult ((_ zero_extend 3) (ite (bvsle _let_15 _let_5) (_ bv1 1) (_ bv0 1))) (bvashr _let_0 v0))))) (let ((_let_95 (not (bvsgt (ite (bvule ((_ sign_extend 1) _let_4) ((_ sign_extend 1) (ite (bvuge (bvshl v3 v3) v0) (_ bv1 1) (_ bv0 1)))) (_ bv1 1) (_ bv0 1)) _let_26)))) (let ((_let_96 (not _let_36))) (let ((_let_97 (not (distinct (bvashr _let_0 v0) v3)))) (let ((_let_98 (not (= ((_ sign_extend 3) (ite (bvule (bvshl v3 v3) v4) (_ bv1 1) (_ bv0 1))) (bvmul (_ bv15 4) (_ bv15 4)))))) (and (or (bvuge _let_19 _let_31) (bvult _let_41 _let_24) _let_76) (or (not (bvsgt (bvadd _let_2 v0) _let_48)) (not (bvuge (bvnor v0 _let_12) _let_10)) _let_77) (or _let_78 (bvslt ((_ repeat 1) _let_1) _let_47) _let_79) (or _let_74 _let_53 (not (bvslt _let_10 _let_31))) (or _let_80 (not (bvult (_ bv11 4) (bvmul v0 _let_2))) (bvult _let_39 ((_ sign_extend 0) _let_10))) (or _let_81 _let_49 _let_44) (or _let_54 (not _let_35) _let_63) (or _let_75 _let_71 (= ((_ sign_extend 3) (ite (bvule (bvshl v3 v3) v4) (_ bv1 1) (_ bv0 1))) (bvmul (_ bv15 4) (_ bv15 4)))) (or (not _let_52) (not _let_55) (distinct _let_12 _let_5)) (or _let_69 (bvult ((_ zero_extend 3) (ite (bvsle _let_15 _let_5) (_ bv1 1) (_ bv0 1))) (bvashr _let_0 v0)) (not (bvslt _let_20 _let_41))) (or _let_59 _let_65 (bvuge ((_ sign_extend 3) _let_25) _let_21)) (or _let_82 _let_83 (distinct (bvashr _let_0 v0) v3)) (or _let_42 (not (distinct ((_ sign_extend 3) (ite (= ((_ sign_extend 3) _let_4) _let_0) (_ bv1 1) (_ bv0 1))) (bvmul (_ bv15 4) (_ bv15 4)))) (bvsle _let_45 (bvashr (_ bv11 4) ((_ sign_extend 3) (ite (bvugt (_ bv13 4) (_ bv15 4)) (_ bv1 1) (_ bv0 1)))))) (or _let_62 _let_56 (bvult _let_22 _let_17)) (or _let_72 (bvuge _let_27 ((_ sign_extend 0) _let_10)) (not (bvsge ((_ sign_extend 3) (ite (bvugt (_ bv13 4) (_ bv15 4)) (_ bv1 1) (_ bv0 1))) _let_8))) (or (not _let_32) _let_55 _let_84) (or (not _let_70) _let_70 _let_57) (or (not (bvugt (_ bv11 4) (_ bv13 4))) (bvule _let_21 _let_1) (not (bvsle _let_5 (ite (= (_ bv1 1) ((_ extract 0 0) v4)) _let_2 ((_ sign_extend 3) (ite (= ((_ sign_extend 3) (ite (bvuge (bvshl v3 v3) v0) (_ bv1 1) (_ bv0 1))) v2) (_ bv1 1) (_ bv0 1))))))) (or (not _let_73) (distinct _let_13 _let_3) _let_38) (or (distinct _let_15 _let_39) (bvugt _let_4 _let_30) (not (bvule ((_ sign_extend 0) _let_10) v3))) (or _let_67 (not _let_40) (not (bvugt _let_45 _let_17))) (or (not _let_64) _let_63 _let_73) (or (bvsle (ite (= (_ bv1 1) ((_ extract 0 0) v4)) _let_2 ((_ sign_extend 3) (ite (= ((_ sign_extend 3) (ite (bvuge (bvshl v3 v3) v0) (_ bv1 1) (_ bv0 1))) v2) (_ bv1 1) (_ bv0 1)))) (ite (= (_ bv1 1) ((_ extract 2 2) v4)) v1 v4)) _let_84 (not _let_71)) (or _let_85 _let_46 (not _let_67)) (or _let_44 _let_33 _let_86) (or (not (bvsle v4 v1)) _let_87 _let_46) (or _let_88 _let_89 (not (bvult _let_0 v1))) (or _let_80 _let_40 _let_90) (or _let_91 _let_88 (bvugt _let_24 ((_ zero_extend 3) (ite (bvslt (bvadd _let_2 v0) _let_1) (_ bv1 1) (_ bv0 1))))) (or (distinct ((_ zero_extend 3) (ite (bvslt (bvadd _let_2 v0) _let_1) (_ bv1 1) (_ bv0 1))) _let_0) _let_85 _let_37) (or (not (distinct _let_47 _let_22)) _let_92 (= _let_2 _let_5)) (or (not (bvule v4 ((_ sign_extend 3) (ite (= ((_ sign_extend 3) (ite (bvuge (bvshl v3 v3) v0) (_ bv1 1) (_ bv0 1))) v2) (_ bv1 1) (_ bv0 1))))) (distinct (ite (= (_ bv1 1) ((_ extract 2 2) v4)) v1 v4) ((_ zero_extend 3) _let_7)) _let_32) (or (bvuge ((_ sign_extend 3) (ite (bvsle _let_15 _let_5) (_ bv1 1) (_ bv0 1))) _let_29) _let_93 (not (bvsle (_ bv15 4) (ite (= (_ bv1 1) ((_ extract 2 2) v4)) v1 v4)))) (or _let_77 _let_81 _let_36) (or _let_94 (not (= v3 _let_13)) _let_93) (or (not (bvult _let_47 _let_13)) (not _let_65) (not _let_38)) (or _let_94 _let_95 _let_42) (or (not (distinct _let_12 (bvmul (bvashr (_ bv11 4) ((_ sign_extend 3) (ite (bvugt (_ bv13 4) (_ bv15 4)) (_ bv1 1) (_ bv0 1)))) (bvmul v0 _let_2)))) _let_68 _let_64) (or _let_83 _let_96 _let_96) (or _let_52 _let_91 _let_74) (or (not _let_51) (not (bvsle _let_13 _let_48)) _let_51) (or (not _let_33) _let_56 _let_78) (or _let_89 _let_82 (bvsgt _let_11 _let_2)) (or _let_50 _let_44 (bvsgt (ite (= (_ bv1 1) ((_ extract 2 2) v4)) v1 v4) _let_24)) (or _let_42 _let_87 (not (bvsle _let_29 (bvashr (_ bv11 4) ((_ sign_extend 3) (ite (bvugt (_ bv13 4) (_ bv15 4)) (_ bv1 1) (_ bv0 1))))))) (or _let_89 _let_68 _let_43) (or _let_66 _let_35 _let_79) (or _let_94 _let_40 (bvult _let_5 _let_27)) (or (not (bvult v2 _let_39)) (not _let_58) _let_85) (or _let_92 _let_37 _let_58) (or (distinct (bvmul (_ bv15 4) (_ bv15 4)) (_ bv13 4)) _let_74 (bvult (_ bv11 4) v2)) (or _let_53 (bvsle (_ bv13 4) _let_41) (= _let_19 _let_60)) (or _let_63 (bvuge _let_31 _let_5) _let_54) (or _let_50 _let_72 _let_65) (or (not _let_43) _let_95 _let_32) (or (bvsgt v2 _let_23) _let_63 _let_59) (or _let_86 _let_97 (not (bvugt _let_14 _let_11))) (or (not (bvugt _let_4 _let_25)) _let_61 _let_97) (or _let_90 _let_92 _let_35) (or (not (bvsgt _let_27 _let_12)) (bvugt (bvmul (bvashr (_ bv11 4) ((_ sign_extend 3) (ite (bvugt (_ bv13 4) (_ bv15 4)) (_ bv1 1) (_ bv0 1)))) (bvmul v0 _let_2)) _let_34) _let_98) (or _let_73 _let_36 (not (bvult _let_14 v1))) (or _let_98 (bvule ((_ sign_extend 3) _let_30) (bvashr _let_0 v0)) _let_37) (or (not _let_62) (not _let_46) _let_76) (or _let_49 (not (bvsle _let_17 _let_12)) (not _let_61)) (or _let_75 (not (bvsgt _let_28 _let_9)) _let_64))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))))) ))