#!/bin/sh

TAGS="--trace quant --trace ml --trace inst-alg --trace enum-engine --trace quant-sampler"

cvc5 --no-cegqi --no-cbqi --qlogging --lightGBModelQuantifiers=model.lgb --qsel-fallback --qsel-epsilon=0.1 --qsel-formula-context --produce-proofs --no-e-matching --full-saturate-quant --dump-instantiations --ml-boost-actives --track-relevant-literals --qlogging-active $TAGS bug1.smt2

