#!/bin/bash

# --ho-elim --full-saturate-quant
# --ho-elim --no-e-matching --full-saturate-quant
# --ho-elim --no-e-matching --fs-sum --full-saturate-quant
# --ho-elim --finite-model-find --uf-ss=no-minimal
# --no-ho-matching --finite-model-find --uf-ss=no-minimal
# --no-ho-matching --full-saturate-quant --fs-interleave --ho-elim-store-ax
# --no-ho-matching --full-saturate-quant --macros-quant-mode=all
# --ho-elim --full-saturate-quant --fs-interleave
# --no-ho-matching --full-saturate-quant --ho-elim-store-ax
# --ho-elim --no-ho-elim-store-ax --full-saturate-quant

#./cvc5 --track-relevant-literals --no-e-matching --full-saturate-quant $@ # test3.smt2 --dump-instantiations
# cvc5 --track-relevant-literals --e-matching --full-saturate-quant $@ # test3.smt2 --dump-instantiations

COMMON="--produce-proofs --no-e-matching --full-saturate-quant --dump-instantiations --print-inst-full --stats --stats-internal"

#TAGS="--trace featurize --trace inst-alg-rd --trace quant"
TAGS="--trace quant --trace ml --trace inst-alg --trace enum-engine --trace quant-sampler --trace active-fmls"
#TAGS="--trace quant --trace ml --trace inst-alg --trace enum-engine --trace quant-sampler"
TAGS=""

cvc5 --no-cegqi --no-cbqi --qlogging --track-relevant-literals $COMMON $TAGS $@
#cvc5 --no-cegqi --produce-proofs --no-e-matching --full-saturate-quant --dump-instantiations --qlogging $TAGS $@
#cvc5 --produce-proofs --no-e-matching --full-saturate-quant --dump-instantiations --qlogging $TAGS $@

