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

#TAGS="--trace featurize --trace inst-alg-rd --trace quant"
TAGS="--trace quant"

cvc5-mikolas --no-qcf-all-conflict --produce-proofs --no-e-matching --full-saturate-quant --dump-instantiations --qlogging $TAGS $@
#cvc5-mikolas --produce-proofs --no-e-matching --full-saturate-quant --dump-instantiations --qlogging $TAGS $@

