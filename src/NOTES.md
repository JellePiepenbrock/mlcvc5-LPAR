### References of `s_logger`

`api/cpp/cvc5.cpp`
`smt/process_assertions.cpp`
`smt/smt_engine.cpp`
`theory/quantifiers/instantiate.cpp`
`theory/quantifiers/inst_strategy_enumerative.cpp`
`theory/quantifiers/term_registry.cpp`
`theory/quantifiers/term_tuple_enumerator.cpp`
`theory/quantifiers/term_tuple_enumerator_ml.cpp`

### Mikolas' files

`theory/quantifiers/embedding_matrices.cpp`
`theory/quantifiers/embedding_matrices.h`
`theory/quantifiers/featurize.cpp`
`theory/quantifiers/featurize.h`
`theory/quantifiers/ml.cpp`
`theory/quantifiers/ml.h`
`theory/quantifiers/quantifier_logger.cpp`
`theory/quantifiers/quantifier_logger.h`
`theory/quantifiers/quantifier_sampler.cpp`
`theory/quantifiers/quantifier_sampler.h`
`theory/quantifiers/term_tuple_enumerator_ml.cpp`
`theory/quantifiers/term_tuple_enumerator_ml.h`
`theory/quantifiers/term_tuple_enumerator_utils.cpp`
`theory/quantifiers/term_tuple_enumerator_utils.h`

### Calls

`api/cpp/cvc5.cpp`:                                   `s_logger->print(std::cout);`
`smt/process_assertions.cpp`:                         `s_logger->recordAssertions(assertions);`
`smt/smt_engine.cpp`:                                 `s_logger->registerUsefulInstantiation(`
`theory/quantifiers/instantiate.cpp`:                 `s_logger->registerCurrentInstantiationBody(q,`
`theory/quantifiers/inst_strategy_enumerative.cpp`:   `s_logger->assertionsBOW());`
`theory/quantifiers/inst_strategy_enumerative.cpp`:   `s_logger->d_qselFeatureProperties.get());`
`theory/quantifiers/inst_strategy_enumerative.cpp`:   `s_logger->d_qselFeatureProperties->initialize();`
`theory/quantifiers/inst_strategy_enumerative.cpp`:   `s_logger->d_qselFeatureProperties.reset(`
`theory/quantifiers/inst_strategy_enumerative.cpp`:   `s_logger->d_quantifierFeatureProperties->initialize();`
`theory/quantifiers/inst_strategy_enumerative.cpp`:   `s_logger->d_quantifierFeatureProperties.reset(`
`theory/quantifiers/inst_strategy_enumerative.cpp`:   `s_logger->d_symbolContexts.lockExisting();`
`theory/quantifiers/inst_strategy_enumerative.cpp`:   `s_logger->d_termFeatureProperties);`
`theory/quantifiers/inst_strategy_enumerative.cpp`:   `s_logger->d_termFeatureProperties->initialize();`
`theory/quantifiers/inst_strategy_enumerative.cpp`:   `s_logger->d_termFeatureProperties.reset(`
`theory/quantifiers/inst_strategy_enumerative.cpp`:   `s_logger->d_tupleFeatureProperties->initialize();`
`theory/quantifiers/inst_strategy_enumerative.cpp`:   `s_logger->d_tupleFeatureProperties.reset(`
`theory/quantifiers/inst_strategy_enumerative.cpp`:   `s_logger->registerInstantiation(`
`theory/quantifiers/inst_strategy_enumerative.cpp`:   `s_logger->registerInstantiationAttempt(`
`theory/quantifiers/inst_strategy_enumerative.cpp`:   `s_logger->registerInstantiationRound(d_treg);`
`theory/quantifiers/term_registry.cpp`:               `s_logger->d_symbolContexts.observeTerm(n);`
`theory/quantifiers/term_tuple_enumerator.cpp`:       `s_logger->increasePhase(d_quantifier);`
`theory/quantifiers/term_tuple_enumerator.cpp`:       `s_logger->registerCandidate(`
`theory/quantifiers/term_tuple_enumerator_ml.cpp`:    `s_logger->d_termFeatureProperties.get()))`
`theory/quantifiers/term_tuple_enumerator_ml.cpp`:    `s_logger->d_termFeatureProperties.get())));`
`theory/quantifiers/term_tuple_enumerator_ml.cpp`:    `s_logger->d_tupleFeatureProperties.get());`
`theory/quantifiers/term_tuple_enumerator_ml.cpp`:    `s_logger->getQuantifierInfo(d_quantifier);`
`theory/quantifiers/term_tuple_enumerator_ml.cpp`:    `s_logger->hasQuantifier(d_quantifier))`

### Calls only

[ ] `s_logger->assertionsBOW`
[x] `s_logger->getQuantifierInfo`
[x] `s_logger->hasQuantifier`
[x] `s_logger->increasePhase`
[x] `s_logger->print`
[x] `s_logger->recordAssertions`
[x] `s_logger->registerCandidate`
[x] `s_logger->registerCurrentInstantiationBody`
[x] `s_logger->registerInstantiation`
[x] `s_logger->registerInstantiationAttempt`
[x] `s_logger->registerInstantiationRound`
[x] `s_logger->registerUsefulInstantiation`

### Properties

[ ] `s_logger->d_qselFeatureProperties.get`
[ ] `s_logger->d_qselFeatureProperties->initialize`
[ ] `s_logger->d_qselFeatureProperties.reset`
[ ] `s_logger->d_quantifierFeatureProperties->initialize`
[ ] `s_logger->d_quantifierFeatureProperties.reset`
[ ] `s_logger->d_symbolContexts`
[ ] `s_logger->d_symbolContexts.lockExisting`
[ ] `s_logger->d_symbolContexts.observeTerm`
[ ] `s_logger->d_termFeatureProperties`
[ ] `s_logger->d_termFeatureProperties.get`
[ ] `s_logger->d_termFeatureProperties->initialize`
[ ] `s_logger->d_termFeatureProperties.reset`
[ ] `s_logger->d_tupleFeatureProperties.get`
[ ] `s_logger->d_tupleFeatureProperties->initialize`
[ ] `s_logger->d_tupleFeatureProperties.reset`
