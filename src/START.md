### Quick list

+ QuantifierLogger
    + `theory/quantifiers/quantifier_logger.cpp`
    + `theory/quantifiers/quantifier_logger.h`

+ `theory/quantifiers/featurize.h`
+ `theory/quantifiers/featurize.cpp`

+ `theory/quantifiers/ml.h`
+ `theory/quantifiers/ml.cpp`

+ `theory/quantifiers/inst_strategy_enumerative.h`
+ `theory/quantifiers/inst_strategy_enumerative.cpp`

### Mikolas' code:

+ `theory/quantifiers/embedding_matrices.cpp`
+ `theory/quantifiers/embedding_matrices.h`
+ `theory/quantifiers/featurize.cpp`
+ `theory/quantifiers/featurize.h`
+ `theory/quantifiers/ml.cpp`
+ `theory/quantifiers/ml.h`
+ `theory/quantifiers/quantifier_logger.cpp`
+ `theory/quantifiers/quantifier_logger.h`
+ `theory/quantifiers/quantifier_sampler.cpp`
+ `theory/quantifiers/quantifier_sampler.h`
+ `theory/quantifiers/term_tuple_enumerator_ml.cpp`
+ `theory/quantifiers/term_tuple_enumerator_ml.h`
+ `theory/quantifiers/term_tuple_enumerator_utils.cpp`
+ `theory/quantifiers/term_tuple_enumerator_utils.h`

mixed:

+ `theory/quantifiers/term_tuple_enumerator.cpp`
+ `theory/quantifiers/term_tuple_enumerator.h`

+ `theory/quantifiers/inst_strategy_enumerative.h`
+ `theory/quantifiers/inst_strategy_enumerative.cpp`
### 

+ `theory/quantifiers/term_tuple_enumerator.cpp`
+ `theory/quantifiers/term_tuple_enumerator.h`

### Mikolas' connections:

REPO: ~/repos/github/private_learning-MikolasJanota/src/theory/quantifiers

api/cpp/cvc5.cpp:    theory::quantifiers::QuantifierLogger::s_logger->print(std::cout);
smt/process_assertions.cpp:  theory::quantifiers::QuantifierLogger::s_logger->recordAssertions(assertions);
smt/smt_engine.cpp:  quantifiers::QuantifierLogger::s_logger =
smt/smt_engine.cpp:  quantifiers::QuantifierLogger::s_logger.reset(nullptr);
smt/smt_engine.cpp:        quantifiers::QuantifierLogger::s_logger->registerUsefulInstantiation(
theory/quantifiers/featurize.cpp:  const auto& ctxs = QuantifierLogger::s_logger->d_symbolContexts;
theory/quantifiers/featurize.cpp:  return QuantifierLogger::s_logger->d_matrices.get();
theory/quantifiers/inst_strategy_enumerative.cpp:    QuantifierLogger::s_logger->d_matrices =
theory/quantifiers/inst_strategy_enumerative.cpp:    QuantifierLogger::s_logger->d_matrices->loadTransformations(input_file);
theory/quantifiers/inst_strategy_enumerative.cpp:    const auto dimension = QuantifierLogger::s_logger->d_matrices->dimension();
theory/quantifiers/inst_strategy_enumerative.cpp:    Assert(!QuantifierLogger::s_logger->d_termFeatureProperties);
theory/quantifiers/inst_strategy_enumerative.cpp:    QuantifierLogger::s_logger->d_termFeatureProperties.reset(
theory/quantifiers/inst_strategy_enumerative.cpp:    QuantifierLogger::s_logger->d_termFeatureProperties->initialize();
theory/quantifiers/inst_strategy_enumerative.cpp:    Assert(!QuantifierLogger::s_logger->d_termFeatureProperties);
theory/quantifiers/inst_strategy_enumerative.cpp:    QuantifierLogger::s_logger->d_quantifierFeatureProperties.reset(
theory/quantifiers/inst_strategy_enumerative.cpp:    QuantifierLogger::s_logger->d_quantifierFeatureProperties->initialize();
theory/quantifiers/inst_strategy_enumerative.cpp:    QuantifierLogger::s_logger->d_qselFeatureProperties.reset(
theory/quantifiers/inst_strategy_enumerative.cpp:    QuantifierLogger::s_logger->d_qselFeatureProperties->initialize();
theory/quantifiers/inst_strategy_enumerative.cpp:    QuantifierLogger::s_logger->d_termFeatureProperties.reset(
theory/quantifiers/inst_strategy_enumerative.cpp:    QuantifierLogger::s_logger->d_termFeatureProperties->initialize();
theory/quantifiers/inst_strategy_enumerative.cpp:  QuantifierLogger::s_logger->d_tupleFeatureProperties.reset(
theory/quantifiers/inst_strategy_enumerative.cpp:  QuantifierLogger::s_logger->d_tupleFeatureProperties->initialize();
theory/quantifiers/inst_strategy_enumerative.cpp:    QuantifierLogger::s_logger->registerInstantiationRound(d_treg);
theory/quantifiers/inst_strategy_enumerative.cpp:      QuantifierLogger::s_logger->d_symbolContexts.lockExisting();
theory/quantifiers/inst_strategy_enumerative.cpp:        QuantifierLogger::s_logger->d_qselFeatureProperties.get());
theory/quantifiers/inst_strategy_enumerative.cpp:      featurizer.addContext(QuantifierLogger::s_logger->assertionsBOW());
theory/quantifiers/inst_strategy_enumerative.cpp:      completedTerms = QuantifierLogger::s_logger->registerInstantiationAttempt(
theory/quantifiers/inst_strategy_enumerative.cpp:      QuantifierLogger::s_logger->registerInstantiation(
theory/quantifiers/instantiate.cpp:    QuantifierLogger::s_logger->registerCurrentInstantiationBody(q, lem);
theory/quantifiers/quantifier_logger.cpp:std::unique_ptr<QuantifierLogger> QuantifierLogger::s_logger;
theory/quantifiers/quantifier_logger.cpp:  const bool useMatrices{QuantifierLogger::s_logger->d_matrices.get()
theory/quantifiers/quantifier_logger.h:      s_logger;  // TODO: get rid of singleton
theory/quantifiers/term_registry.cpp:    QuantifierLogger::s_logger->d_symbolContexts.observeTerm(n);
theory/quantifiers/term_tuple_enumerator.cpp:               || QuantifierLogger::s_logger->d_matrices);
theory/quantifiers/term_tuple_enumerator.cpp:        QuantifierLogger::s_logger->d_matrices->d_modelFile,
theory/quantifiers/term_tuple_enumerator.cpp:        QuantifierLogger::s_logger->d_matrices.get());
theory/quantifiers/term_tuple_enumerator.cpp:        anyTerms = QuantifierLogger::s_logger->registerCandidate(
theory/quantifiers/term_tuple_enumerator.cpp:        anyTerms = QuantifierLogger::s_logger->registerCandidate(
theory/quantifiers/term_tuple_enumerator.cpp:    QuantifierLogger::s_logger->increasePhase(d_quantifier);
theory/quantifiers/term_tuple_enumerator_ml.cpp:  const bool useMatrices{QuantifierLogger::s_logger->d_matrices.get()
theory/quantifiers/term_tuple_enumerator_ml.cpp:              QuantifierLogger::s_logger->d_termFeatureProperties.get()))
theory/quantifiers/term_tuple_enumerator_ml.cpp:              QuantifierLogger::s_logger->d_termFeatureProperties.get())));
theory/quantifiers/term_tuple_enumerator_ml.cpp:  Assert(QuantifierLogger::s_logger->hasQuantifier(d_quantifier))
theory/quantifiers/term_tuple_enumerator_ml.cpp:      QuantifierLogger::s_logger->getQuantifierInfo(d_quantifier);
theory/quantifiers/term_tuple_enumerator_ml.cpp:        QuantifierLogger::s_logger->d_tupleFeatureProperties.get());
theory/quantifiers/term_tuple_enumerator_ml.cpp:        QuantifierLogger::s_logger->getQuantifierInfo(d_quantifier);

### Notes

api/cpp/cvc5.cpp:    CALL print()
smt/process_assertions.cpp:  CALL recordAssertions()
smt/smt_engine.cpp:  init/destruct logger
theory/quantifiers/featurize.cpp:  return QuantifierLogger::s_logger->d_matrices.get();
theory/quantifiers/inst_strategy_enumerative.cpp:    QuantifierLogger::s_logger->d_matrices =
theory/quantifiers/instantiate.cpp:    QuantifierLogger::s_logger->registerCurrentInstantiationBody(q, lem);
theory/quantifiers/quantifier_logger.cpp:  const bool useMatrices{QuantifierLogger::s_logger->d_matrices.get()
theory/quantifiers/term_registry.cpp:    QuantifierLogger::s_logger->d_symbolContexts.observeTerm(n);
theory/quantifiers/term_tuple_enumerator.cpp:               || QuantifierLogger::s_logger->d_matrices);
theory/quantifiers/term_tuple_enumerator_ml.cpp:  const bool useMatrices{QuantifierLogger::s_logger->d_matrices.get()

