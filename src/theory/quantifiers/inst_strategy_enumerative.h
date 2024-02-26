/******************************************************************************
 * Top contributors (to current version):
 *   Andrew Reynolds, Andres Noetzli, Aina Niemetz
 *
 * This file is part of the cvc5 project.
 *
 * Copyright (c) 2009-2022 by the authors listed in the file AUTHORS
 * in the top-level source directory and their institutional affiliations.
 * All rights reserved.  See the file COPYING in the top-level source
 * directory for licensing information.
 * ****************************************************************************
 *
 * Enumerative instantiation.
 */

#include "cvc5_private.h"

#ifndef CVC5__INST_STRATEGY_ENUMERATIVE_H
#define CVC5__INST_STRATEGY_ENUMERATIVE_H

#include <random>

#include "theory/quantifiers/quant_module.h"
#include "theory/quantifiers/ml.h"
#include "theory/quantifiers/pytorch_wrapper.h"
//#include "theory/quantifiers/quantifier_logger.h"
//#include "options/quantifiers_options.h"

//using namespace cvc5::context;
namespace cvc5::internal {
namespace theory {
namespace quantifiers {

class RelevantDomain;

struct LearningGlobalContext
{
  LearningGlobalContext() {}
  virtual ~LearningGlobalContext() {}
  //TermRegistry* d_treg = nullptr;
  //std::unique_ptr<PredictorInterface> d_ml;
  //std::unique_ptr<PredictorInterface> d_tuplePredictor;
  std::unique_ptr<PredictorInterface> d_quantifierPredictor;
  std::unique_ptr<PyTorchWrapper> d_graphquantifierPredictor;
  std::unique_ptr<PyTorchWrapper> d_graphtermPredictor;
  std::map<int, std::set<int>> d_orphan_terms;
  std::set<TypeNode> d_persistent_type_set;
  std::set<Node> d_persistent_term_set;

  int d_round_counter = 0;
  std::vector<int> d_time_spent_process_list;

  // also store the node normalization map here
  std::unique_ptr<std::unordered_map<int, int>> d_node_cvc_graph_index;

  std::unique_ptr<std::unordered_map<int, int>> d_node_cvc_graph_index_reverse;

  std::unique_ptr<std::map<std::pair<int,int>, float>> d_varterm_score_lookup;
  // todo
  // std::unique_ptr<std::map<int, std::map<int, float>>

  // for rlstyle
  // store instantiations every round


  //PredictorInterface* makeMLFromOptions();

  //TimerStat d_learningTimer, d_mlTimer, d_featurizeTimer;
  //IntStat d_learningCounter;

  std::mt19937 d_mt;
};


/** Enumerative instantiation
 *
 * This class implements enumerative instantiation described
 * in Reynolds et al., "Revisiting Enumerative Instantiation".
 *
 * It is an instance of QuantifiersModule, whose main
 * task is to make calls to QuantifiersEngine during
 * calls to QuantifiersModule::check(...).
 *
 * This class adds instantiations based on enumerating
 * well-typed terms that occur in the current context
 * based on a heuristically determined ordering on
 * tuples of terms. This ordering incorporates
 * reasoning about the relevant domain of quantified
 * formulas (see theory/quantifiers/relevant_domain.h).
 * We consider only ground terms that occur in the
 * context due to Theorem 1 of "Revisiting Enumerative
 * Instantiation". Notice this theorem holds only for theories
 * with compactness. For theories such as arithmetic,
 * this class may introduce "default" terms that are
 * used in instantiations, say 0 for arithmetic, even
 * when the term 0 does not occur in the context.
 *
 * This strategy is not enabled by default, and
 * is enabled by the option:
 *   --full-saturate-quant
 *
 * It is generally called with lower priority than
 * other instantiation strategies, although this
 * option interleaves it with other strategies
 * during quantifier effort level QEFFORT_STANDARD:
 *   --fs-interleave
 */
class InstStrategyEnum : public QuantifiersModule
{
 public:
  InstStrategyEnum(Env& env,
                   QuantifiersState& qs,
                   QuantifiersInferenceManager& qim,
                   QuantifiersRegistry& qr,
                   TermRegistry& tr,
                   RelevantDomain* rd);
  ~InstStrategyEnum() {}
  /** Presolve */
  void presolve() override;
  /** Needs check. */
  bool needsCheck(Theory::Effort e) override;
  /** Reset round. */
  void reset_round(Theory::Effort e) override;
  /** Check.
   * Adds instantiations for all currently asserted
   * quantified formulas via calls to process(...)
   */
  void check(Theory::Effort e, QEffort quant_e) override;
  /** Identify. */
  std::string identify() const override
  {
    return std::string("InstStrategyEnum");
  }

 private:
  /** Pointer to the relevant domain utility of quantifiers engine */
  RelevantDomain* d_rd;
  /** process quantified formula
   *
   * q is the quantified formula we are constructing instances for.
   * fullEffort is whether we are called at full effort.
   *
   * If this function returns true, then one instantiation
   * (determined by an enumeration) was added via a successful
   * call to QuantifiersEngine::addInstantiation(...).
   *
   * If fullEffort is true, then we may introduce a "default"
   * well-typed term *not* occurring in the current context.
   * This handles corner cases where there are no well-typed
   * ground terms in the current context to instantiate with.
   *
   * The flag isRd indicates whether we are trying relevant domain
   * instantiations. If this flag is false, we are trying arbitrary ground
   * term instantiations.
   */
  bool process(Node q, bool fullEffort, bool isRd);
  /**
   * A limit on the number of rounds to apply this strategy, where a value < 0
   * means no limit. This value is set to the value of fullSaturateLimit()
   * during presolve.
   */
  int32_t d_enumInstLimit;

  LearningGlobalContext d_learningContext;

}; /* class InstStrategyEnum */

}  // namespace quantifiers
}  // namespace theory
}  // namespace cvc5::internal

#endif