/*
 * File:  quantifier_logger.h
 * Author:  mikolas
 * Created on:  Fri Dec 25 14:39:39 CET 2020
 * Copyright (C) 2020, Mikolas Janota
 */
#ifndef QUANTIFIER_LOGGER_H_15196
#define QUANTIFIER_LOGGER_H_15196
#include <fstream>
#include <map>
#include <set>
#include <unordered_map>
#include <algorithm>
#include <random>
#include <ctime>

#include "base/map_util.h"
#include "preprocessing/assertion_pipeline.h"
//#include "theory/quantifiers/embedding_matrices.h"
#include "theory/quantifiers/featurize.h"
#include "theory/quantifiers/inst_match_trie.h"
#include "theory/quantifiers/instantiation_list.h"
#include "theory/quantifiers/term_registry.h"
//#include "theory/quantifiers/term_tuple_enumerator_utils.h"
#include "theory/quantifiers_engine.h"

namespace cvc5::internal {
namespace theory {
namespace quantifiers {
//std::ostream& printAnonymousTree(std::ostream& o,
//                                 Node n,
//                                 Node quantifier = Node::null(),
//                                 std::optional<size_t> cursor = std::nullopt);
class QuantifierLogger
{
 public:
  QuantifierLogger(Env& env) : 
     d_env(env),
     d_assertionsBOW(false)
   {
   } ;

  static std::unique_ptr<QuantifierLogger>
      s_logger;  // TODO: get rid of singleton
 public:
//  std::unique_ptr<Matrices> d_matrices;
//  SymbolContexts d_symbolContexts;
//  std::unique_ptr<FeaturePropertiesBase> d_termFeatureProperties;
//  std::unique_ptr<QuantifierFeatureProperties> d_quantifierFeatureProperties;
  std::unique_ptr<QSelFeatureProperties> d_qselFeatureProperties;
//  std::unique_ptr<FeaturePropertiesBase> d_tupleFeatureProperties;
  const Options& options() const { return d_env.getOptions(); }
//  TermRegistry* d_term_registry_store;
//  std::unique_ptr<TermRegistry> d_term_registry_store;
//  std::unique_ptr<TermDb> d_term_db_store;
  TermDb* d_term_database_store;

  std::map<Node, NodeVector> d_round_instantiation_store;

  /// The purpose of these 3 vectors is to store the information that we need to print at the end so
  // we don't have to spend so much time printing throughout the process
  std::vector<std::map<Node, QuantifierInfo>> d_infos_store_vector;
  std::vector<std::map<Node, NodeVector>> d_round_instantiation_store_vector;
  std::vector<RLFeaturizerGraph> d_featurizers_store_vector;

  // TODO RNG is based on current time at start
  std::default_random_engine d_term_rng;

  struct InstantiationExplanation
  {
    Node d_quantifier;
    NodeVector d_instantiation;
  };

  class InstantiationExplanation_equal
  {
   public:
    inline bool operator()(const InstantiationExplanation& s1,
                           const InstantiationExplanation& s2) const
    {
      return s1.d_quantifier == s2.d_quantifier
             && s1.d_instantiation.equals(s2.d_instantiation);
    }
  };

  class InstantiationExplanation_hash
  {
   public:
    std::hash<Node> nh;
    ImmutableVector_hash<Node, std::hash<Node>> vh;
    inline size_t operator()(const InstantiationExplanation& e) const
    {
      return nh(e.d_quantifier) ^ vh(e.d_instantiation);
    }
  };

//  typedef std::unordered_set<InstantiationExplanation,
//                             InstantiationExplanation_hash,
//                             InstantiationExplanation_equal>
//      InstantiationExplanationSet;

  struct InstantiationInfo
  {
    Node d_quantifier;
    NodeVector d_instantiation;
    Node d_body;
  };

  std::ostream& print(std::ostream& out);
  Node d_currentInstantiationBody;
  Node d_currentInstantiationQuantifier;
  void registerCurrentInstantiationBody(Node quantifier, Node body);
  bool registerCandidate(Node quantifier,
                         size_t varIx,
                         Node candidate,
                         bool relevant);

  void recordAssertions(const preprocessing::AssertionPipeline&);

  NodeVector registerInstantiationAttempt(Node quantifier,
                                          const std::vector<Node>& inst);

  void registerInstantiationRound(TermRegistry& treg);
  void registerInstantiation(Node quantifier,
                             bool successful,
                             const NodeVector& inst);

  void registerUsefulInstantiation(const InstantiationList& instantiations);
  /* void registerInstantiations(Node quantifier, QuantifiersEngine*); */

//  size_t getCurrentPhase(Node quantifier) const;
  void increasePhase(Node quantifier);

  QuantifierInfo& getQuantifierInfo(Node quantifier);

  void updateActiveQuantifier(Node quantifier);
  void updateActiveQuantifiers(std::unordered_set<Node> active);
  double getActiveQuantBoost(Node quantifier);

  //------graphs-----
  void registerTermRegistry(TermRegistry& treg);
  const std::vector<TNode>& getAssertionList() const {
    return d_assertion_list;
  }
  void registerInstantiationRL(Node quantifier, const NodeVector& instantiation);

  std::vector<InstantiationInfo>& getInstantiations() {
    return d_instantiationBodies;
  }
    //------------------
  std::ostream& printActiveQuantifiers(std::ostream& out);

  const BOWCalculator& assertionsBOW() const 
  { 
    Trace("quant") << "assertionsBOW: " << std::endl;
    return d_assertionsBOW; 
  }

  bool hasQuantifier(Node quantifier) const
  {
    Trace("quant") << "hasQuantifier: " << quantifier << std::endl;
    return ContainsKey(d_infos, quantifier);
  }
  
  virtual ~QuantifierLogger() { clear(); }
  //
  std::ostream& printRLStyleSamples(std::ostream& out, RLFeaturizerGraph& featurizer, std::vector<int> used_qs);
  std::ostream& printRLStyleSamplesTerms(std::ostream& out, RLFeaturizerGraph& featurizer, std::map<Node, std::vector<Node>> used_terms);
//  std::ostream& printTermSamplesRL(std::ostream& out, RLFeaturizerGraph& featurizer);
  //
  std::ostream& printTermSamplesRL(std::ostream& out, RLFeaturizerGraph& featurizer, std::map<TypeNode, std::vector<Node>>& avail_terms, std::vector<int>& qlist, std::vector<int>& label_list);

  std::ostream& printTermSamplesRLAtEnd(std::ostream& out, RLFeaturizerGraph& featurizer, std::map<Node, QuantifierInfo>& current_d_infos, std::map<Node, NodeVector>& current_round_inst_store);
  std::ostream& printTermSamplesGNNRLNoInfos(
      std::ostream& out, Node quantifier, std::map<TypeNode, std::vector<Node>>& available_terms_of_type, NodeVector& tried_terms,  std::unordered_map<int,int>& node_norm_map
      );
  std::map<Node, QuantifierInfo> d_infos;
 protected:
  Env& d_env;
  BOWCalculator d_assertionsBOW;
  size_t d_globalPhase = 0;
  std::set<TypeNode> d_seenTypes;

  std::map<Node, size_t> d_globalPhases;
  std::vector<InstantiationInfo> d_instantiationBodies;
  std::set<Node> d_hasExplanation;
  std::unordered_map<InstantiationExplanation,
                     std::set<Node>,
                     InstantiationExplanation_hash,
                     InstantiationExplanation_equal>
      d_reasons;

  std::map<Node, size_t> d_activeQuantifiers;
  size_t d_activeQuantifierSum = 0;

  //--------graphs---------
  // get the assertions

  std::vector<TNode> d_assertion_list;
  std::vector<Node> d_inst_list;

  //-----------------------
  void registerTryCandidate(Node quantifier, size_t varIx, Node candidate);
  bool isFromInstantiation(Node n);

  void clear()
  {
    // std::cout << "clearing logger\n";
    d_seenTypes.clear();
    d_infos.clear();
    d_globalPhases.clear();
    d_instantiationBodies.clear();
    d_hasExplanation.clear();
    d_reasons.clear();
    d_activeQuantifiers.clear();
//    d_matrices.reset(nullptr);
  }

  void transitiveExplanation();
//  void calculateReasonScore();
//  int calculateReasonScoreRec(
//      const InstantiationExplanation& reason,
//      const std::map<Node, InstantiationExplanationSet>& tupleContents,
//      std::unordered_set<InstantiationExplanation,
//                         InstantiationExplanation_hash,
//                         InstantiationExplanation_equal>& open);
  void findExplanation(
      InstantiationExplanation reason,
      Node body,
      std::unordered_set<TNode, std::hash<TNode>>& needsExplaining);

//  std::ostream& printExtensive(std::ostream& out);
//  std::ostream& printTermSamples(std::ostream& out);
//  std::ostream& printTermSamplesQuantifier(std::ostream& out,
//                                           Node quantifier,
//                                           const QuantifierInfo& info);
  std::ostream& printQuantifiersSamples(std::ostream& out);
//  std::ostream& printTupleSamples(std::ostream& out);
//  std::ostream& printTupleSamplesQuantifier(std::ostream& out,
//                                            Node quantifier,
//                                            const QuantifierInfo& info);
//  std::ostream& printTupleSample(
//      std::ostream& out,
//      Node quantifier,
//      const NodeVector& instantiation,
//      FeatureVector& featureVector,
//      const BOWCalculator& quantifierFeatures,
//      bool isUseful,
//      std::vector<std::map<Node, TermCandidateInfo>> termsInfo);


  //-----graphs-----
  std::ostream& printTermSamples(std::ostream& out);



  std::ostream& printTermSamplesGNN(std::ostream& out, Node quantifier, const QuantifierInfo& info, std::map<int,int>& node_norm_map);
  // this one does not use the final terms, but just the actual instantiations in each round.
  std::ostream& printTermSamplesGNNRL(std::ostream& out, Node quantifier, const QuantifierInfo& info, NodeVector tried_terms,  std::map<int,int>& node_norm_map);

  std::ostream& printQuantifiersGraphSamples(std::ostream& out);
  std::ostream& printQuantifierListGraphSamples(std::ostream& out);
  std::ostream& printQuantifierPlusAssertionsListGraphSamples(std::ostream& out);


  std::ostream& printQuantifierGraphSample(std::ostream& out,
                                           Node quantifier,
                                           const QuantifierInfo& info);


  //----------------
};

}  // namespace quantifiers
}  // namespace theory
}  // namespace cvc5

#endif /* QUANTIFIER_LOGGER_H_15196 */
