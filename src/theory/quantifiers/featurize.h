/*
 * File:  theory/quantifiers/featurize.h
 * Author:  mikolas
 * Created on:  Tue Apr 6 12:22:17 CEST 2021
 * Copyright (C) 2021, Mikolas Janota
 */
#ifndef THEORY_QUANTIFIERS_FEATURIZE_H_9495
#define THEORY_QUANTIFIERS_FEATURIZE_H_9495
#include <cmath>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "base/check.h"
#include "base/map_util.h"
#include "expr/node.h"
#include "util/rational.h"
//#include "theory/quantifiers/embedding_matrices.h"
#include "theory/quantifiers/term_tuple_enumerator_utils.h"
namespace cvc5::internal {
/**\brief Information we store about each term that appears as a candidate for
 * qualifier instantiation.**/

struct TermCandidateInfo
{
  bool d_initialized = false;
  size_t d_age = -1, d_phase = -1;
  bool d_relevant;
  size_t d_tried = 0;
  static TermCandidateInfo mk(size_t age, size_t phase, bool relevant)
  {
    return TermCandidateInfo{true, age, phase, relevant, 0};
  }
};

typedef ImmutableVector<Node, std::hash<Node>> NodeVector;
typedef std::unordered_set<NodeVector,
                           ImmutableVector_hash<Node>,
                           ImmutableVector_equal<Node>>
                           //ImmutableVector_hash<Node, std::hash<Node>>,
                           //ImmutableVector_equal<Node, std::hash<Node>>>
    TupleSet;

struct QuantifierInfo
{
  std::vector<std::map<Node, TermCandidateInfo>> d_infos;
  TupleSet d_usefulInstantiations;
  TupleSet d_successfulInstantiations;
  TupleSet d_rejectedInstantiations;
  TupleSet d_allInstantiations;
//  std::unordered_map<NodeVector,
//                     int,
//                     ImmutableVector_hash<Node, std::hash<Node>>,
//                     ImmutableVector_equal<Node, std::hash<Node>>>
//      d_instantiationScores;
  size_t d_currentPhase = 0;
};

/**\brief Maintaining the set of features we are using.*/
class FeaturePropertiesBase
{
 public:
  FeaturePropertiesBase(bool variableLength)
      : d_initialized(false), d_variableLength(variableLength)
  {
  }

  virtual ~FeaturePropertiesBase(){};

  [[nodiscard]] const std::vector<std::string>& names() const
  {
    Assert(d_initialized);
    return d_names;
  }

  [[nodiscard]] size_t size() const
  {
    Assert(d_initialized);
    return d_names.size();
  }

  std::ostream& printNames(std::ostream& out)
  {
    Assert(d_initialized);
    for (size_t index = 0; index < d_names.size(); index++)
    {
      out << " " << index << ":" << d_names[index];
    }
    return out;
  }

  bool hasVariableLength() const { return d_variableLength; }
  virtual void initialize() = 0;

 protected:
  bool d_initialized;
  const bool d_variableLength;
  std::vector<std::string> d_names;
  void addName(const std::string& name) { d_names.push_back(name); }
  void addKinds(const char* prefix);
};

///**\brief Maintaining the set of features we are using for a single term.*/
//class MxTermFeatureProperties : public FeaturePropertiesBase
//{
// public:
//  MxTermFeatureProperties(size_t dimension);
//  virtual ~MxTermFeatureProperties(){};
//  virtual void initialize() override { d_initialized = true; }
//
// private:
//  size_t d_dimension = -1;
//};
//
///**\brief Maintaining the set of features we are using for a single
// * quantifier.*/
//class QuantifierFeatureProperties : public FeaturePropertiesBase
//{
// public:
//  struct Config
//  {
//    bool d_procedural, d_age, d_bow, d_numerals, d_varContext, d_termContext;
//  };
//  const static char* FT_QVARIABLES;
//  const static char* FT_CLOSING_FEATURE;
//  QuantifierFeatureProperties(Config config)
//      : FeaturePropertiesBase(false), d_config(config){};
//  virtual ~QuantifierFeatureProperties(){};
//  virtual void initialize() override;
//
//  const Config& getConfig() const { return d_config; }
//
// protected:
//  Config d_config;
//  virtual void addCoreFeatures();
//};

class QSelFeatureProperties : public FeaturePropertiesBase
{
 public:
  struct Config
  {
    bool d_formulaBOW;
  };
  const static char* FT_QVARIABLES;
  const static char* FT_CLOSING_FEATURE;
  QSelFeatureProperties(Config config)
      : FeaturePropertiesBase(false), d_config(config){};
  virtual ~QSelFeatureProperties(){};
  virtual void initialize() override;
  const Config& getConfig() const { return d_config; }

 protected:
  Config d_config;
  virtual void addCoreFeatures();
};

///**\brief Maintaining the set of features we are using for a single term.*/
//class TermFeatureProperties : public QuantifierFeatureProperties
//{
// public:
//  const static char* FT_RELEVANT;
//  const static char* FT_TDEPTH;
//  const static char* FT_VARFREQUENCY;
//
// public:
//  TermFeatureProperties(Config config) : QuantifierFeatureProperties(config) {}
//
//  virtual ~TermFeatureProperties(){};
//
// protected:
//  virtual void addCoreFeatures() override;
//};
//
///**\brief Maintaining the set of features we are using for a tuples of terms.*/
//class TermTupleFeatureProperties : public FeaturePropertiesBase
//{
// public:
//  TermTupleFeatureProperties();
//  virtual ~TermTupleFeatureProperties(){};
//
//  size_t size(size_t variableCount) const
//  {
//    return d_variableSizes[variableCount];
//  }
//  virtual void initialize() override { d_initialized = true; }
//
// private:
//  std::vector<size_t> d_variableSizes;
//};

/**\brief  A feature vector, which should respect features set up in
 * FeatureProperties::s_features.*/
class FeatureVector
{
 public:
  /* this vector behaves according to  of these feature properties */
  const FeaturePropertiesBase* d_featureProperties;

  FeatureVector(const FeaturePropertiesBase* featureProperties);
  virtual ~FeatureVector() = default;

  /* determines if the vector is already full according to feature properties*/
  bool isFull() const { return d_featureProperties->size() == d_values.size(); }

  /* append a value to the future vector */
  void addValue(float value)
  {
    Assert(!isFull());
    Trace("featurize") << "[featurize] addValue: "
                       << d_featureProperties->names()[d_values.size()] << "="
                       << value << std::endl;
    d_values.push_back(value);
  }

  /** When called, the next pop will return to the state.*/
  void push() { d_markers.push_back(d_values.size()); }

  /**Return to the state marked by last push.*/
  void pop();

  const std::vector<float>& values() const { return d_values; }
  const std::vector<float>& rawValues() const { return d_values; }
  const size_t size() const { return d_values.size(); }

 private:
  std::vector<float> d_values;
  std::vector<size_t> d_markers;
};

Cvc5ostream& operator<<(Cvc5ostream& out, const FeatureVector& features);

/**\brief A class used to store bag of words. */
class BOW
{
 public:
  BOW() {}
  virtual ~BOW() = default;

  void addToVector(FeatureVector& vec) const;
  static void addEmptyToVector(FeatureVector& vec);

  int getFrequency(Kind feature) const
  {
    return feature >= 0 && static_cast<size_t>(feature) < d_frequencies.size()
               ? d_frequencies[feature]
               : 0;
  }

  int increaseFeature(int id)
  {
    AlwaysAssert(id >= 0);
    const auto index = static_cast<size_t>(id);
    if (index >= d_frequencies.size())
    {
      d_frequencies.resize(index + 1, 0);
    }
    return ++d_frequencies[index];
  }

 private:
  std::vector<int> d_frequencies;  //  frequency of each word
};

class IFeatureCalculator
{
 public:
  IFeatureCalculator() {}
  virtual ~IFeatureCalculator() {}
  virtual void count(TNode n) = 0;
};

/**\brief  A class used to featurize as BOW.
 *
 * Currently we support back of words (BOW).
 * To calculate BOW, run count and then either obtain each frequency by
 * getFrequency or run addBOWToVector
 * to add all the features to a feature vector.*/
class BOWCalculator : public IFeatureCalculator
{
 public:
  const bool d_trackBoundVariables;
//  const bool d_trackBoundVariablesContext = true;
//
  BOWCalculator(bool trackBoundVariables);
  virtual ~BOWCalculator() {}
  virtual void count(TNode n) override;
  const BOW& getTermBOW() const { return d_termBOW; }
//  const BOW& getVarContext(size_t varIx) const
//  {
//    Assert(d_trackBoundVariablesContext);
//    return d_boundVarContexts[varIx];
//  }
//
//  int getVariableFrequency(size_t variableIndex) const
//  {
//    Assert(d_trackBoundVariables);
//    return variableIndex < d_boundFrequencies.size()
//               ? d_boundFrequencies[variableIndex]
//               : 0;
//  }

  const std::optional<Rational>& get_max_numeral() const
  {
    return d_max_numeral;
  }
  const std::optional<Rational>& get_min_numeral() const
  {
    return d_min_numeral;
  }

 private:
  BOW d_termBOW;                        // BOW of the term
  std::vector<BOW> d_boundVarContexts;  // context for each bound variable
  std::vector<int>
      d_boundFrequencies;  // number of occurrences for bound variables

  std::set<Node> d_visited;
  Node d_quantifier;  // current quantifier being visited, we are assuming we
                      // cannot visit more than one at a time

  std::optional<Rational> d_max_numeral;
  std::optional<Rational> d_min_numeral;
  void visit(TNode n);
  void touch(TNode n, TNode parent);
};

//class FeaturizeMatrices : public IFeatureCalculator
//{
// public:
//  FeaturizeMatrices(const Matrices& matrices,
//                    std::optional<size_t> trackBoundVariable)
//      : d_matrices(matrices), d_trackBoundVariable(trackBoundVariable)
//  {
//  }
//  virtual ~FeaturizeMatrices();
//  virtual void count(TNode n) override;
//  void addToVector(TNode n, FeatureVector* dest) const;
//  const float* getValues(TNode n) const { return d_values.at(n); }
//
// private:
//  const Matrices& d_matrices;
//  std::optional<size_t> d_trackBoundVariable;
//  std::map<Node, float*> d_values;
//  Node d_quantifier;
//  void visit(TNode n, const std::vector<float*>& childrenValues);
//};
//
//class SymbolContexts
//{
// public:
//  SymbolContexts();
//  virtual ~SymbolContexts();
//  void observeTerm(TNode n);
//  std::optional<Node> getSymbol(TNode n) const;
//  bool hasContext(Node node) const { return ContainsKey(d_symbol2BOW, node); }
//  const BOW& getContext(Node node) const
//  {
//    Assert(ContainsKey(d_symbol2BOW, node))
//        << "missing context for:" << node << std::endl;
//    return d_symbol2BOW.at(node);
//  }
//
//  // all symbols within some context will not be updated anymore
//  void lockExisting()
//  {
//    d_locked.insert(d_visited.begin(),
//                    d_visited.end());  // TODO:  not terribly efficient here
//  }
//
// private:
//  std::map<Node, BOW> d_symbol2BOW;  // context for each symbol
//  std::set<Node> d_visited;
//  std::set<Node> d_locked;
//  inline void increaseContext(TNode n, Kind parentKind)
//  {
//    const auto opts = getSymbol(n);
//    if (!opts || ContainsKey(d_locked, *opts))
//    {
//      return;
//    }
//    const auto& s = *opts;
//    Trace("featurize") << "[featurize] symbol: " << n << " : " << s
//                       << std::endl;
//    d_symbol2BOW[s].increaseFeature(parentKind);
//    Trace("featurize") << "[featurize] inc ctx: " << s << "-" << parentKind
//                       << std::endl;
//  }
//};

namespace theory {
namespace quantifiers {

class GraphCalculator {

 public:
  GraphCalculator();
  const std::vector<int>& getNodeInits() const {return d_gnodeInits; }
  const std::vector<std::vector<int>>& getGraphEdges() const {return d_ggraphEdges;}
  const std::map<int, int>& getNodeIdToKind() const {return d_gnodeIdToKind; }

  std::vector<int> d_gnodeInits;
  std::vector<std::vector<int>> d_ggraphEdges;
  // TODO have std::vector<int> that signifies 'edgetype' to encode argument order better?
  std::map<int, int> d_gnodeIdToKind;

  void run_dfs(Node n);
  void run_dfs_shared_visitedcheck(Node topLevel, std::map<int, int>& shared_visitcheck);
};


// This was originally just for 1 quantifier at a time; but this means there is no context.
// perhaps run() can be chained with the global node ids to make a bigger graph.
class QuantifierFeaturizerGraph{
 public:
  QuantifierFeaturizerGraph(Node quantifier);
  void run();
  const std::vector<int>& nodeInits() const {return d_nodeInits; }
  const std::vector<std::vector<int>>& graphEdges() const {return d_graphEdges; }
  const Node& topNode() const {return d_quantifier;}
 protected:

  const Node d_quantifier;
  std::unique_ptr<GraphCalculator> d_graphFeatures;
  std::vector<int> d_nodeInits;
  std::vector<std::vector<int>> d_graphEdges;

};

// need to somehow choose which quantifier is selected; some subtlety might be required.
class QuantifierListFeaturizerGraph{
 public:
  QuantifierListFeaturizerGraph(std::vector<Node> quantifier_list);
  void run();
  const std::vector<int>& nodeInits() const {return d_nodeInits;}
  const std::vector<std::vector<int>>& graphEdges() const {return d_graphEdges;}
  std::map<int,int>& getNodeIdToKind() {return d_nodeIdToKind;}

  std::vector<std::vector<int>>& getUnnormalizedEdges() {return d_unnormalizedEdges;}

  std::vector<int>& getUnnormalizedTargets() {return d_target_indices_unnormalized;}
  std::vector<int>& getNormalizedTargets() {return d_target_indices;}

 protected:
  const std::vector<Node> d_quantifier_list;

  std::unique_ptr<GraphCalculator> d_graphFeatures;

  // d_nodeInits & d_graphEdges are meant to contain normalized (node index contiguous and starts from 0)
  // data.
  std::vector<int> d_nodeInits;
  std::vector<std::vector<int>> d_graphEdges;

  // d_nodeIdToKind is meant to contain the non-normalized sum of all quantifier level maps
  std::map<int,int> d_nodeIdToKind;
  // d_unnormalizedEdges is meant to contain the concatenation of the quantifier level vectors containing
  // the edges. This will get normalized and stored into d_graphEdges after.
  std::vector<std::vector<int>> d_unnormalizedEdges;
  // contains the indices of the q nodes, that we will get a prediction for.
  std::vector<int> d_target_indices;

  std::vector<int> d_target_indices_unnormalized;
};

class QuantifierPlusAssertionsListFeaturizerGraph{
 public:
  QuantifierPlusAssertionsListFeaturizerGraph(std::vector<TNode> assert_list, std::vector<Node> quantifier_list);
  void run();
  const std::vector<int>& nodeInits() const {return d_nodeInits;}
  const std::vector<std::vector<int>>& graphEdges() const {return d_graphEdges;}
  std::map<int,int>& getNodeIdToKind() {return d_nodeIdToKind;}

  std::vector<std::vector<int>>& getUnnormalizedEdges() {return d_unnormalizedEdges;}

  std::vector<int>& getUnnormalizedTargets() {return d_target_indices_unnormalized;}
  std::vector<int>& getNormalizedTargets() {return d_target_indices;}

 protected:
  const std::vector<Node> d_quantifier_list;
  const std::vector<TNode> d_feat_assertion_list;

  std::unique_ptr<GraphCalculator> d_graphFeatures;

  // d_nodeInits & d_graphEdges are meant to contain normalized (node index contiguous and starts from 0)
  // data.
  std::vector<int> d_nodeInits;
  std::vector<std::vector<int>> d_graphEdges;

  // d_nodeIdToKind is meant to contain the non-normalized sum of all quantifier level maps
  std::map<int,int> d_nodeIdToKind;
  // d_unnormalizedEdges is meant to contain the concatenation of the quantifier level vectors containing
  // the edges. This will get normalized and stored into d_graphEdges after.
  std::vector<std::vector<int>> d_unnormalizedEdges;
  // contains the indices of the q nodes, that we will get a prediction for.
  std::vector<int> d_target_indices;

  std::vector<int> d_target_indices_unnormalized;
};

class RLFeaturizerGraph{
 public:
  RLFeaturizerGraph();
  void init(std::vector<TNode> assert_list, std::vector<Node> quantifier_list, std::vector<Node> inst_list, std::vector<Node> term_list);
  void run();
  const std::vector<int>& nodeInits() const {return d_nodeInits;}
  const std::vector<std::vector<int>>& graphEdges() const {return d_graphEdges;}
  std::map<int,int>& getNodeIdToKind() {return d_nodeIdToKind;}

  std::vector<std::vector<int>>& getUnnormalizedEdges() {return d_unnormalizedEdges;}

  std::vector<int>& getUnnormalizedTargets() {return d_target_indices_unnormalized;}
  std::vector<int>& getNormalizedTargets() {return d_target_indices;}
  std::unordered_map<int,int>& getNodeNormalizationMap() {return d_node_normalization_map;}

 protected:
  std::vector<Node> d_quantifier_list;
  std::vector<TNode> d_feat_assertion_list;
  std::vector<Node> d_inst_list;
  std::vector<Node> d_term_list;



  // d_nodeInits & d_graphEdges are meant to contain normalized (node index contiguous and starts from 0)
  // data.
  std::vector<int> d_nodeInits;
  std::vector<std::vector<int>> d_graphEdges;

  // d_nodeIdToKind is meant to contain the non-normalized sum of all quantifier level maps
  std::map<int,int> d_nodeIdToKind;
  // map that contains the getId ids as in the solver, mapped to the ids used in the GNN.
  std::unordered_map<int, int> d_node_normalization_map;
  // d_unnormalizedEdges is meant to contain the concatenation of the quantifier level vectors containing
  // the edges. This will get normalized and stored into d_graphEdges after.
  std::vector<std::vector<int>> d_unnormalizedEdges;
  // contains the indices of the q nodes, that we will get a prediction for.
  std::vector<int> d_target_indices;

  std::vector<int> d_target_indices_unnormalized;
};

//#ifdef FOOBAR
//void featurizeQuantifierBOW(/*out*/ FeatureVector* dest,
//                            const BOWCalculator& quantifierFeatures);
//void featurizeTermBOW(/*out*/ FeatureVector* dest,
//                      const Node term,
//                      size_t variableIx,
//                      const TermCandidateInfo& termInfo,
//                      const BOWCalculator& quantifierFeatures);
//void featurizeQuantifierMx(/*out*/ FeatureVector* dest,
//                           const Node quantifier,
//                           const FeaturizeMatrices& quantifierFeatures);
//void featurizeTermMx(/*out*/ FeatureVector* dest,
//                     const Matrices& matrices,
//                     const Node term,
//                     const Node quantifier,
//                     size_t variableIx,
//                     const TermCandidateInfo& termInfo,
//                     const FeaturizeMatrices& quantifierFeatures);
//
//class QuantifierFeaturizer
//{
// public:
//  QuantifierFeaturizer(Node quantifier,
//                       const QuantifierInfo& info,
//                       const FeaturePropertiesBase* featureProperties)
//      : d_quantifier(quantifier),
//        d_info(info),
//        d_featureVector(featureProperties)
//  {
//  }
//  virtual ~QuantifierFeaturizer() {}
//  virtual void initialize();
//  const FeatureVector& featureVector() const { return d_featureVector; }
//
// protected:
//  Node d_quantifier;
//  const QuantifierInfo& d_info;
//  FeatureVector d_featureVector;
//};
//#endif
//
//class ITermFeaturizer
//{
// public:
//  ITermFeaturizer(Node quantifier,
//                  const FeaturePropertiesBase* featureProperties)
//      : d_quantifier(quantifier),
//        d_featureProperties(featureProperties),
//        d_featureVector(featureProperties)
//  {
//  }
//  virtual ~ITermFeaturizer() {}
//  virtual void addTerm(Node term,
//                       size_t varIx,
//                       const TermCandidateInfo& termInfo) = 0;
//  virtual void initialize() = 0;
//  virtual void removeTerm() = 0;
//  const FeatureVector& featureVector() const { return d_featureVector; }
//
// protected:
//  const Node d_quantifier;
//  const FeaturePropertiesBase* const d_featureProperties;
//  FeatureVector d_featureVector;
//};
//
//class QuantifierFeaturizerBOW
//{
// public:
//  const TermFeatureProperties::Config& d_config;
//  QuantifierFeaturizerBOW(Node quantifier,
//                          const QuantifierFeatureProperties* featureProperties);
//  void run();
//  const FeatureVector& featureVector() const { return d_featureVector; }
//
// protected:
//  const Node d_quantifier;
//  const FeaturePropertiesBase* const d_featureProperties;
//  std::unique_ptr<BOWCalculator> d_quantifierFeatures;
//  FeatureVector d_featureVector;
//};

class QSelFeaturizerBOW
{
 public:
  QSelFeaturizerBOW(const QSelFeatureProperties* featureProperties);
  void addContext(const BOWCalculator&);
  void pushQuantifier(Node quantifier);
  void pop();
  const FeatureVector& featureVector() const { return d_featureVector; }

 protected:
  const FeaturePropertiesBase* const d_featureProperties;
  std::unique_ptr<BOWCalculator> d_quantifierFeatures;
  FeatureVector d_featureVector;
};

//class TermFeaturizerBOW : public ITermFeaturizer
//{
// public:
//  const QuantifierFeatureProperties::Config& d_config;
//  TermFeaturizerBOW(Node quantifier,
//                    const FeaturePropertiesBase* featureProperties);
//  virtual void addTerm(Node term,
//                       size_t varIx,
//                       const TermCandidateInfo& termInfo) override;
//  virtual void initialize() override;
//  virtual void removeTerm() override;
//  static void featurizeQuantifierBOW(
//      const Node quantifier,
//      const TermFeatureProperties::Config& config,
//      const BOWCalculator& quantifierFeatures,
//      /*out*/ FeatureVector* featureVector);
//
// protected:
//  std::unique_ptr<BOWCalculator> d_quantifierFeatures;
//  void featurizeQuantifierBOW(const BOWCalculator&);
//};
//
//class TermFeaturizerMx : public ITermFeaturizer
//{
// public:
//  TermFeaturizerMx(Node quantifier,
//                   const FeaturePropertiesBase* featureProperties);
//  virtual void initialize() override;
//  virtual void addTerm(Node term,
//                       size_t varIx,
//                       const TermCandidateInfo& termInfo) override;
//  virtual void removeTerm() override;
//
// protected:
//  std::unique_ptr<IFeatureCalculator> d_quantifierFeatures;
//  std::vector<std::unique_ptr<FeaturizeMatrices>> d_variableCache;
//  const Matrices* getMatrices();
//  FeaturizeMatrices d_termFeatures;
//};

}  // namespace quantifiers
}  // namespace theory
}  // namespace cvc5
#endif /* THEORY_QUANTIFIERS_FEATURIZE_H_9495 */
