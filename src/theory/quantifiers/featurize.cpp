/*
 * File:  src/theory/quantifiers/featurize.cpp
 * Author:  mikolas
 * Created on:  Tue Apr 6 12:24:10 CEST 2021
 * Copyright (C) 2021, Mikolas Janota
 */
#include "theory/quantifiers/featurize.h"

#include <algorithm>
#include <iomanip>
#include <sstream>
#include <string>
#include <type_traits>

#include "base/check.h"
#include "base/map_util.h"
#include "theory/quantifiers/pytorch_wrapper.h"
#include "options/quantifiers_options.h"
#include "theory/quantifiers/quantifier_logger.h"
#include "theory/quantifiers/term_util.h"
#include "util/rational.h"

/* static float sigmoid(float x) */
/* { */
/*   if (x < 0) */
/*   { */
/*     const double expx = std::exp(x); */
/*     return expx / (1 + expx); */
/*   } */
/*   return 1 / (1 + std::exp(-x)); */
/* } */

namespace cvc5::internal {
//#ifdef CVC5_TRACING
//#define TRACE_VALUE(value, dimension)                                        \
//  do                                                                         \
//  {                                                                          \
//    std::copy(                                                               \
//        value,                                                               \
//        value + dimension,                                                   \
//        std::ostream_iterator<float>(                                        \
//            Trace("featurize") << std::fixed << std::setprecision(4), " ")); \
//  } while (0);
//#else
//#define TRACE_VALUE(value, dimension)
//#endif
//
//FeaturizeMatrices::~FeaturizeMatrices()
//{
//  for (auto& [n, value] : d_values) delete[] value;
//}
//
//static void multiplyAdd(const MLSimpleMatrix& transformation,
//                        const float* value1,
//                        const float* value2,
//                        /*out*/ float* value)
//{
//  const auto dimension = transformation.rows();
//  Assert(transformation.cols() == 2 * dimension);
//  for (auto col = dimension; col--;)
//  {
//    const auto v1Col = value1[col];
//    for (auto row = dimension; row--;)
//      value[row] += transformation.get(row, col) * v1Col;
//  }
//  if (value2)
//  {
//    for (auto col = dimension; col--;)
//    {
//      const auto v2Col = value2[col];
//      const auto colOffset = col + dimension;
//      for (auto row = dimension; row--;)
//        value[row] += transformation.get(row, colOffset) * v2Col;
//    }
//  }
//  for (size_t i = dimension; i--;) value[i] = std::max(value[i], 0.0f);
//  /* for (size_t i = dimension; i--;) value[i] = sigmoid(value[i]); */
//}
//
//void FeaturizeMatrices::visit(TNode n,
//                              const std::vector<float*>& childrenValues)
//{
//  Trace("featurize") << "[featurize] visit: " << n << " (" << n.getKind() << ")"
//                     << std::endl;
//  auto transformationKind = n.getKind();
//  if (d_trackBoundVariable && transformationKind == Kind::BOUND_VARIABLE
//      && !d_quantifier.isNull())
//  {
//    // if we are visiting the variable to be tracked, change type to NULL
//    Node::iterator it =
//        std::find(d_quantifier[0].begin(), d_quantifier[0].end(), n);
//    if (it != d_quantifier[0].end())
//    {
//      const size_t varIx = it - d_quantifier[0].begin();
//      if (varIx == *d_trackBoundVariable) transformationKind = Kind::NULL_EXPR;
//    }
//  }
//  const auto& transformation = d_matrices.transformation(transformationKind);
//  const auto& mult = transformation.d_mult;
//  const auto& bias = transformation.d_bias;
//  const auto dimension = d_matrices.dimension();
//  auto* retv = new float[d_matrices.dimension()];
//  if (childrenValues.size() == 0)
//  {
//    /* just copy the first column of the transformation */
//    AlwaysAssert(bias.rows() == dimension)
//        << "missing data for: " << transformationKind << std::endl;
//    /* Assert(mult.size() == 0); */
//    std::copy_n(bias.data(), dimension, retv);
//  }
//  else
//  {
//    AlwaysAssert(mult.rows() == dimension)
//        << "missing data for: " << transformationKind << std::endl;
//    if (bias.rows() != 0)
//      std::copy_n(bias.data(), dimension, retv);
//    else
//      std::fill_n(retv, dimension, 0.0);
//    multiplyAdd(mult,
//                childrenValues[0],
//                childrenValues.size() > 1 ? childrenValues[1] : nullptr,
//                retv);
//    if (childrenValues.size() > 2)
//    {
//      /* apply multiplication iteratively (left parentheses) */
//      auto* intermediate = new float[d_matrices.dimension()];
//      for (size_t childIx = 2; childIx < childrenValues.size(); childIx++)
//      {
//        std::swap(intermediate, retv);
//        if (bias.rows() != 0)
//          std::copy_n(bias.data(), dimension, retv);
//        else
//          std::fill_n(retv, dimension, 0.0);
//        multiplyAdd(mult, intermediate, childrenValues[childIx], retv);
//      }
//      delete[] intermediate;
//    }
//  }
//  // cache
//  d_values[n] = retv;
//  Trace("featurize") << "[featurize] " << n << " -> ";
//  TRACE_VALUE(retv, dimension);
//  Trace("featurize") << std::endl;
//}
//
//void FeaturizeMatrices::addToVector(TNode n, FeatureVector* dest) const
//{
//  const float* const value = d_values.at(n);
//  for (size_t i = 0; i < d_matrices.dimension(); i++)
//  {
//    dest->addValue(value[i]);
//  }
//}
//
//void FeaturizeMatrices::count(TNode topLevel)
//{
//  Trace("featurize") << "[featurize] " << topLevel;
//  if (d_trackBoundVariable)
//    Trace("featurize") << " var" << *d_trackBoundVariable;
//  Trace("featurize") << std::endl;
//  if (d_trackBoundVariable && topLevel.getKind() == Kind::FORALL)
//  {
//    Assert(d_quantifier.isNull());
//    d_quantifier = topLevel;
//    Trace("featurize") << "[featurize] quantifier: " << topLevel << std::endl;
//  }
//
//  std::vector<float*> childrenValues;
//  std::vector<TNode> todo({topLevel});
//  do
//  {
//    const auto cur = todo.back();
//    if (ContainsKey(d_values, cur))  // already evaluated
//    {
//      todo.pop_back();
//      continue;
//    }
//    if (cur.getKind() == Kind::INST_PATTERN_LIST)
//    {
//      todo.pop_back();
//      d_values[cur] = nullptr;
//      continue;
//    }
//    // check if all children evaluated, otherwise put them on the stack
//    childrenValues.clear();
//    bool childrenEvaluated = true;
//    for (const auto& child : cur)
//    {
//      const auto i = d_values.find(child);
//      if (i == d_values.end())
//      {
//        childrenEvaluated = false;
//        todo.push_back(child);
//      }
//      else
//      {
//        if (i->second) childrenValues.push_back(i->second);
//      }
//    }
//    if (childrenEvaluated)
//    {
//      Assert(cur == todo.back());
//      todo.pop_back();
//      visit(cur, childrenValues);
//    }
//  } while (!todo.empty());
//}

namespace theory {
namespace quantifiers {

#ifdef CVC5_ASSERTIONS
#define CHECK_FT(name, featureVector)                                        \
  do                                                                         \
  {                                                                          \
    Assert(                                                                  \
        (featureVector).d_featureProperties->names()[(featureVector).size()] \
        == (name))                                                           \
        << "expecting feature '" << (name) << "' instead of '"               \
        << (featureVector)                                                   \
               .d_featureProperties->names()[(featureVector).size()]         \
        << "'";                                                              \
  } while (0)
#else
#define CHECK_FT(a, b)
#endif

//void TermFeaturizerBOW::featurizeQuantifierBOW(
//    const Node quantifier,
//    const TermFeatureProperties::Config& config,
//    const BOWCalculator& quantifierFeatures,
//    /*out*/ FeatureVector* featureVector)
//{
//  if (config.d_bow)
//  {
//    quantifierFeatures.getTermBOW().addToVector(*featureVector);
//  }
//
//  if (config.d_procedural)
//  {
//    featureVector->addValue(TermUtil::getTermDepth(quantifier) + 1);
//    CHECK_FT(TermFeatureProperties::FT_QVARIABLES, *featureVector);
//    featureVector->addValue(quantifier[0].getNumChildren());
//  }
//}
//
//void numeralProperties(/*out*/ FeatureVector* dest,
//                       const std::optional<Rational>& min,
//                       const std::optional<Rational>& max)
//{
//  Assert(min.has_value() == max.has_value());
//  if (max)
//  {
//    dest->addValue(min->getDouble());
//    dest->addValue(max->getDouble());
//  }
//  else
//  {
//    dest->addValue(1.0f);
//    dest->addValue(-1.0f);
//  }
//}
//void termContext(/*out*/ FeatureVector* dest, const Node term)
//{
//  const auto& ctxs = QuantifierLogger::s_logger->d_symbolContexts;
//  const auto opts = ctxs.getSymbol(term);
//  if (opts)
//  {
//    if (ctxs.hasContext(*opts))
//    {
//      ctxs.getContext(*opts).addToVector(*dest);
//    }
//    else
//    {
//      BOW::addEmptyToVector(*dest);
//    }
//  }
//  else
//  {
//    AlwaysAssert(false) << term << " has no top level symbol\n";
//  }
//}
//
//void featurizeQuantifierMx(/*out*/ FeatureVector* dest,
//                           const Node quantifier,
//                           const FeaturizeMatrices& quantifierFeatures)
//{
//  quantifierFeatures.addToVector(quantifier, dest);
//}
//
//QuantifierFeaturizerBOW::QuantifierFeaturizerBOW(
//    Node quantifier, const QuantifierFeatureProperties* featureProperties)
//    : d_config(static_cast<const TermFeatureProperties*>(featureProperties)
//                   ->getConfig()),
//      d_quantifier(quantifier),
//      d_featureProperties(featureProperties),
//      d_featureVector(featureProperties)
//{
//}

/* ----------------------- */
QSelFeaturizerBOW::QSelFeaturizerBOW(
    const QSelFeatureProperties* featureProperties)
    : d_featureProperties(featureProperties), d_featureVector(featureProperties)
{
}

void QSelFeaturizerBOW::addContext(const BOWCalculator& bow)
{
  bow.getTermBOW().addToVector(d_featureVector);
}

void QSelFeaturizerBOW::pop() { d_featureVector.pop(); } // namespace

void QSelFeaturizerBOW::pushQuantifier(Node quantifier)
{
  d_featureVector.push();
  d_quantifierFeatures = std::make_unique<BOWCalculator>(false);
  d_quantifierFeatures->count(quantifier);  // BOW
  d_quantifierFeatures->getTermBOW().addToVector(d_featureVector);
  d_featureVector.addValue(TermUtil::getTermDepth(quantifier) + 1);
  CHECK_FT(QSelFeatureProperties::FT_QVARIABLES, d_featureVector);
  d_featureVector.addValue(quantifier[0].getNumChildren());
  CHECK_FT(QSelFeatureProperties::FT_CLOSING_FEATURE, d_featureVector);
  d_featureVector.addValue(1);
}

// -------graphs------

QuantifierFeaturizerGraph::QuantifierFeaturizerGraph(
    Node quantifier
    ) :
        d_quantifier(quantifier)
{
}

QuantifierListFeaturizerGraph::QuantifierListFeaturizerGraph(

    std::vector<Node> quantifier_list
    ) :
        d_quantifier_list(quantifier_list)

{
}

QuantifierPlusAssertionsListFeaturizerGraph::QuantifierPlusAssertionsListFeaturizerGraph(

    std::vector<TNode> assertion_list,
    std::vector<Node> quantifier_list

    ) :
        d_quantifier_list(quantifier_list),
        d_feat_assertion_list(assertion_list)

{
}

RLFeaturizerGraph::RLFeaturizerGraph()

{
}


bool intInMap(std::map<int, int>& map, int key)
{
  if (map.find(key) == map.end()) return false;
  return true;
}


bool intInMap(std::unordered_map<int, int>& map, int key)
{
  if (map.find(key) == map.end()) return false;
  return true;
}

GraphCalculator::GraphCalculator()
{

}

void GraphCalculator::run_dfs_shared_visitedcheck(Node topLevel, std::map<int, int>& shared_visitcheck) {
  std::vector<std::pair<Node, Node>> node_stack;

  // This first node is the quantifier node; perhaps we can use this as a classifier target.
  //std::cout << "QUANT: " << topLevel.getKind() << "Numeric: " << static_cast<int>(topLevel.getKind()) << "ID: " << topLevel.getId() << std::endl;
  node_stack.push_back({topLevel, Node::null()});

  do
  {
    const auto [cur, parent] = node_stack.back();

    node_stack.pop_back();

    if (cur.getKind() == Kind::INST_PATTERN_LIST) {
      continue;
    }
    // do_something
//    std::cout << "Current node: " << cur << std::endl;
//    std::cout << "Putting this node on the stack: " << cur.getKind() << " Numeric Kind " << static_cast<int>(cur.getKind()) << " ID : " << cur.getId() << "Type: " << cur.getType() << std::endl;
//    std::cout << "Type ID: " << cur.getType().getId() << std::endl;
//    std::cout << "Type Kind: " << cur.getType().getKind() << std::endl;
//    if (static_cast<int>(cur.getKind()) == 22 )
//    {
//      std::cout << cur.getOperator() << std::endl;
//      std::cout << cur.getOperator().getId() << std::endl;
//      std::cout << cur.getOperator().getKind() << std::endl;
//    }

    d_gnodeInits.push_back(cur.getKind());

    if (!intInMap(shared_visitcheck, cur.getId())){

      shared_visitcheck[cur.getId()] = 1;
      d_gnodeIdToKind[cur.getId()] = cur.getKind();

    }

    // add them but make the edge types unique so that I can filter at training time.
    bool add_operators_and_types = false;

    if (add_operators_and_types) {

//       TYPE edges
      if (!intInMap(shared_visitcheck, cur.getType().getId()) )
      {
        shared_visitcheck[cur.getType().getId()] = 1;
        d_gnodeIdToKind[cur.getType().getId()] = cur.getType().getKind();
      }
      d_ggraphEdges.push_back(
          {static_cast<int>(cur.getType().getId()), static_cast<int>(cur.getId()), 1});
      d_ggraphEdges.push_back(
          {static_cast<int>(cur.getId()), static_cast<int>(cur.getType().getId()), 2});

      // OPERATORS
      if (static_cast<int>(cur.getKind()) == 22 ) // if kind == APPLY_UF
      {
        if (!intInMap(shared_visitcheck, cur.getOperator().getId()))
        {
          shared_visitcheck[cur.getOperator().getId()] = 1;
          d_gnodeIdToKind[cur.getOperator().getId()] = cur.getOperator().getKind();
        }
        d_ggraphEdges.push_back(
            {static_cast<int>(cur.getOperator().getId()), static_cast<int>(cur.getId()), 3});
        d_ggraphEdges.push_back(
            {static_cast<int>(cur.getId()), static_cast<int>(cur.getOperator().getId()), 4});

      }
    }

    // Construct edges and put more nodes on the stack;
    // Somehow create argument order edges.

    int previous_sibling = -1;
    int sibling_counter = 0;
    for (const auto& child : cur)
    {

      // don't put inst pattern lists onto the stack
      if (!(child.getKind() == Kind::INST_PATTERN_LIST))
      {
//        std::cout << "Child node: " << child << std::endl;
//        std::cout << "Kind: " << child.getKind() << " Numeric Kind " << static_cast<int>(child.getKind()) << " ID : " << child.getId() <<  "Type: " << child.getType()  << std::endl;
//        std::cout << "Type ID" << child.getType().getId() << std::endl;
//        std::cout << "Type Kind: " << child.getType().getKind() << std::endl;
//        if (static_cast<int>(child.getKind()) == 22 )
//        {
//          std::cout << child.getOperator() << std::endl;
//          std::cout << child.getOperator().getId() << std::endl;
//          std::cout << child.getOperator().getKind() << std::endl;
//        }



        // don't put the already visited nodes back on the stack. BUT we do have to put the edges there.
        if (!intInMap(shared_visitcheck, child.getId()))
        {
          node_stack.push_back({child, cur});

        }

        // std::cout << "Edge: [" << static_cast<int>(cur.getId()) << "->" << static_cast<int>(child.getId()) << "]" << std::endl;

        // TODO create edge labels
        // for each edges, create 3rd and 4th entries
        // 3rd is the parent term's id. (isn't this already clear?)
        // 4th is the position in the arguments order

        // NORMAL
//        d_ggraphEdges.push_back(
//            {static_cast<int>(cur.getId()), static_cast<int>(child.getId())});
//        d_ggraphEdges.push_back(
//            {static_cast<int>(child.getId()), static_cast<int>(cur.getId())});

        // EDGE_LABELS
        const bool edge_features =  true;
        if (sibling_counter > 5)
        {
          // make this max 5 for now
          sibling_counter = 5;
        }
        // I want to distinguish between edges going down and edges going up.
//        std::cout << sibling_counter << std::endl;
        d_ggraphEdges.push_back(
            {static_cast<int>(cur.getId()), static_cast<int>(child.getId()), 5 + sibling_counter});
        d_ggraphEdges.push_back(
            {static_cast<int>(child.getId()), static_cast<int>(cur.getId()), 5 + sibling_counter + 5});

        // push back two type 0 edges here

        // Sibling edges for argument order.
        // //
        if (!edge_features)
        {
          if (!(previous_sibling == -1))
          {
            //          if (static_cast<int>(cur.getId()) == previous_sibling) {
            //
            //            std::cout << "This is not an acyclic graph? : left is " << previous_sibling << " and right is same: " << cur.getId() << std::endl;
            //
            //
            //          }

            // TODO don't need to count here
            d_ggraphEdges.push_back(
                {previous_sibling, static_cast<int>(child.getId())});
            // add sibling type edges here

            // then increment
            //
          }

          previous_sibling = static_cast<int>(child.getId());
        }
        sibling_counter += 1 ;
      }


    }

  } while (! node_stack.empty());

}

void GraphCalculator::run_dfs(Node topLevel) {
  std::vector<std::pair<Node, Node>> node_stack;

  // This first node is the quantifier node; perhaps we can use this as a classifier target.
  //std::cout << "QUANT: " << topLevel.getKind() << "Numeric: " << static_cast<int>(topLevel.getKind()) << "ID: " << topLevel.getId() << std::endl;
  node_stack.push_back({topLevel, Node::null()});

  do
  {
    const auto [cur, parent] = node_stack.back();

    node_stack.pop_back();
//    std::cout << "Popped " << cur.getId() << std::endl;
    if (cur.getKind() == Kind::INST_PATTERN_LIST) {
      continue;
    }
    // do_something

    //std::cout << "Putting this node on the stack: " << cur.getKind() << " Numeric Kind " << static_cast<int>(cur.getKind()) << " ID : " << cur.getId() << std::endl;

    d_gnodeInits.push_back(cur.getKind());

    if (!intInMap(d_gnodeIdToKind, cur.getId())){

      d_gnodeIdToKind[cur.getId()] = cur.getKind();
    }
    // Construct edges and put more nodes on the stack;
    // Somehow create argument order edges.

    int previous_sibling = -1;
    for (const auto& child : cur)
    {
//      std::cout << "Child " << child.getId() << std::endl;
//      if (cur.getId() == child.getId()) {
//
//          std::cout << "This is not an acyclic graph? : parent is " << cur.getId() << " and child is same: " << child.getId() << std::endl;
//
//
//        }
      // don't put inst pattern lists onto the stack
      if (!(child.getKind() == Kind::INST_PATTERN_LIST))
      {

        // don't put the already visited nodes back on the stack. BUT we do have to put the edges there.
        if (!intInMap(d_gnodeIdToKind, child.getId()))
        {
          node_stack.push_back({child, cur});

        }

        // std::cout << "Edge: [" << static_cast<int>(cur.getId()) << "->" << static_cast<int>(child.getId()) << "]" << std::endl;

        d_ggraphEdges.push_back(
            {static_cast<int>(cur.getId()), static_cast<int>(child.getId())});
        d_ggraphEdges.push_back(
            {static_cast<int>(child.getId()), static_cast<int>(cur.getId())});
        // Sibling edges for argument order.
        // //
        if (!(previous_sibling == -1))
        {
//          if (static_cast<int>(cur.getId()) == previous_sibling) {
//
//            std::cout << "This is not an acyclic graph? : left is " << previous_sibling << " and right is same: " << cur.getId() << std::endl;
//
//
//          }
          d_ggraphEdges.push_back(
              {previous_sibling, static_cast<int>(child.getId())});
        }


        previous_sibling = static_cast<int>(child.getId());
      }
    }

  } while (! node_stack.empty());

}

template<typename K, typename V>
void print_map_helper(std::map<K, V> const &m)
{
  for (auto const &pair: m) {
    Trace("quant-sampler") << "{" << pair.first << ": " << pair.second << "}" << std::endl;
  }
}

void print_nested_vector_helper(std::vector<std::vector<int>> const &m)
{
  for (auto const &row: m) {
    Trace("quant-sampler") << row[0] << "->" << row[1] << std::endl;
  }
}

void print_vector_helper(std::vector<int> const &m)

{
  for (auto const &entry:m) {
    Trace("quant-sampler") << entry << " " ;

  }
  Trace("quant-sampler") << std::endl;
}

// for the quantifierlist perhaps just run run_dfs multiple times and concat the resultsing map + vector
// then normalize all together
void QuantifierFeaturizerGraph::run()
{
  d_graphFeatures = std::make_unique<GraphCalculator>();

  d_graphFeatures->run_dfs(d_quantifier);

  int quant_node = static_cast<int>(d_quantifier.getId());
  std::map<int, int> node_normalization_map;

  int node_index = 0;


  for(const auto & variable_name : d_graphFeatures->getNodeIdToKind()) {


    if (! intInMap(node_normalization_map, variable_name.first)) {
      node_normalization_map[variable_name.first] = node_index;
      d_nodeInits.push_back(variable_name.second);
      node_index += 1;
    }

  }

  for (auto& edge : d_graphFeatures->getGraphEdges()) {

    std::vector<int> normalized_edge;
    normalized_edge.push_back(node_normalization_map.at(edge[0]));
    normalized_edge.push_back(node_normalization_map.at(edge[1]));


    d_graphEdges.push_back(normalized_edge);
  }

}

void remove_duplicates(std::vector<std::vector<int>> &vec) {
  // Take a set, check if element has already been seen
  // insert . second is true if insertion happened
  // i.e. the element was not already in there
  // If so, point itr to current and afterwards ++ itr
  // erase the overhanging end.
  std::vector<std::vector<int>>::iterator itr = vec.begin();

  std::set<std::vector<int>> seen_set;

  for (auto current = vec.begin(); current != vec.end(); ++current)
  {
    if (seen_set.insert(*current).second) {
      *itr++ = *current;
    }
  }
  vec.erase(itr, vec.end());
}

void QuantifierListFeaturizerGraph::run()
{

  //std::vector<int> unnorm_target_indices;

  // TODO process the assertions at the start.



  for (auto& quantifier : d_quantifier_list) {
    getUnnormalizedTargets().push_back(static_cast<int>(quantifier.getId()));
    d_graphFeatures = std::make_unique<GraphCalculator>();

    d_graphFeatures->run_dfs(quantifier);

    //int quant_node = static_cast<int>(quantifier.getId());
    getNodeIdToKind().insert(d_graphFeatures->getNodeIdToKind().begin(), d_graphFeatures->getNodeIdToKind().end());
    getUnnormalizedEdges().insert(getUnnormalizedEdges().end(), d_graphFeatures->getGraphEdges().begin(), d_graphFeatures->getGraphEdges().end());


  }

  std::map<int, int> node_normalization_map;

  int node_index = 0;


  for(const auto & variable_name : getNodeIdToKind()) {


    if (! intInMap(node_normalization_map, variable_name.first)) {
      node_normalization_map[variable_name.first] = node_index;
      d_nodeInits.push_back(variable_name.second);
      node_index += 1;
    }

  }

  for (auto& edge : getUnnormalizedEdges()) {

    std::vector<int> normalized_edge;
    normalized_edge.push_back(node_normalization_map[edge[0]]);
    normalized_edge.push_back(node_normalization_map[edge[1]]);


    d_graphEdges.push_back(normalized_edge);
  }

  // At the moment, I don't know what to do with duplicate edges;
  //std::cout << "before dedup: " << d_graphEdges.size() << std::endl;
  remove_duplicates(d_graphEdges);
  //std::cout << "after dedup: " << d_graphEdges.size() << std::endl;
  for (auto& unnorm_index : getUnnormalizedTargets()){
    d_target_indices.push_back(node_normalization_map[unnorm_index]);
  }

}

void QuantifierPlusAssertionsListFeaturizerGraph::run()
{

  //std::vector<int> unnorm_target_indices;

  // TODO process the assertions at the start.

  for (auto& assertion : d_feat_assertion_list)
  {
    d_graphFeatures = std::make_unique<GraphCalculator>();

    d_graphFeatures->run_dfs(assertion);

    getNodeIdToKind().insert(d_graphFeatures->getNodeIdToKind().begin(), d_graphFeatures->getNodeIdToKind().end());
    getUnnormalizedEdges().insert(getUnnormalizedEdges().end(), d_graphFeatures->getGraphEdges().begin(), d_graphFeatures->getGraphEdges().end());


  }

  for (auto& quantifier : d_quantifier_list) {
    getUnnormalizedTargets().push_back(static_cast<int>(quantifier.getId()));
    d_graphFeatures = std::make_unique<GraphCalculator>();

    d_graphFeatures->run_dfs(quantifier);

    //int quant_node = static_cast<int>(quantifier.getId());
    getNodeIdToKind().insert(d_graphFeatures->getNodeIdToKind().begin(), d_graphFeatures->getNodeIdToKind().end());
    getUnnormalizedEdges().insert(getUnnormalizedEdges().end(), d_graphFeatures->getGraphEdges().begin(), d_graphFeatures->getGraphEdges().end());


  }

  // this map needs to be persistent
  std::map<int, int> node_normalization_map;

  int node_index = 0;


  for(const auto & variable_name : getNodeIdToKind()) {


    if (! intInMap(node_normalization_map, variable_name.first)) {
      node_normalization_map[variable_name.first] = node_index;
      d_nodeInits.push_back(variable_name.second);
      node_index += 1;
    }

  }

  for (auto& edge : getUnnormalizedEdges()) {

    std::vector<int> normalized_edge;
    normalized_edge.push_back(node_normalization_map[edge[0]]);
    normalized_edge.push_back(node_normalization_map[edge[1]]);


    d_graphEdges.push_back(normalized_edge);
  }

  // At the moment, I don't know what to do with duplicate edges;
  //std::cout << "before dedup: " << d_graphEdges.size() << std::endl;
  remove_duplicates(d_graphEdges);
  //std::cout << "after dedup: " << d_graphEdges.size() << std::endl;
  for (auto& unnorm_index : getUnnormalizedTargets()){
    d_target_indices.push_back(node_normalization_map[unnorm_index]);
  }

}

void RLFeaturizerGraph::init(std::vector<TNode> assert_list, std::vector<Node> quantifier_list, std::vector<Node> inst_list, std::vector<Node> term_list){

  d_quantifier_list = quantifier_list;
  d_feat_assertion_list = assert_list;
  d_inst_list = inst_list;
  d_term_list = term_list;


}

void RLFeaturizerGraph::run()
{

  //std::vector<int> unnorm_target_indices;
  //

  // TODO process the assertions at the start.
  std::unique_ptr<GraphCalculator> d_graphFeatures;
  //todo have a shared visitcheck
  std::map<int, int> shared_visit_map;

  for (auto& assertion : d_feat_assertion_list)
  {
    d_graphFeatures = std::make_unique<GraphCalculator>();

//    d_graphFeatures->run_dfs(assertion);
    d_graphFeatures->run_dfs_shared_visitedcheck(assertion, shared_visit_map);
    getNodeIdToKind().insert(d_graphFeatures->getNodeIdToKind().begin(), d_graphFeatures->getNodeIdToKind().end());
    getUnnormalizedEdges().insert(getUnnormalizedEdges().end(), d_graphFeatures->getGraphEdges().begin(), d_graphFeatures->getGraphEdges().end());


  }
  std::cout << "STATS: Num nodes [ASSERTIONS] " << getNodeIdToKind().size() << std::endl;
  std::cout << "STATS: Num edges [ASSERTIONS] " << getUnnormalizedEdges().size() << std::endl;

  for (auto& quantifier : d_quantifier_list) {
    getUnnormalizedTargets().push_back(static_cast<int>(quantifier.getId()));
    d_graphFeatures = std::make_unique<GraphCalculator>();

//    d_graphFeatures->run_dfs(quantifier);
    d_graphFeatures->run_dfs_shared_visitedcheck(quantifier, shared_visit_map);
    //int quant_node = static_cast<int>(quantifier.getId());
    getNodeIdToKind().insert(d_graphFeatures->getNodeIdToKind().begin(), d_graphFeatures->getNodeIdToKind().end());
    getUnnormalizedEdges().insert(getUnnormalizedEdges().end(), d_graphFeatures->getGraphEdges().begin(), d_graphFeatures->getGraphEdges().end());


  }

  std::cout << "STATS: Num nodes [ASSERTIONS+QS] " << getNodeIdToKind().size() << std::endl;
  std::cout << "STATS: Num edges [ASSERTIONS+QS] " << getUnnormalizedEdges().size() << std::endl;

  for (auto& lemma : d_inst_list)
  {
    d_graphFeatures = std::make_unique<GraphCalculator>();

//    d_graphFeatures->run_dfs(lemma);
    d_graphFeatures->run_dfs_shared_visitedcheck(lemma, shared_visit_map);

    getNodeIdToKind().insert(d_graphFeatures->getNodeIdToKind().begin(), d_graphFeatures->getNodeIdToKind().end());
    getUnnormalizedEdges().insert(getUnnormalizedEdges().end(), d_graphFeatures->getGraphEdges().begin(), d_graphFeatures->getGraphEdges().end());


  }

  std::cout << "STATS: Num nodes [ASSERTIONS+QS+LEMMAS] " << getNodeIdToKind().size() << std::endl;
  std::cout << "STATS: Num edges [ASSERTIONS+QS+LEMMAS] " << getUnnormalizedEdges().size() << std::endl;
  std::cout << "STATS: Num lemmas " << d_inst_list.size() << std::endl;

  for (auto& term : d_term_list)
  {
    d_graphFeatures = std::make_unique<GraphCalculator>();

//    d_graphFeatures->run_dfs(term);
    d_graphFeatures->run_dfs_shared_visitedcheck(term, shared_visit_map);
    getNodeIdToKind().insert(d_graphFeatures->getNodeIdToKind().begin(), d_graphFeatures->getNodeIdToKind().end());
    getUnnormalizedEdges().insert(getUnnormalizedEdges().end(), d_graphFeatures->getGraphEdges().begin(), d_graphFeatures->getGraphEdges().end());


  }

  std::cout << "STATS: Num nodes [ASSERTIONS+QS+LEMMAS+TERMS] " << getNodeIdToKind().size() << std::endl;
  std::cout << "STATS: Num edges [ASSERTIONS+QS+LEMMAS+TERMS] " << getUnnormalizedEdges().size() << std::endl;

  std::unordered_map<int, int> node_normalization_map;

  int node_index = 0;


  for(const auto & variable_name : getNodeIdToKind()) {


    if (! intInMap(node_normalization_map, variable_name.first)) {
      node_normalization_map[variable_name.first] = node_index;
      d_nodeInits.push_back(variable_name.second);
      node_index += 1;

      // ADD self-edges to prevent some indexing problems.
//      std::vector<int> self_edge;
//
//      self_edge.push_back(node_index);
//      self_edge.push_back(node_index);
//      d_graphEdges.push_back(self_edge);
    }

  }
  // copy to store for later use.
  getNodeNormalizationMap() = node_normalization_map;

  for (auto& edge : getUnnormalizedEdges()) {

    std::vector<int> normalized_edge;
//    std::cout << edge[0] << "->" << edge[1] << std::endl;
    normalized_edge.push_back(node_normalization_map.at(edge[0]));
//    std::cout << "target" << std::endl;
    normalized_edge.push_back(node_normalization_map.at(edge[1]));

    normalized_edge.push_back(edge[2]);

    d_graphEdges.push_back(normalized_edge);
  }
  std::cout << "Edges done" << std::endl;

  // At the moment, I don't know what to do with duplicate edges;

  //
  // make sure each node has at least 1 neighbor, itself, simplifies things.
  std::cout << "before dedup (before selfedges): " << d_graphEdges.size() << std::endl;
  auto num_nodes = static_cast<int> (d_nodeInits.size());
  for (int i_se = 0; i_se < num_nodes; ++i_se)
  {
    std::vector<int> self_edge;
    self_edge.push_back(i_se);
    self_edge.push_back(i_se);
    self_edge.push_back(0);
    d_graphEdges.push_back(self_edge);
  }
  std::cout << "before dedup (after selfedges): " << d_graphEdges.size() << std::endl;
  std::chrono::time_point<std::chrono::high_resolution_clock> start_dedup = std::chrono::high_resolution_clock::now();

//  remove_duplicates(d_graphEdges);

  std::chrono::time_point<std::chrono::high_resolution_clock> end_dedup = std::chrono::high_resolution_clock::now();
  std::cout << "cost of dedup: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_dedup - start_dedup).count() << std::endl;

  std::cout << "after dedup: " << d_graphEdges.size() << std::endl;


  for (auto& unnorm_index : getUnnormalizedTargets()){
    d_target_indices.push_back(node_normalization_map.at(unnorm_index));
  }



}

//------graphs-----


/* ----------------------- */

//void TermFeaturizerBOW::initialize()
//{
//  d_quantifierFeatures = std::make_unique<BOWCalculator>(d_config.d_varContext);
//  d_quantifierFeatures->count(d_quantifier);  // BOW
//
//  featurizeQuantifierBOW(
//      d_quantifier, d_config, *(d_quantifierFeatures.get()), &d_featureVector);
//}
//
//void TermFeaturizerMx::initialize()
//{
//  const auto& matrices = *getMatrices();
//  auto* const fm = new FeaturizeMatrices(matrices, std::nullopt);
//  fm->count(d_quantifier);
//  featurizeQuantifierMx(&d_featureVector, d_quantifier, *fm);
//  d_quantifierFeatures.reset(fm);
//  d_variableCache.resize(d_quantifier[0].getNumChildren());
//  for (size_t variableIx = 0; variableIx < d_variableCache.size(); variableIx++)
//  {
//    d_variableCache[variableIx] = std::make_unique<FeaturizeMatrices>(
//        matrices, std::make_optional<size_t>(variableIx));
//    d_variableCache[variableIx]->count(d_quantifier);
//  }
//}
//
//TermFeaturizerBOW::TermFeaturizerBOW(
//    Node quantifier, const FeaturePropertiesBase* featureProperties)
//    : ITermFeaturizer(quantifier, featureProperties),
//      d_config(static_cast<const TermFeatureProperties*>(featureProperties)
//                   ->getConfig())
//{
//}
//
//const Matrices* TermFeaturizerMx::getMatrices()
//{
//  return QuantifierLogger::s_logger->d_matrices.get();
//}
//
//void TermFeaturizerBOW::addTerm(Node term,
//                                size_t variableIx,
//                                const TermCandidateInfo& termInfo)
//{
//  d_featureVector.push();
//
//  if (d_config.d_varContext)
//  {
//    d_quantifierFeatures->getVarContext(variableIx)
//        .addToVector(d_featureVector);
//    CHECK_FT(TermFeatureProperties::FT_VARFREQUENCY, d_featureVector);
//    d_featureVector.addValue(
//        d_quantifierFeatures->getVariableFrequency(variableIx));
//  }
//
//  if (d_config.d_bow)
//  {
//    // BOW of the term
//    BOWCalculator termFeatures(false);
//    termFeatures.count(term);
//    termFeatures.getTermBOW().addToVector(d_featureVector);
//
//    // numerals
//    if (d_config.d_numerals)
//    {
//      numeralProperties(&d_featureVector,
//                        termFeatures.get_min_numeral(),
//                        termFeatures.get_max_numeral());
//      numeralProperties(&d_featureVector,
//                        d_quantifierFeatures->get_min_numeral(),
//                        d_quantifierFeatures->get_max_numeral());
//    }
//  }
//
//  if (d_config.d_termContext)
//  {
//    termContext(&d_featureVector, term);
//  }
//
//  if (d_config.d_procedural)
//  {
//    if (d_config.d_age)
//    {
//      d_featureVector.addValue(termInfo.d_age);
//    }
//    d_featureVector.addValue(termInfo.d_phase);
//    d_featureVector.addValue(termInfo.d_tried);
//    CHECK_FT(TermFeatureProperties::FT_RELEVANT, d_featureVector);
//    d_featureVector.addValue(termInfo.d_relevant);
//    CHECK_FT(TermFeatureProperties::FT_TDEPTH, d_featureVector);
//    d_featureVector.addValue(TermUtil::getTermDepth(term) + 1);
//  }
//  CHECK_FT(TermFeatureProperties::FT_CLOSING_FEATURE, d_featureVector);
//  d_featureVector.addValue(1);
//}
//
//TermFeaturizerMx::TermFeaturizerMx(
//    Node quantifier, const FeaturePropertiesBase* featureProperties)
//    : ITermFeaturizer(quantifier, featureProperties),
//      d_termFeatures(*getMatrices(), std::nullopt)
//{
//}
//
//void TermFeaturizerMx::addTerm(Node term,
//                               size_t varIx,
//                               const TermCandidateInfo& termInfo)
//{
//  d_featureVector.push();
//  d_variableCache[varIx]->addToVector(d_quantifier, &d_featureVector);
//  // features of the term
//  d_termFeatures.count(term);
//  d_termFeatures.addToVector(term, &d_featureVector);
//}
//
//void TermFeaturizerBOW::removeTerm() { d_featureVector.pop(); }
//void TermFeaturizerMx::removeTerm() { d_featureVector.pop(); }
}  // namespace quantifiers
}  // namespace theory

void FeaturePropertiesBase::addKinds(const char* prefix)
{
  for (int i = kind::NULL_EXPR; i < kind::LAST_KIND; i++)
  {
    const auto k = static_cast<Kind>(i);
    std::stringstream ss;
    ss << prefix << k;
    addName(ss.str());
  }
}

//MxTermFeatureProperties::MxTermFeatureProperties(size_t dimension)
//    : FeaturePropertiesBase(false), d_dimension(dimension)
//{
//  for (const auto name : {"Q_", "V_", "T_"})
//    for (size_t i = 0; i < d_dimension; i++) addName(name + std::to_string(i));
//}
//
//const char* TermFeatureProperties::FT_RELEVANT = "T_relevant";
//const char* TermFeatureProperties::FT_TDEPTH = "T_depth";
//const char* TermFeatureProperties::FT_VARFREQUENCY = "varFrequency";
//const char* QuantifierFeatureProperties::FT_QVARIABLES = "Q_variables";
//const char* QuantifierFeatureProperties::FT_CLOSING_FEATURE = "__CLOSING__";
const char* QSelFeatureProperties::FT_QVARIABLES = "Q_variables";
const char* QSelFeatureProperties::FT_CLOSING_FEATURE = "__CLOSING__";
//
//void QuantifierFeatureProperties::initialize()
//{
//  d_initialized = true;
//  addCoreFeatures();
//  addName(FT_CLOSING_FEATURE);
//}
//
//void QuantifierFeatureProperties::addCoreFeatures()
//{
//  if (d_config.d_bow)
//  {
//    addKinds("Q_");  // BOW for the encompassing quantifier
//  }
//
//  if (d_config.d_procedural)
//  {
//    addName("Q_depth");      // The depth of the quantifier
//    addName(FT_QVARIABLES);  // The number of quantified variables
//  }
//}

void QSelFeatureProperties::initialize()
{
  d_initialized = true;
  addCoreFeatures();
  addName(FT_CLOSING_FEATURE);
}

void QSelFeatureProperties::addCoreFeatures()
{
  if (d_config.d_formulaBOW)
  {
    addKinds("F_");  // BOW for the encompassing formula
  }

  addKinds("Q_");          // BOW for the encompassing quantifier
  addName("Q_depth");      // The depth of the quantifier
  addName(FT_QVARIABLES);  // The number of quantified variables
}

//void TermFeatureProperties::addCoreFeatures()
//{
//  QuantifierFeatureProperties::addCoreFeatures();
//  if (d_config.d_varContext)
//  {
//    addKinds("VC_");  // context of the quantified variable as BOW
//    addName(
//        FT_VARFREQUENCY);  // The number of occurrences of the variable in quant
//  }
//
//  if (d_config.d_bow)
//  {
//    addKinds("T_");  // BOW for the candidate term
//    if (d_config.d_numerals)
//    {
//      addName("T_NUM_MIN");
//      addName("T_NUM_MAX");
//      addName("Q_NUM_MIN");
//      addName("Q_NUM_MAX");
//    }
//  }
//
//  if (d_config.d_termContext)
//    addKinds("TC_");  // context of top level term as BOW
//
//  if (d_config.d_procedural)
//  {
//    if (d_config.d_age) addName("age");  // The age of the candidate term
//    addName("phase");                    // The phase of the candidate term
//    addName("tried");  // The number of times the candidate term has been tried
//    addName(FT_RELEVANT);  // The relevancy of the
//                           // candidate term
//    addName(FT_TDEPTH);    // The depth of the candidate term
//  }
//}
//
//TermTupleFeatureProperties::TermTupleFeatureProperties()
//    : FeaturePropertiesBase(false)
//{
//  addKinds("Q_");  // BOW for the encompassing quantifier
//  d_variableSizes.push_back(d_names.size());
//  for (size_t varIx = 0; varIx < 100; varIx++)
//  {
//    std::string prefix = "T" + std::to_string(varIx) + "_";
//    addKinds(prefix.c_str());  // BOW for the candidate term
//    addName(prefix
//            + "varFrequency");  // The number of occurrences of the variable in
//    addName(prefix + "age");    // The age of the candidate term
//    addName(prefix + "phase");  // The phase of the candidate term
//    addName(prefix + "relevant");  // The relevancy of the candidate term
//    addName(prefix + "depth");     // The depth of the candidate term
//    addName(
//        prefix
//        + "tried");  // The number of times the candidate term has been tried
//    // the  quantifier
//    d_variableSizes.push_back(d_names.size());
//  }
//}

FeatureVector::FeatureVector(const FeaturePropertiesBase* featureProperties)
    : d_featureProperties(featureProperties)
{
  d_values.reserve(d_featureProperties->size());
}

Cvc5ostream& operator<<(Cvc5ostream& out, const FeatureVector& features)
{
  const auto names = features.d_featureProperties->names();
  const auto values = features.values();
  for (size_t i = 0; i < values.size(); i++)
  {
    const auto value = values[i];
    if (FP_ZERO != std::fpclassify(value))
    {
      out << (i ? " " : "") << names[i] << "(" << i << "):" << value;
    }
  }
  return out;
}

void FeatureVector::pop()
{
  Assert(!d_markers.empty());
  while (d_values.size() > d_markers.back())
  {
    d_values.pop_back();
  }
  d_markers.pop_back();
}

//SymbolContexts::SymbolContexts() {}
//SymbolContexts::~SymbolContexts() {}
//
//std::optional<Node> SymbolContexts::getSymbol(TNode n) const
//{
//  switch (n.getMetaKind())
//  {
//    case kind::MetaKind::INVALID: Assert(0); break;
//    case kind::MetaKind::OPERATOR:
//    case kind::MetaKind::PARAMETERIZED:
//    case kind::MetaKind::NULLARY_OPERATOR: return n.getOperator(); break;
//    case kind::MetaKind::VARIABLE:
//    case kind::MetaKind::CONSTANT: return n; break;
//  }
//  Unreachable();
//}
//
//void SymbolContexts::observeTerm(TNode topLevel)
//{
//  Trace("featurize") << "[featurize] context: " << topLevel << std::endl;
//  if (ContainsKey(d_visited, topLevel)) return;
//  std::vector<TNode> todo;
//  todo.push_back(topLevel);
//  increaseContext(topLevel, Kind::NULL_EXPR);
//  do
//  {
//    const auto cur = std::move(todo.back());
//    const auto curKind = cur.getKind();
//    todo.pop_back();
//    if (cur.getKind() == Kind::INST_PATTERN_LIST)
//    {
//      continue;
//    }
//    Trace("featurize") << "[featurize] cur: " << cur << std::endl;
//    const auto confirmation = d_visited.insert(cur);
//    if (!confirmation.second)
//    {
//      continue;
//    }
//    for (const auto& child : cur)
//    {
//      increaseContext(child, curKind);
//      todo.push_back(child);
//    }
//  } while (!todo.empty());
//}

BOWCalculator::BOWCalculator(bool trackBoundVariables)
    : d_trackBoundVariables(trackBoundVariables)
{
}

void BOWCalculator::count(TNode topLevel)
{
  Trace("featurize") << "[featurize] featurize: " << topLevel << std::endl;
  if (d_trackBoundVariables && topLevel.getKind() == Kind::FORALL)
  {
    Assert(d_quantifier.isNull());
    d_quantifier = topLevel;
    d_boundVarContexts.resize(d_quantifier[0].getNumChildren());
    Trace("featurize") << "[featurize] quantifier: " << topLevel << std::endl;
  }
  std::vector<std::pair<TNode, TNode>> todo;
  todo.push_back({topLevel, TNode::null()});
  do
  {
    const auto [cur, parent] = todo.back();
    todo.pop_back();
    if (cur.getKind() == Kind::INST_PATTERN_LIST)
    {
      continue;
    }
    Trace("featurize") << "[featurize] cur, parent: " << cur << " " << parent
                       << std::endl;
    touch(cur, parent);
    const auto confirmation = d_visited.insert(cur);
    if (!confirmation.second)
    {
      continue;
    }
    visit(cur);
    for (const auto& child : cur) todo.push_back({child, cur});
  } while (!todo.empty());
  d_quantifier = Node::null();
}

void BOWCalculator::touch(TNode n, TNode parent)
{
  if (!d_trackBoundVariables || n.getKind() != Kind::BOUND_VARIABLE
      || d_quantifier.isNull())
  {
    return;
  }
  Node::iterator it =
      std::find(d_quantifier[0].begin(), d_quantifier[0].end(), n);
  if (it == d_quantifier[0].end())
  {
    // there might be some quantifier nesting so there might be some bound
    // variables not in the current quantifier, so we bail
    return;
  }
  Trace("featurize") << "[featurize] touching bound variable: " << n
                     << std::endl;
  const size_t varIx = it - d_quantifier[0].begin();
  if (d_boundFrequencies.size() <= varIx)
  {
    d_boundFrequencies.resize(varIx + 1, 0);
  }
  d_boundFrequencies[varIx]++;
  Assert(!parent.isNull());
  Assert(varIx < d_boundVarContexts.size());
  d_boundVarContexts[varIx].increaseFeature(parent.getKind());
}

void BOWCalculator::visit(TNode n)
{
  if (n.getKind() == kind::CONST_RATIONAL)
  {
    const Rational& rc = n.getConst<Rational>();
    if (!d_max_numeral)
    {
      d_max_numeral = d_min_numeral = std::make_optional(rc);
    }
    else
    {
      if (rc > *d_max_numeral) d_max_numeral = std::make_optional(rc);
      if (rc < *d_min_numeral) d_min_numeral = std::make_optional(rc);
    }
  }
  d_termBOW.increaseFeature(n.getKind());
}

void BOW::addEmptyToVector(FeatureVector& vec)
{
  for (int i = kind::NULL_EXPR; i < kind::LAST_KIND; i++)
  {
    vec.addValue(0);
  }
}

void BOW::addToVector(FeatureVector& vec) const
{
  for (int i = kind::NULL_EXPR; i < kind::LAST_KIND; i++)
  {
    const auto k = static_cast<Kind>(i);
    const auto frequency = getFrequency(k);
    vec.addValue(frequency);
  }
}
}  // namespace cvc5
