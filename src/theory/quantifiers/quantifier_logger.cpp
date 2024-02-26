/*
 * File:  quantifier_logger.cpp
 * Author:  mikolas
 * Created on:  Fri Dec 25 14:39:45 CET 2020
 * Copyright (C) 2020, Mikolas Janota
 */
#include "theory/quantifiers/quantifier_logger.h"

#include <cmath>
#include <iostream>
#include <iterator>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <unordered_set>
#include <utility>

#include "base/map_util.h"
#include "expr/skolem_manager.h"
#include "options/quantifiers_options.h"
#include "theory/quantifiers/featurize.h"
#include "theory/quantifiers/first_order_model.h"
#include "theory/quantifiers/quantifiers_attributes.h"
//#include "theory/quantifiers/term_tuple_enumerator_utils.h"
#include "theory/quantifiers/term_util.h"

namespace cvc5::internal {
namespace theory {
namespace quantifiers {

#define ASSERT_EXISTENCE(termsInfo, term, varIx, quantifier)      \
  Assert(ContainsKey(termsInfo[varIx], term))                     \
      << "missing info about: '" << term << "' for var " << varIx \
      << " in quant '" << quantifier << "'" << std::endl;

static Cvc5ostream& operator<<(
    Cvc5ostream& out, const QuantifierLogger::InstantiationExplanation& t)
{
  return out << "[" << t.d_quantifier << " <- " << t.d_instantiation << "]";
}

std::ostream& operator<<(std::ostream& o,
                         const QuantifierLogger::InstantiationExplanation& e)
{
  return o << e.d_quantifier << " <- " << e.d_instantiation;
}

//Kind filterKind(Node n, Node quantifier, std::optional<size_t> cursorVarIx)
//{
//  const auto original = n.getKind();
//  if (!cursorVarIx || quantifier.isNull() || original != Kind::BOUND_VARIABLE)
//    return original;
//  //  if we are visiting the variable to be tracked, change type to NULL
//  Node::iterator it = std::find(quantifier[0].begin(), quantifier[0].end(), n);
//  if (it == quantifier[0].end()) return original;
//  const size_t varIx = it - quantifier[0].begin();
//  return varIx == *cursorVarIx ? Kind::NULL_EXPR : original;
//}
//
//static std::ostream& printTree(std::ostream& o,
//                               TNode n,
//                               Node quantifier = Node::null(),
//                               std::optional<size_t> cursor = std::nullopt)
//{
//  o << "(";
//  switch (n.getKind())
//  {
//    case Kind::APPLY_UF: o << n.getOperator(); break;
//
//    case Kind::CONST_RATIONAL:
//    {
//      const Rational& r = n.getConst<Rational>();
//      if (r.isIntegral())
//        o << r;
//      else
//        o << n;
//      break;
//    }
//    case Kind::BOUND_VAR_LIST:
//    {
//      o << n.getOperator();
//      for (const auto& child : n)
//      {
//        printTree(o << " ", child, quantifier, cursor);
//        /* o << " " << child.getType() << ")"; */
//        /* o << ")"; */
//      }
//      return o << ")";
//    }
//    case Kind::BOUND_VARIABLE:
//    {
//      if (quantifier.isNull())
//        o << n;
//      else
//      {
//        auto it = std::find(quantifier[0].begin(), quantifier[0].end(), n);
//        if (it == quantifier[0].end())
//          o << "!foreign_QVAR!";
//        else
//        {
//          const size_t varIx = it - quantifier[0].begin();
//          if (cursor && varIx == *cursor)
//            o << "NULL";
//          else
//            o << "!QVAR_" << varIx << "!";
//        }
//      }
//      break;
//    }
//    default:
//    {
//      if (n.getNumChildren() == 0)
//        o << n;
//      else
//        o << filterKind(n, quantifier, cursor);
//    }
//  }
//  for (const auto& child : n)
//  {
//    printTree(o << " ", child, quantifier, cursor);
//  }
//  return o << ")";
//}
//
//std::ostream& printAnonymousTree(std::ostream& o,
//                                 Node n,
//                                 Node quantifier,
//                                 std::optional<size_t> cursor)
///* Node quantifier = Node::null(), */
///* std::optional<size_t> cursor = std::nullopt) */
//{
//  o << "(" << filterKind(n, quantifier, cursor);
//  for (const auto& child : n)
//    printAnonymousTree(o << " ", child, quantifier, cursor);
//  return o << ")";
//}
//static std::ostream& printAnonymous(std::ostream& o,
//                                    Node n,
//                                    Node quantifier = Node::null(),
//                                    std::optional<size_t> cursor = std::nullopt)
//{
//  std::map<TNode, size_t> ids;
//  size_t maxID = 0;
//  std::vector<TNode> todo({n});
//  std::vector<size_t> childrenValues;
//  do
//  {
//    const auto cur = todo.back();
//    if (ContainsKey(ids, cur))  //  already evaluated
//    {
//      todo.pop_back();
//      continue;
//    }
//    // check if all children evaluated, otherwise put them on the stack
//    childrenValues.clear();
//    bool childrenEvaluated = true;
//    for (const auto& child : cur)
//    {
//      const auto i = ids.find(child);
//      if (i == ids.end())
//      {
//        childrenEvaluated = false;
//        todo.push_back(child);
//      }
//      else
//      {
//        Assert(i->second);
//        childrenValues.push_back(i->second);
//      }
//    }
//    if (childrenEvaluated)
//    {
//      Assert(cur == todo.back());
//      todo.pop_back();
//      ids[cur] = ++maxID;
//      o << "(" << maxID << ":" << filterKind(cur, quantifier, cursor);
//      for (auto child : childrenValues)
//      {
//        o << " " << child;
//      }
//      o << ")";
//    }
//  } while (!todo.empty());
//  return o;
//}

static std::ostream& printSample(std::ostream& out,
                                 int label,
                                 const FeatureVector& features)

{
  out << label;
  Assert(features.d_featureProperties->hasVariableLength()
         || features.isFull());
  const auto& values = features.values();
  for (size_t i = 0; i < values.size(); i++)
  {
    const auto value = values[i];
    if (FP_ZERO != std::fpclassify(value))
    {
      out << " " << i << ":" << value;
    }
  }
  out << std::endl;

#ifdef CVC5_TRACING
  const auto& names = features.d_featureProperties->names();
  out << "; " << label;
  for (size_t i = 0; i < values.size(); i++)
  {
    const auto value = values[i];
    if (FP_ZERO != std::fpclassify(value))
    {
      out << " " << names[i] << ":" << value;
    }
  }
  out << std::endl;
#endif

  return out;
}

std::unique_ptr<QuantifierLogger> QuantifierLogger::s_logger;

QuantifierInfo& QuantifierLogger::getQuantifierInfo(Node quantifier)
{
  Trace("quant") << "getQuantifierInfo: " << quantifier << std::endl;
  auto [it, wasInserted] = d_infos.insert({quantifier, QuantifierInfo()});
  auto& qi = it->second;
  if (wasInserted)
  {
    qi.d_infos.resize(quantifier[0].getNumChildren());
  }
  Assert(qi.d_infos.size() == quantifier[0].getNumChildren());
  return qi;
}

//size_t QuantifierLogger::getCurrentPhase(Node quantifier) const
//{
//  const auto i = d_infos.find(quantifier);
//  return i != d_infos.end() ? i->second.d_currentPhase : 0;
//}
//
void QuantifierLogger::increasePhase(Node quantifier)
{
  Trace("quant") << "increasePhase: " << quantifier << std::endl;
  if (!ContainsKey(d_infos, quantifier))
  {
    d_infos[quantifier].d_infos.resize(quantifier[0].getNumChildren());
    d_infos[quantifier].d_currentPhase = 1;
  }
  else
  {
    d_infos[quantifier].d_currentPhase++;
  }
}

void QuantifierLogger::registerUsefulInstantiation(
    const InstantiationList& instantiations)
{
  Trace("quant") << "registerUsefulInstantiation: " << instantiations << std::endl;
   auto& qi = getQuantifierInfo(instantiations.d_quant);
   for (const auto& instantiation : instantiations.d_inst)
   {
     NodeVector v(instantiation.d_vec);
     Assert(ContainsKey(qi.d_allInstantiations, v));
     qi.d_usefulInstantiations.insert(v);
   }
}

void QuantifierLogger::registerInstantiationRound(TermRegistry& treg)
{
  Trace("quant") << "registerInstantiationRound: " << d_globalPhases << std::endl;
  Trace("ml") << "registerInstantiationRound " << d_globalPhase << std::endl;
  FirstOrderModel* const fm = treg.getModel();
  const TermDb* const db = treg.getTermDatabase();
  std::set<TypeNode> newTypes;
  for (auto quantifierIx = fm->getNumAssertedQuantifiers(); quantifierIx--;)
  {
    Node q = fm->getAssertedQuantifier(quantifierIx, true);
    for (auto varIx{q[0].getNumChildren()}; varIx--;)
    {
      const TypeNode typeNode = q[0][varIx].getType();
      const auto it = d_seenTypes.insert(typeNode);
      if (it.second) newTypes.insert(typeNode);
    }
  }
  for (const auto& typeNode : d_seenTypes)
  {
    const auto isNewType = ContainsKey(newTypes, typeNode);
    const size_t ground_terms_count = db->getNumTypeGroundTerms(typeNode);
    for (size_t j = 0; j < ground_terms_count; j++)
    {
      Node gt = db->getTypeGroundTerm(typeNode, j);
      const size_t phase = isNewType ? 0 : d_globalPhase;
      const auto it = d_globalPhases.insert({gt, phase});
      if (it.second)
        Trace("ml") << "[ml] register phase " << gt << "@" << phase
                    << std::endl;
    }
  }
  d_globalPhase++;
}

void QuantifierLogger::registerTermRegistry(TermRegistry& treg){
//  d_term_registry_store = &treg;
}
//TODO create a version that logs the created instantiation per round, for an RL type setting.
void QuantifierLogger::registerInstantiationRL(Node quantifier, const NodeVector& instantiation)
{
  Trace("quant") << "registerInstantiationRL: " << quantifier  << " | " << instantiation << std::endl;
//  std::cout << "registerInstantiationRL: " << quantifier  << " | " << instantiation << std::endl;
//  Assert(!successful || quantifier == d_currentInstantiationQuantifier);
  auto& qi = getQuantifierInfo(quantifier);

  Assert(ContainsKey(qi.d_allInstantiations, instantiation));
  d_round_instantiation_store[quantifier] = instantiation;

}

void QuantifierLogger::registerInstantiation(Node quantifier,
                                             bool successful,
                                             const NodeVector& instantiation)
{
  Trace("quant") << "registerInstantiation: " << quantifier << " | " << successful << " | " << instantiation << std::endl;
  Assert(!successful || quantifier == d_currentInstantiationQuantifier);
  auto& qi = getQuantifierInfo(quantifier);
  Assert(ContainsKey(qi.d_allInstantiations, instantiation));
  if (successful)
  {
    d_instantiationBodies.push_back(
        {quantifier, instantiation, d_currentInstantiationBody});
    qi.d_successfulInstantiations.insert(instantiation);
    d_currentInstantiationBody = Node::null();
    d_currentInstantiationQuantifier = Node::null();
  }
  else
  {
    qi.d_rejectedInstantiations.insert(instantiation);
  }
}

void QuantifierLogger::recordAssertions(
    const preprocessing::AssertionPipeline& assertions)
{
  Trace("quant") << "recordAssertions: " << std::endl;
  if (!d_env.getOptions().quantifiers.qselFormulaContext) return;
  for (const auto& n : assertions)
  {
    Trace("featurize") << "recordAssertions: " << n << std::endl;
//    std::cout << "assertions are being recorded: " << n << std::endl;
    d_assertionsBOW.count(n);

    d_assertion_list.push_back(n);
  }
}

NodeVector QuantifierLogger::registerInstantiationAttempt(
    Node quantifier, const std::vector<Node>& instantiation)
{
  Trace("quant") << "registerInstantiationAttempt: " << quantifier << " | " << instantiation << std::endl;
  NodeVector v(instantiation);
  getQuantifierInfo(quantifier).d_allInstantiations.insert(v);
  for (size_t vx = instantiation.size(); vx--;)
  {
    registerTryCandidate(quantifier, vx, instantiation[vx]);
  }
  return v;
}

void QuantifierLogger::registerCurrentInstantiationBody(Node quantifier,
                                                        Node body)
{
  Trace("quant") << "registerCurrentInstantiationBody: " << quantifier << " | " << body << std::endl;
  Assert(d_currentInstantiationBody.isNull()
         && d_currentInstantiationQuantifier.isNull());
  d_currentInstantiationQuantifier = quantifier;
  d_currentInstantiationBody = body;
}

bool QuantifierLogger::registerCandidate(Node quantifier,
                                         size_t varIx,
                                         Node candidate,
                                         bool relevant)
{
  Trace("quant") << "registerCandidate: " << quantifier << " | " << varIx << " | " << candidate << " | " << relevant << std::endl;
  auto& qi = getQuantifierInfo(quantifier);
  auto& candidates = qi.d_infos[varIx];
  auto [it, wasInserted] = candidates.try_emplace(candidate);
  if (wasInserted)
  {
    TermCandidateInfo& info = it->second;
    info.d_age = candidates.size();
    info.d_phase = qi.d_currentPhase;
    info.d_relevant = relevant;
    info.d_initialized = true;
  }
  return wasInserted;
}

void QuantifierLogger::registerTryCandidate(Node quantifier,
                                            size_t varIx,
                                            Node candidate)
{
  auto& vinfos = getQuantifierInfo(quantifier).d_infos;

  Assert(vinfos.size() == quantifier[0].getNumChildren());
  ASSERT_EXISTENCE(vinfos, candidate, varIx, quantifier);

  auto& cinfo = vinfos[varIx].at(candidate);
  Assert(cinfo.d_initialized);
  cinfo.d_tried++;
}
  
void QuantifierLogger::updateActiveQuantifier(Node quantifier)
{
  d_activeQuantifierSum++;
  d_activeQuantifiers[quantifier]++;
}

void QuantifierLogger::updateActiveQuantifiers(std::unordered_set<Node> active)
{
  Trace("active-fmls") << "*** " << active.size() << " active formulas BEFORE CHECK" << std::endl; 
  for (const Node& n : active)
  {
    bool isLemma = ((n.getKind() == kind::IMPLIES) && 
          (n[0].getKind() == kind::FORALL));
    if (isLemma)
    {
      updateActiveQuantifier(n[0]);
    }
    Trace("active-fmls") << "  " << n << std::endl;
  }
}

double QuantifierLogger::getActiveQuantBoost(Node quantifier)
{
  if (!d_activeQuantifierSum)
  {
    return 0;
  }
  return ((double)d_activeQuantifiers[quantifier]) / d_activeQuantifierSum;
}
  
std::ostream& QuantifierLogger::printActiveQuantifiers(std::ostream& out)
{
  if (!options().quantifiers.qloggingActive) 
  {
     return out;
  }
  out << "; Active_Quantifier_Count = " << d_activeQuantifiers.size() << std::endl;
  out << "; Active_Quantifier_Sum = " << d_activeQuantifierSum << std::endl;
  out << "; Active_Quantifier_List:" << std::endl;
  for (const auto &item : d_activeQuantifiers)
  {
    out << ";    " << item.second << " " << item.first << std::endl;
  }
  return out;
}

//static std::ostream& printTupleSet(std::ostream& out, const TupleSet& tuples)
//{
//  for (const auto& ns : tuples)
//  {
//    out << "    ( ";
//    std::copy(ns.begin(), ns.end(), std::ostream_iterator<Node>(out, " "));
//    out << ")" << std::endl;
//  }
//  return out;
//}
//
//#ifdef ML_USE_TUPLES
//std::ostream& QuantifierLogger::printTupleSample(
//    std::ostream& out,
//    Node quantifier,
//    const NodeVector& instantiation,
//    FeatureVector& featureVector,
//    const BOWCalculator& quantifierFeatures,
//    bool isUseful,
//    std::vector<std::map<Node, TermCandidateInfo>> termsInfo)
//{
//  out << "; " << isUseful << " : " << instantiation << std::endl;
//  Assert(instantiation.size() == termsInfo.size());
//  featureVector.push();
//  // featurize each term separately
//  for (size_t varIx = instantiation.size(); varIx--;)
//  {
//    const Node& term = instantiation[varIx];
//    ASSERT_EXISTENCE(termsInfo, term, varIx, quantifier);
//    const auto& candidateInfo = termsInfo[varIx].at(term);
//    featurizeTermBOW(
//        &featureVector, term, varIx, candidateInfo, quantifierFeatures);
//  }
//  printSample(out, isUseful, featureVector);
//  featureVector.pop();
//  return out;
//}
//
//std::ostream& QuantifierLogger::printTupleSamplesQuantifier(
//    std::ostream& out, Node quantifier, const QuantifierInfo& info)
//{
//  const auto& allInstantiations = info.d_allInstantiations;
//  const auto& termsInfo = info.d_infos;
//  const auto& usefulInstantiations = info.d_usefulInstantiations;
//  // calculate features for quantifier just once
//  BOWCalculator quantifierFeatures(true);
//  quantifierFeatures.count(quantifier);
//  FeatureVector featureVector(d_tupleFeatureProperties.get());
//  featurizeQuantifierBOW(&featureVector, quantifierFeatures);
//  out << "; Q : " << quantifier << std::endl;
//  // featurize all the tuples
//  for (const auto& instantiation : allInstantiations)
//  {
//    const bool useful =
//        ContainsKey(usefulInstantiations, NodeVector(instantiation));
//    printTupleSample(out,
//                     quantifier,
//                     instantiation,
//                     featureVector,
//                     quantifierFeatures,
//                     useful,
//                     termsInfo);
//  }
//  return out;
//}
//#else
//std::ostream& QuantifierLogger::printTupleSample(
//    std::ostream& out,
//    Node,
//    const NodeVector&,
//    FeatureVector&,
//    const BOWCalculator&,
//    bool,
//    std::vector<std::map<Node, TermCandidateInfo>>)
//{
//  AlwaysAssert(0);
//  return out;
//}
//
//std::ostream& QuantifierLogger::printTupleSamplesQuantifier(
//    std::ostream& out, Node, const QuantifierInfo&)
//{
//  AlwaysAssert(0);
//  return out;
//}
//
//#endif
//
//std::ostream& QuantifierLogger::printTupleSamples(std::ostream& out)
//{
//  out << "; TUPLE SAMPLES" << std::endl;
//  size_t maxSize = 0;
//  for (const auto& entry : d_infos)  // go through all quantifiers
//  {
//    printTupleSamplesQuantifier(out, entry.first, entry.second);
//    maxSize = std::max(maxSize, entry.second.d_infos.size());
//  }
//  out << "; FEATURE_NAMES up to " << maxSize << " vars" << std::endl;
//  const auto& properties =
//      static_cast<TermTupleFeatureProperties*>(d_tupleFeatureProperties.get());
//  const auto& names = properties->names();
//  for (size_t index = 0; index < properties->size(maxSize); index++)
//  {
//    out << " " << index << ":" << names[index];
//  }
//
//  return out << std::endl;
//}

bool QuantifierLogger::isFromInstantiation(Node n)
{
  if (!n.hasAttribute(cvc5::internal::SkolemFormAttribute())
      && (!n.hasAttribute(InstLevelAttribute())
          || n.getAttribute(InstLevelAttribute()) > 0))
  {
    const auto it = d_globalPhases.find(n);
    return it != d_globalPhases.end() && it->second > 0;
  }
  return false;
}

//void QuantifierLogger::calculateReasonScore()
//{
//  std::unordered_set<InstantiationExplanation,
//                     InstantiationExplanation_hash,
//                     InstantiationExplanation_equal>
//      open;
//  std::map<Node, InstantiationExplanationSet> tupleContents;
//  for (const auto& [quantifier, info] : d_infos)
//    for (const auto& instantiation : info.d_usefulInstantiations)
//      for (const auto& n : instantiation)
//        tupleContents[n].insert({quantifier, instantiation});
//
//  for (const auto& [quantifier, info] : d_infos)
//    for (const auto& instantiation : info.d_usefulInstantiations)
//      calculateReasonScoreRec({quantifier, instantiation}, tupleContents, open);
//}
//
//int QuantifierLogger::calculateReasonScoreRec(
//    const InstantiationExplanation& instantiation,
//    const std::map<Node, InstantiationExplanationSet>& tupleContents,
//    std::unordered_set<InstantiationExplanation,
//                       InstantiationExplanation_hash,
//                       InstantiationExplanation_equal>& open)
//{
//  auto& scores = d_infos.at(instantiation.d_quantifier).d_instantiationScores;
//  const auto it_scores = scores.find(instantiation.d_instantiation);
//  if (it_scores != scores.end())
//  {
//    Trace("ml") << "score " << instantiation << " @ " << it_scores->second
//                << std::endl;
//    return it_scores->second;
//  }
//  if (!open.insert(instantiation).second)
//  {
//    Trace("ml") << "explanation cycle on: " << instantiation << std::endl;
//    return 0;  //  cycle explanations
//  }
//
//  Trace("ml") << "scoring " << instantiation << std::endl;
//  int score = 1;
//  const auto it_reasons = d_reasons.find(instantiation);
//  if (it_reasons != d_reasons.end())
//  {
//    for (const auto& explained : it_reasons->second)
//    {
//      Trace("ml") << "explained term: " << explained << std::endl;
//      const auto it_contents = tupleContents.find(explained);
//      if (it_contents == tupleContents.end()) continue;
//      int tmp_score = 0;
//      for (const auto& child : it_contents->second)
//      {
//        const auto s = calculateReasonScoreRec(child, tupleContents, open);
//        if (s == 0)
//        {
//          tmp_score = 0;
//          break;
//        }
//        tmp_score += s;
//      }
//      if (tmp_score != 0) score += tmp_score;
//    }
//  }
//  Trace("ml") << "scoring " << instantiation << " @ " << score << std::endl;
//  scores.insert(it_scores, {instantiation.d_instantiation, score});
//  open.erase(instantiation);
//  return score;
//}
//
void QuantifierLogger::transitiveExplanation()
{
  Trace("ml") << "transitive explanation" << std::endl;
  // identify terms in the proof that appeared due to instantiations
  //std::unordered_set<TNode, TNodeHashFunction> needsExplaining;
  std::unordered_set<TNode, std::hash<TNode>> needsExplaining;
  for (const auto& entry : d_infos)
    for (const auto& i : entry.second.d_usefulInstantiations)
      for (const auto& n : i)
        if (isFromInstantiation(n)) needsExplaining.insert(n);

  Trace("ml") << "Needs explanation:" << needsExplaining << std::endl;
  size_t lastExplainedSize = 0;
  while (!needsExplaining.empty()
         && needsExplaining.size() != lastExplainedSize)
  {
    lastExplainedSize = needsExplaining.size();
    for (const auto& [quantifier, instantiation, body] : d_instantiationBodies)
      findExplanation({quantifier, instantiation}, body, needsExplaining);
  }
  Trace("ml") << "explanations marked" << std::endl;
  if (!needsExplaining.empty())
    Trace("ml") << "Warning, some things remained unexplained "
                << needsExplaining << std::endl;
  // mark instantiations in reasons as useful
  for (const auto& it : d_reasons)
  {
    d_infos.at(it.first.d_quantifier)
        .d_usefulInstantiations.insert(it.first.d_instantiation);
  }
  //if (options::mlParentsIncreasedScore())
  //{
  //  Trace("ml") << "calculateReasonScore:" << std::endl;
  //  calculateReasonScore();
  //}
  Trace("ml") << "Done explaining" << std::endl;
}

///* Look for nodes in needsExplaining in the current body. If some are found,
// * marke them as explained by this reason. If it happens to explain something,
// * the respective instantiation needs to be marked also to be explained.*/
void QuantifierLogger::findExplanation(
    InstantiationExplanation reason,
    Node body,
    std::unordered_set<TNode, std::hash<TNode>>& needsExplaining)
{
  Trace("ml") << "Explaining by " << body << std::endl;
  Trace("ml") << "from: " << reason << std::endl;
  std::vector<TNode> todo({body});
  bool explainsSomething = false;
  while (!needsExplaining.empty() && !todo.empty())
  {
    auto top(todo.back());
    todo.pop_back();
    if (top.getKind() == Kind::FORALL) continue;
    if (needsExplaining.erase(top) > 0)
    {
      Trace("ml") << top << " explained by " << reason << std::endl;
      d_reasons[reason].insert(top);
      [[maybe_unused]] const auto i = d_hasExplanation.insert(top);
      Assert(i.second);
      explainsSomething = true;
    }
    todo.insert(todo.end(), top.begin(), top.end());
  }
  // mark the instantiation leading to this body as something to be explained
  if (explainsSomething)
  {
    for (auto& n : reason.d_instantiation)
      if (isFromInstantiation(n) && !ContainsKey(d_hasExplanation, n))
        needsExplaining.insert(n);
  }
}

std::ostream& QuantifierLogger::printTermSamplesGNNRLNoInfos(
    std::ostream& out, Node quantifier, std::map<TypeNode, std::vector<Node>>& available_terms_of_type, NodeVector& tried_terms,  std::unordered_map<int,int>& node_norm_map
) {
  //TODO & vector
  //  const auto& usefulInstantiations = info.d_usefulInstantiations;

  const int variableCount = quantifier[0].getNumChildren();
  std::vector<std::set<Node>> triedPerVariable(variableCount);
  //

  //  {
//  std::cout <<
  Assert(static_cast<int>(tried_terms.size()) == variableCount);
  for (int varIx = 0; varIx < variableCount; varIx++)
  {
    const auto gt = tried_terms[varIx];
    triedPerVariable[varIx].insert(gt);
    //
  }
  //

  std::ostringstream oss;
  for (int varIx = 0; varIx < variableCount; varIx++) {
//    std::cout << "QB3: " << quantifier.getId() << std::endl;
    oss << "VARIABLE: " << quantifier[0][varIx].getId() << " " << node_norm_map.at(quantifier[0][varIx].getId()) << " " << quantifier[0][varIx].getType() << "\n";
    //      out << "KIND: " << quantifier[0][varIx].getKind() << std::endl;

    bool contains_used = false;
    for (const auto& term : available_terms_of_type.at(quantifier[0][varIx].getType())) {

      int used;
      // Either it was what happened or not.
      used = ContainsKey(triedPerVariable[varIx], term) ? 1 : 0;
//      out << "case2" << std::endl;

      if (used == 1) {
        contains_used = true;
      }

      // the assumption is that at least 1 term got used


      //        out << "Useful: " << useful << std::endl; //
      //        term.getId();
      //        std::cout << "termid exists" << std::endl;
      //        node_norm_map.at(term.getId()); // harbinger line
      //        std::cout << "termid exists in normmap" << std::endl;
//      out << "used: " << used << std::endl;
      if (ContainsKey(node_norm_map, term.getId())) {
          oss << "TERM: " << term.getId() << " "
              << node_norm_map.at(term.getId()) << " " << used << "\n";

      }
      else {
          oss << "node not in map: " << term << "--- " << used << "\n";
      }


      //        out << "KIND: " << term.getKind() << std::endl;
    }
    // something must happen.
//    if (! contains_used) {
//      out << "No term was used? Cannot be true." << std::endl;
//    }
//    std::cout << "End of termlist" << std::endl;



  }

  out << oss.str();
  return out;
}
//
//std::ostream& QuantifierLogger::printTermSamplesGNNRL(
//    std::ostream& out, Node quantifier, const QuantifierInfo& info, NodeVector tried_terms,  std::map<int,int>& node_norm_map
//) {
//  //TODO & vector
////  const auto& usefulInstantiations = info.d_usefulInstantiations;
//  const auto& termInfos = info.d_infos;
//  const auto variableCount = termInfos.size();
//
//  std::vector<std::set<Node>> triedPerVariable(variableCount);
////
//
////  {
//    Assert(tried_terms.size() == variableCount);
//    for (size_t varIx = 0; varIx < variableCount; varIx++)
//    {
//      const auto gt = tried_terms[varIx];
//      triedPerVariable[varIx].insert(gt);
////
//    }
////
//
//  for (size_t varIx = 0; varIx < variableCount; varIx++) {
//    out << "VARIABLE: " << quantifier[0][varIx].getId() << " " << node_norm_map.at(quantifier[0][varIx].getId()) << " " << quantifier[0][varIx].getType() << std::endl;
//    //      out << "KIND: " << quantifier[0][varIx].getKind() << std::endl;
//
//    bool contains_useful = false;
//    for (const auto& term_index : termInfos[varIx]) {
//      const Node& term = term_index.first;
//      const auto& candidateInfo = term_index.second;
//
//      int useful;
//      // never tried -> useless term; this is a choicepoint.
//      if (candidateInfo.d_tried == 0)
//      {
//        useful = 0;  // ignoring any terms that were never tried
//        out << "case1" << std::endl;
//      }
//      else
//      {
//        // is it in the current instantiation?
//        useful = ContainsKey(triedPerVariable[varIx], term) ? 1 : 0;
//        out << "case2" << std::endl;
//
//        if (useful == 1) {
//          contains_useful = true;
//        }
//      }
//      // the assumption is that at least 1 term got used
//
//
//      //        out << "Useful: " << useful << std::endl; //
//      //        term.getId();
//      //        std::cout << "termid exists" << std::endl;
//      //        node_norm_map.at(term.getId()); // harbinger line
//      //        std::cout << "termid exists in normmap" << std::endl;
//      out << "useful: " << useful << std::endl;
//      if (useful == 0) {
//        if (ContainsKey(node_norm_map, term.getId()))
//        {
//          out << "TERM: " << term.getId() << " "
//              << node_norm_map.at(term.getId()) << " " << useful << std::endl;
//        }
//        else {
//          out << "node not in map: " << term << std::endl;
//        }
//        //          else {
//        //            std::cout << "Unsuccesful term is unknown, value of d_tried: " << candidateInfo.d_tried << std::endl;
//        //          }
//      }
//
//      else {
//          out << "TERM: " << term.getId() << " " << node_norm_map.at(term.getId()) << " " << useful << std::endl;
//      }
//
//
//      //        out << "KIND: " << term.getKind() << std::endl;
//    }
//    // something must happen.
//    if (! contains_useful) {
//      out << "No term was used? Cannot be true." << std::endl;
//    }
//    //    std::cout << "Enf of termlist" << std::endl;
//
//
//
//  }
//
//  return out;
//}

//std::ostream& QuantifierLogger::printTermSamplesGNN(
//    std::ostream& out, Node quantifier, const QuantifierInfo& info, std::map<int,int>& node_norm_map
//    ) {
//  const auto& usefulInstantiations = info.d_usefulInstantiations;
//  const auto& termInfos = info.d_infos;
//  const auto variableCount = termInfos.size();
//
//  std::vector<std::set<Node>> usefulPerVariable(variableCount);
//
//  for (const auto& instantiation : usefulInstantiations)
//  {
//    Assert(instantiation.size() == variableCount);
//    for (size_t varIx = 0; varIx < variableCount; varIx++)
//    {
//      const auto gt = instantiation[varIx];
//      usefulPerVariable[varIx].insert(gt);
//
//    }
//  }
//
//  for (size_t varIx = 0; varIx < variableCount; varIx++) {
//      out << "VARIABLE: " << quantifier[0][varIx].getId() << " " << node_norm_map[quantifier[0][varIx].getId()] << " " << quantifier[0][varIx].getType() << std::endl;
////      out << "KIND: " << quantifier[0][varIx].getKind() << std::endl;
//      for (const auto& term_index : termInfos[varIx]) {
//        const Node& term = term_index.first;
//        const auto& candidateInfo = term_index.second;
//
//        int useful;
//        // never tried, but got a proof? useless term; this is a choicepoint.
//        if (candidateInfo.d_tried == 0)
//              {
//                useful = 0;  // ignoring any terms that were never tried
//              }
//        else
//              {
//                useful = ContainsKey(usefulPerVariable[varIx], term) ? 1 : 0;
//              }
////        out << "Useful: " << useful << std::endl; //
////        term.getId();
////        std::cout << "termid exists" << std::endl;
////        node_norm_map.at(term.getId()); // harbinger line
////        std::cout << "termid exists in normmap" << std::endl;
//        if (useful == 0) {
//          if (ContainsKey(node_norm_map, term.getId())){
//            out << "TERM: " << term.getId() << " " << node_norm_map.at(term.getId()) << " " << useful << std::endl;
//          }
////          else {
////            std::cout << "Unsuccesful term is unknown, value of d_tried: " << candidateInfo.d_tried << std::endl;
////          }
//        }
//        else {
//          out << "TERM: " << term.getId() << " " << node_norm_map.at(term.getId()) << " " << useful << std::endl;
//        }
//
//
////        out << "KIND: " << term.getKind() << std::endl;
//    }
////    std::cout << "Enf of termlist" << std::endl;
//
//
//
//  }
//
//  return out;
//}
//std::ostream& QuantifierLogger::printTermSamplesRLAtEnd(std::ostream& out, RLFeaturizerGraph& featurizer, std::map<Node, QuantifierInfo>& current_d_infos, std::map<Node, NodeVector>& current_round_inst_store)
//{
//
//  out << "; TERM SAMPLES" << std::endl;
//  //
//
//  out << "SAMPLE" << std::endl;
//
//  out << "NODE INITIALIZATIONS" << std::endl;
//  for (auto & node_kind : featurizer.nodeInits() ) {
//    out << node_kind << " ";
//
//  }
//  out << std::endl;
//
//  out << "EDGES" << std::endl;
//  //Trace("quant-sampler") << "# " << featurizer.graphEdges().size() << std::endl;
//
//  for (auto & edge : featurizer.graphEdges()) {
//    out << "[" << edge[0] << "," << edge[1] << "]" << std::endl;
//  }
//
//  out << "; END GRAPH" << std::endl;
//
//
//  for (const auto& [quantifier, info] : current_d_infos)
//  {
//    auto terms_tried = current_round_inst_store.at(quantifier);
//    printTermSamplesGNNRL(out, quantifier, info, terms_tried, featurizer.getNodeNormalizationMap());
//  }
//  out << "END VTL" << std::endl;
//  return out << std::endl;
//}
//
std::ostream& QuantifierLogger::printTermSamplesRL(std::ostream& out, RLFeaturizerGraph& featurizer, std::map<TypeNode, std::vector<Node>>& avail_terms, std::vector<int>& qlist, std::vector<int>& label_list)
{

  // CHECK IF THERE ARE ANY QUANTIFIERS WITH INSTANTIATIONS, OTHERWISE DON'T PRINT ANYTHING.
  // TODO: CHECK IF PRINTING INTO A STREAM FIRST IS FASTER.
  // TODO: CHECK IF \n ISTEAD OF ENDL IS FASTER BECAUSE OF LESS BUFFER CLEARING

  std::ostringstream oss;
//
  bool any_samples = false;
  for (const auto& item : d_round_instantiation_store) {

    auto terms_tried = d_round_instantiation_store.at(item.first);
    if (terms_tried.size() > 0) {
      any_samples = true;
    }

  }

  if (! any_samples) {
    return out << std::endl;
  }

  // START PRINTING
  oss << "; TERM SAMPLES" << "\n";
  //

  oss << "SAMPLE" << "\n";

  oss << "NODE INITIALIZATIONS" << "\n";

  for (auto & node_kind : featurizer.nodeInits() ) {
    oss << node_kind << " ";

  }
  oss << "\n";

  if (! (qlist.size() == label_list.size() )) {
    std::cout << "WARNING: qs and label size do not match." << std::endl;
  }
  oss << "TARGETS" << std::endl;
  for (auto & target : qlist) {
    oss << target << " ";

  }

  oss <<  "\n";
  oss << "LABELS" << std::endl;
  for (auto & lab : label_list) {
    oss << lab << " ";

  }
  oss <<  "\n";


  // Perhaps have different edgetypes here?
  // normal, sibling, types

  oss << "EDGES" << "\n";
  //Trace("quant-sampler") << "# " << featurizer.graphEdges().size() << std::endl;
  const bool edge_features = true;
  if (not edge_features)
  {
    for (auto& edge : featurizer.graphEdges())
    {
      oss << "[" << edge[0] << "," << edge[1] << "]"
          << "\n";
    }
  }
  else{
    for (auto& edge : featurizer.graphEdges())
    {
      oss << "[" << edge[0] << "," << edge[1] << "," << edge[2]  << "]"
          << "\n";
    }
  }
  oss << "; END GRAPH" << "\n";

  std::cout << oss.str() ;

//  out << "quantifiers d_info size " << d_infos.size() << std::endl;


//  for (const auto& [quantifier, info] : d_infos)
//  {
//    auto terms_tried = d_round_instantiation_store.at(quantifier);
//
//    out << "mapsizes: feat" << featurizer.getNodeNormalizationMap().size() << std::endl;
////    printTermSamplesGNNRL(out, quantifier, info, terms_tried, featurizer.getNodeNormalizationMap());
//    printTermSamplesGNNRLNoInfos(out, quantifier, avail_terms, terms_tried, featurizer.getNodeNormalizationMap());
//  }

  // only do quants that had a successful instantiation - not sure if the best choice.
  for (const auto& item : d_round_instantiation_store) {
//    std::cout << "QB2: " << item.first.getId() << std::endl;
//    std::cout << "SIZE: " << d_round_instantiation_store.size() << " RIN: ";

//    for (auto& [qstored, inst] : d_round_instantiation_store) {
//      std::cout << qstored.getId() << " ";
//    }
//    std::cout << std::endl;
    auto terms_tried = d_round_instantiation_store.at(item.first);
    if (terms_tried.size() > 0)
    {
      printTermSamplesGNNRLNoInfos(out,
                                   item.first,
                                   avail_terms,
                                   terms_tried,
                                   featurizer.getNodeNormalizationMap());
    }
  }
  out << "END VTL" << std::endl;
  return out << std::endl;
}

//std::ostream& QuantifierLogger::printTermSamples(std::ostream& out)
//{
//  // FIRST, WE PRINT THE WHOLE GRAPH STRUCTURE;
//  // Print quantifiers, assertions, instantiations
//
//  // keep in mind that we need to normalize the ids
//
//  // AFTERWARDS, WE PRINT WHICH TERMS WERE USEFUL/NOT USEFUL FOR EACH VARIABLE.
//
//  out << "; SAMPLES" << std::endl;
//
//  // going through all quantifiers
//  auto assertion_list = getAssertionList();
//  auto instantiation_list =
//      QuantifierLogger::s_logger->getInstantiations();
//
//  std::vector<Node> inst_list;
//
//  for (auto& inst_info : instantiation_list)
//  {
//    inst_list.push_back(inst_info.d_body);
//  }
//
//  std::vector<Node> quantifier_list;
//  std::vector<int> label_vector;
//  std::vector<Node> term_list;
////  std::cout << "Looking for TERMDB " << std::endl;
////  auto termdb = QuantifierLogger::s_logger->d_term_r
///
///
/// egistry_store->getTermDatabase();
//  auto termdb = QuantifierLogger::s_logger->d_term_database_store;
////  std::cout << "TERMDB apprehended" << std::endl;
//  std::set<TypeNode> type_set;
//
//
//  // std::map<int, int> target_index_to_label;
//
//  for (const auto& [quantifier, info] : d_infos)
//  {
//    quantifier_list.push_back(quantifier);
//
//    const int label = info.d_usefulInstantiations.empty() ? 0 : 1;
//    label_vector.push_back(label);
//    // target_index_to_label[static_cast<int>(quantifier.getId())] = label;
//  }
//  for (auto& quant : quantifier_list) {
//
//    auto num_vars = quant[0].getNumChildren();
//
//    for (int varindex = 0; varindex < static_cast<int>(num_vars); ++varindex) {
//      type_set.insert(quant[0][varindex].getType());
//    }
//  }
//  //        std::vector<Node> term_list;
////  std::cout << "TYPESET " << type_set.size() << std::endl;
//  for (auto& type : type_set) {
//    const size_t ground_terms_count = termdb->getNumTypeGroundTerms(type);
////    std::cout << "TYHPE: " << ground_terms_count << std::endl;
//    for (size_t j = 0; j < ground_terms_count; j++) {
//      //            std::cout << " " << std::endl;
//      Node gt = termdb->getTypeGroundTerm(type, j);
//      term_list.push_back(gt);
//    }
//  }
//
//  RLFeaturizerGraph featurizer;
////  std::cout << "Logger counts: AS: " << assertion_list.size() << " QS: " << quantifier_list.size() << " I: " << inst_list.size() << " Ts: " << term_list.size() << std::endl;
//  featurizer.init(assertion_list, quantifier_list, inst_list, term_list);
//  featurizer.run();
//
//  out << "; QUANTIFIER SAMPLES" << std::endl;
////
//
//  out << "SAMPLE" << std::endl;
//
//
//  out << "LABELS" << std::endl;
//  for (auto & lab : label_vector) {
//    out << lab << " ";
//
//  }
//  out << std::endl;
//
//  out << "TARGETS" << std::endl;
//  for (auto & target : featurizer.getNormalizedTargets() ) {
//    out << target << " ";
//
//  }
//  out << std::endl;
//
//  out << "NODE INITIALIZATIONS" << std::endl;
//  for (auto & node_kind : featurizer.nodeInits() ) {
//    out << node_kind << " ";
//
//  }
//  out << std::endl;
//
//  out << "EDGES" << std::endl;
//  //Trace("quant-sampler") << "# " << featurizer.graphEdges().size() << std::endl;
//
//  for (auto & edge : featurizer.graphEdges()) {
//    out << "[" << edge[0] << "," << edge[1] << "]" << std::endl;
//  }
//  out << "; END QUANTIFIER SAMPLES" << std::endl;
//
//
//  for (const auto& [quantifier, info] : d_infos)
//  {
//    printTermSamplesGNN(out, quantifier, info, featurizer.getNodeNormalizationMap());
//  }
//  out << "END VTL" << std::endl;
//  return out << std::endl;
//}

//std::ostream& QuantifierLogger::printTermSamplesQuantifier(
//    std::ostream& out, Node quantifier, const QuantifierInfo& info)
//{
//  const auto& usefulInstantiations = info.d_usefulInstantiations;
//  const auto& termInfos = info.d_infos;
//  const auto variableCount = termInfos.size();
//  // calculate features for quantifier just once
//
//  const bool useMatrices{QuantifierLogger::s_logger->d_matrices.get()
//                         != nullptr};
//  std::unique_ptr<ITermFeaturizer> featurizer(
//      useMatrices ? static_cast<ITermFeaturizer*>(
//          new TermFeaturizerMx(quantifier, d_termFeatureProperties.get()))
//                  : static_cast<ITermFeaturizer*>(new TermFeaturizerBOW(
//                      quantifier, d_termFeatureProperties.get())));
//
//  featurizer->initialize();
//
//  out << "; Q : " << quantifier << std::endl;
//  printAnonymous(out << "; Q : ", quantifier) << std::endl;
//  printAnonymousTree(out << "; Q : ", quantifier) << std::endl;
//  printTree(out << "; Q : ", quantifier, quantifier) << std::endl;
//  for (size_t cursor = 0; cursor < variableCount; cursor++)
//  {
//    printAnonymous(
//        out << "; var" << cursor << " : ", quantifier, quantifier, cursor)
//        << std::endl;
//    printAnonymousTree(
//        out << "; var" << cursor << " : ", quantifier, quantifier, cursor)
//        << std::endl;
//    printTree(out << "; var" << cursor << " : ", quantifier, quantifier, cursor)
//        << std::endl;
//  }
//
//  if (0 && options::mlParentsIncreasedScore())
//  {
//    const auto& scores = info.d_instantiationScores;
//    for (const auto& [instantiation, score] : scores)
//      out << "; SCORE : " << instantiation << " : " << score << std::endl;
//  }
//
//  // for each variable calculate if a term was ever useful
//  std::vector<std::set<Node>> usefulPerVariable(variableCount);
//  std::vector<std::map<Node, int>> scorePerVariable(variableCount);
//  for (const auto& instantiation : usefulInstantiations)
//  {
//    Assert(instantiation.size() == variableCount);
//    const auto instantiation_score =
//        options::mlParentsIncreasedScore()
//            ? info.d_instantiationScores.at(instantiation)
//            : 0;
//    for (size_t varIx = 0; varIx < variableCount; varIx++)
//    {
//      const auto gt = instantiation[varIx];
//      usefulPerVariable[varIx].insert(gt);
//      if (options::mlParentsIncreasedScore())
//      {
//        const auto scores_it = scorePerVariable[varIx].insert({gt, 0});
//        scores_it.first->second += instantiation_score;
//      }
//    }
//  }
//
//  // go through all the variables and terms
//  for (size_t varIx = 0; varIx < variableCount; varIx++)
//  {
//    for (const auto& term_index : termInfos[varIx])
//    {
//      const Node& term = term_index.first;
//      const auto& candidateInfo = term_index.second;
//      if (candidateInfo.d_tried == 0)
//      {
//        continue;  // ignoring any terms that were never tried
//      }
//
//      const int useful = ContainsKey(usefulPerVariable[varIx], term) ? 1 : 0;
//      const int score = (options::mlParentsIncreasedScore() && useful)
//                            ? scorePerVariable[varIx].at(term)
//                            : useful;
//      out << "; T : " << term << " @ var" << varIx << " = " << score
//          << std::endl;
//      printAnonymous(out << "; T : ", term)
//          << " @ var" << varIx << " = " << score << std::endl;
//      printAnonymousTree(out << "; T : ", term)
//          << " @ var" << varIx << " = " << score << std::endl;
//      printTree(out << "; T : ", term)
//          << " @ var" << varIx << " = " << score << std::endl;
//      featurizer->addTerm(term, varIx, candidateInfo);
//      printSample(out, score, featurizer->featureVector());
//      featurizer->removeTerm();
//    }
//  }
//  return out;
//}


std::ostream& QuantifierLogger::printQuantifiersSamples(std::ostream& out)
{
//  AlwaysAssert(!d_matrices);
  out << "; QUANTIFIER SAMPLES" << std::endl;
  std::unique_ptr<QSelFeaturizerBOW> featurizer =
      std::make_unique<QSelFeaturizerBOW>(d_qselFeatureProperties.get());
  if (d_env.getOptions().quantifiers.qselFormulaContext)
  {
    featurizer->addContext(assertionsBOW());
  }
  // going through all quantifiers
  for (const auto& [quantifier, info] : d_infos)
  {
    featurizer->pushQuantifier(quantifier);
    const int label = info.d_usefulInstantiations.empty() ? 0 : 1;
    out << "; Q : " << quantifier << " " << label << " " << std::endl;
    printSample(out, label, featurizer->featureVector());
    featurizer->pop();
  }
  out << "; END QUANTIFIER SAMPLES" << std::endl;
//
//  if (d_matrices)
//    out << "; QUANTIFIER MX FEATURE_NAMES" << std::endl;
//  else
  out << "; QUANTIFIER FEATURE_NAMES" << std::endl;
  d_qselFeatureProperties->printNames(out) << std::endl;
  return out << std::endl;
}

//std::ostream& QuantifierLogger::printTermSamples(std::ostream& out)
//{
//  out << "; SAMPLES" << std::endl;
//  // going through all quantifiers
//  for (const auto& [quantifier, info] : d_infos)
//  {
//    printTermSamplesQuantifier(out, quantifier, info);
//  }
//
//  if (d_matrices)
//    out << "; MX FEATURE_NAMES" << std::endl;
//  else
//    out << "; FEATURE_NAMES" << std::endl;
//  d_termFeatureProperties->printNames(out) << std::endl;
//  return out << std::endl;
//}
//
//std::ostream& QuantifierLogger::printExtensive(std::ostream& out)
//{
//  std::set<Node> useful_terms;
//  std::set<Node> all_candidates;
//  out << "(quantifier_candidates " << std::endl;
//  for (const auto& entry : d_infos)
//  {
//    if (entry.second.d_allInstantiations.empty()) continue;
//    const auto& quantifier = entry.first;
//    const auto& usefulInstantiations = entry.second.d_usefulInstantiations;
//    const auto& successfulInstantiations =
//        entry.second.d_successfulInstantiations;
//    const auto& rejectedInstantiations = entry.second.d_rejectedInstantiations;
//    const auto name = quantifier;
//    const auto& infos = entry.second.d_infos;
//    const auto variableCount = infos.size();
//
//    // count how many times a term was ever useful for each variable
//    std::vector<std::map<Node, int32_t>> usefulPerVariable(variableCount);
//    for (const auto& instantiation : usefulInstantiations)
//    {
//      Assert(instantiation.size() == variableCount);
//      for (size_t varIx = 0; varIx < variableCount; varIx++)
//      {
//        const auto& term = instantiation[varIx];
//        auto [it, wasInserted] = usefulPerVariable[varIx].insert({term, 1});
//        if (!wasInserted)
//        {
//          it->second++;
//        }
//      }
//    }
//
//    out << "(candidates " << name << " " << std::endl;
//    for (size_t varIx = 0; varIx < variableCount; varIx++)
//    {
//      out << "  (variable " << varIx;
//      for (const auto& term_index : infos[varIx])
//      {
//        const Node& term = term_index.first;
//        const auto& candidateInfo = term_index.second;
//        if (candidateInfo.d_tried == 0)
//        {
//          continue;
//        }
//
//        all_candidates.insert(term);
//        const auto i = usefulPerVariable[varIx].find(term);
//        const auto termUseful =
//            (i == usefulPerVariable[varIx].end()) ? 0 : i->second;
//        out << " (candidate " << term;
//        out << " (age " << candidateInfo.d_age << ")";
//        out << " (phase " << candidateInfo.d_phase << ")";
//        out << " (relevant " << candidateInfo.d_relevant << ")";
//        out << " (depth " << quantifiers::TermUtil::getTermDepth(term) << ")";
//        out << " (tried " << candidateInfo.d_tried << ")";
//        out << " (useful " << termUseful << ")";
//        out << ")";  // close candidate
//      }
//      out << "  )" << std::endl;  // close variable
//    }
//
//    printTupleSet(out << "  (rejected_instantiations " << std::endl,
//                  rejectedInstantiations)
//        << "  ) " << std::endl;
//    printTupleSet(out << "  (successful_instantiations " << std::endl,
//                  successfulInstantiations)
//        << "  ) " << std::endl;
//    printTupleSet(out << "  (useful_instantiations " << std::endl,
//                  usefulInstantiations)
//        << "  )" << std::endl;
//    out << ")" << std::endl;  // close candidates
//  }
//  return out << ")" << std::endl;  //  close everything
//}
//

//----graphs----
std::ostream& QuantifierLogger::printQuantifierGraphSample(
    std::ostream& out, Node quantifier, const QuantifierInfo& info)
{
  std::cout << "SAMPLE" << std::endl;
  const int label = info.d_usefulInstantiations.empty() ? 0 : 1;

  std::cout << "LABEL: " << label << std::endl;
  std::unique_ptr<QuantifierFeaturizerGraph> featurizer =
      std::make_unique<QuantifierFeaturizerGraph>(
          quantifier);
  featurizer->run();

  // print the node inits
  std::cout << "NODE INITIALIZATIONS" << std::endl;
  for (auto & node_init : featurizer->nodeInits()) {
    std::cout << node_init << " ";

  }
  std::cout << std::endl;
  std::cout << "EDGES" << std::endl;
  // Trace("quant-sampler") << "[ " ;
  for (auto & edge : featurizer->graphEdges()) {
    std::cout << "[" << edge[0] << "," << edge[1] << "]" << std::endl;
  }
  // : std::vector<std::vector<int>> d_graphEdges;

  // Trace("quant-sampler") << " ]" << std::endl;
  // std::cout << "---" << std::endl;
  return out;

}


std::ostream& QuantifierLogger::printQuantifierListGraphSamples(std::ostream& out)
{
  std::vector<Node> quantifier_list;
  std::vector<int> label_vector;
  // std::map<int, int> target_index_to_label;

  for (const auto& [quantifier, info] : d_infos)
  {
    quantifier_list.push_back(quantifier);

    const int label = info.d_usefulInstantiations.empty() ? 0 : 1;
    label_vector.push_back(label);
    // target_index_to_label[static_cast<int>(quantifier.getId())] = label;
  }

  QuantifierListFeaturizerGraph featurizer(quantifier_list);
  featurizer.run();

  out << "; QUANTIFIER SAMPLES" << std::endl;


  out << "SAMPLE" << std::endl;


  out << "LABELS" << std::endl;
  for (auto & lab : label_vector) {
    out << lab << " ";

  }
  out << std::endl;

  out << "TARGETS" << std::endl;
  for (auto & target : featurizer.getNormalizedTargets() ) {
    out << target << " ";

  }
  out << std::endl;



  out << "NODE INITIALIZATIONS" << std::endl;
  for (auto & node_kind : featurizer.nodeInits() ) {
    out << node_kind << " ";

  }
  out << std::endl;


  out << "EDGES" << std::endl;
  //Trace("quant-sampler") << "# " << featurizer.graphEdges().size() << std::endl;

  for (auto & edge : featurizer.graphEdges()) {
    out << "[" << edge[0] << "," << edge[1] << "]" << std::endl;
  }
  out << "; END QUANTIFIER SAMPLES" << std::endl;




  return out << std::endl;

}

std::ostream& QuantifierLogger::printQuantifierPlusAssertionsListGraphSamples(std::ostream& out)
{
  auto assertion_list = getAssertionList();
  std::vector<Node> quantifier_list;
  std::vector<int> label_vector;


  // std::map<int, int> target_index_to_label;

  for (const auto& [quantifier, info] : d_infos)
  {
    quantifier_list.push_back(quantifier);

    const int label = info.d_usefulInstantiations.empty() ? 0 : 1;
    label_vector.push_back(label);
    // target_index_to_label[static_cast<int>(quantifier.getId())] = label;
  }

  QuantifierPlusAssertionsListFeaturizerGraph featurizer(assertion_list, quantifier_list);
  featurizer.run();

  out << "; QUANTIFIER SAMPLES" << std::endl;


  out << "SAMPLE" << std::endl;


  out << "LABELS" << std::endl;
  for (auto & lab : label_vector) {
    out << lab << " ";

  }
  out << std::endl;

  out << "TARGETS" << std::endl;
  for (auto & target : featurizer.getNormalizedTargets() ) {
    out << target << " ";

  }
  out << std::endl;



  out << "NODE INITIALIZATIONS" << std::endl;
  for (auto & node_kind : featurizer.nodeInits() ) {
    out << node_kind << " ";

  }
  out << std::endl;


  out << "EDGES" << std::endl;
  //Trace("quant-sampler") << "# " << featurizer.graphEdges().size() << std::endl;

  for (auto & edge : featurizer.graphEdges()) {
    out << "[" << edge[0] << "," << edge[1] << "]" << std::endl;
  }
  out << "; END QUANTIFIER SAMPLES" << std::endl;

  return out << std::endl;

}




std::ostream& QuantifierLogger::printRLStyleSamples(std::ostream& out, RLFeaturizerGraph& featurizer, std::vector<int> used_qs)
{

//  QuantifierPlusAssertionsListFeaturizerGraph featurizer(assertion_list, quantifier_list);
//  featurizer.run();

  out << "; QUANTIFIER SAMPLE" << std::endl;

  out << "SAMPLE" << std::endl;


  out << "LABELS" << std::endl;
  for (auto & lab : used_qs) {
    out << lab << " ";

  }
  out << std::endl;

  out << "TARGETS" << std::endl;
  out << "TARGETS # : " << featurizer.getNormalizedTargets().size() << std::endl;
  for (auto & target : featurizer.getNormalizedTargets() ) {
    out << target << " ";

  }
  out << std::endl;

  out << "NODE INITIALIZATIONS" << std::endl;
  for (auto & node_kind : featurizer.nodeInits() ) {
    out << node_kind << " ";

  }
  out << std::endl;


  out << "EDGES" << std::endl;
  //Trace("quant-sampler") << "# " << featurizer.graphEdges().size() << std::endl;

  for (auto & edge : featurizer.graphEdges()) {
    out << "[" << edge[0] << "," << edge[1] << "]" << std::endl;
  }
  out << "; END QUANTIFIER SAMPLE" << std::endl;


  return out << std::endl;

}


std::ostream& QuantifierLogger::printQuantifiersGraphSamples(std::ostream& out)
{

  out << "; QUANTIFIER SAMPLES" << std::endl;
  // going through all quantifiers
  //
  // when using list features we have to do something else
  for (const auto& [quantifier, info] : d_infos)
  {
    printQuantifierGraphSample(out, quantifier, info);
  }
  out << "; END QUANTIFIER SAMPLES" << std::endl;

  return out << std::endl;
}

//--------------


std::ostream& QuantifierLogger::print(std::ostream& out)
{
  Trace("quant") << "print: " << std::endl;
  if (d_env.getOptions().quantifiers.mlParents) transitiveExplanation();
  /* printExtensive(out); */
//  printTermSamples(out);
//  printQuantifiersSamples(out);

  // graph stuff
  // printQuantifiersGraphSamples(out);
//  printQuantifierListGraphSamples(out);


//  if (! d_env.getOptions().quantifiers.QuantifierRLSamples)
//  {
//     if( ! options().quantifiers.QuantifierListUseAssertions)
//     {
//       printQuantifierListGraphSamples(out);
//     }
//     else {
//       printQuantifierPlusAssertionsListGraphSamples(out);
//
//     }
//  }
//  Assert(d_infos_store_vector.size()  == d_round_instantiation_store_vector.size());
//  Assert(d_round_instantiation_store_vector.size() == d_featurizers_store_vector.size());
//  for (int i=0; i < static_cast<int>(d_infos_store_vector.size()); ++i) {
//    printTermSamplesRLAtEnd(out, d_featurizers_store_vector[i], d_infos_store_vector[i], d_round_instantiation_store_vector[i]);
//  }
//  out << "ENDOFSAMPLESHERE" << std::endl;

  // TODO this should probably not happen in RL mode.
//  printTermSamples(out);

  /* printTupleSamples(out); */

//  printActiveQuantifiers(out);

  return out;
}
}  // namespace quantifiers
}  // namespace theory
}  // namespace cvc5
