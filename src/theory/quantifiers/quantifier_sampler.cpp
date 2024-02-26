/*
 * File:  quantifier_sampler.cpp
 * Author:  mikolas
 * Created on:  Thu Jan 27 10:58:04 CET 2022
 * Copyright (C) 2022, Mikolas Janota
 */
#include "theory/quantifiers/quantifier_sampler.h"
using namespace cvc5::internal::kind;
using namespace cvc5::context;
#include <algorithm>
#include <limits>

#include "options/quantifiers_options.h"

namespace cvc5::internal {
namespace theory {
namespace quantifiers {

void QuantifierSampler::close()
{
  d_closed = true;
  d_maxPrediction = std::numeric_limits<double>::lowest();
  for (const auto& q : d_quantifiers)
    d_maxPrediction = std::max(d_maxPrediction, q.d_prediction);
  d_distributionSel = std::make_unique<DistributionType>(0, d_maxPrediction);
  d_distributionEpsilon = std::make_unique<DistributionType>(0, 1);
}

bool QuantifierSampler::choose(std::mt19937& mt, const QuantifierInfo& q)
{
  Assert(d_closed) << "close method needs to be run before choosing";
  const Options& options = QuantifierLogger::s_logger->options();
  const auto epsilon = options.quantifiers.qselEpsilon;
  const auto& pred = q.d_prediction;
  const auto retv =
      (std::fpclassify(epsilon) != FP_ZERO
       && (*d_distributionEpsilon)(mt) < epsilon)
      || (std::fpclassify(pred) != FP_ZERO && (*d_distributionSel)(mt) < pred);
  Trace("quant-sampler") << "[quant-sample] " << q.d_quantifier << " : "
                         << (retv ? "YES" : "NO") << std::endl;
  return retv;
}

void SoftmaxQuantifierSampler::close()
{

//  int max_index = static_cast<int>(d_quantifiers.size()) - 1;
  std::default_random_engine generator;
  generator.seed(std::chrono::high_resolution_clock::now().time_since_epoch().count());
  d_distributionSel = std::make_unique<std::discrete_distribution<int>>(d_probabilities.begin(), d_probabilities.end());
  chosen_index = (*d_distributionSel)(generator);
  d_closed = true;
}

bool SoftmaxQuantifierSampler::choose(std::mt19937& mt, const QuantifierInfo& q)
{
  Assert(d_closed) << "close method needs to be run before choosing";
  std::cout << "OrigIndex: " << q.d_originalIndex << " Chosen: " << chosen_index << " with prob: " << d_probabilities[chosen_index] << std::endl;
  const auto retv = (static_cast<int>(q.d_originalIndex) == chosen_index);

  return retv;
}

void MaxQuantifierChooser::close()
{
  chosen_index = std::distance(d_probabilities.begin(),std::max_element(d_probabilities.begin(), d_probabilities.end()));
  d_closed = true;
} //

bool MaxQuantifierChooser::choose(std::mt19937& mt, const QuantifierInfo& q)
{
  Assert(d_closed) << "close method needs to be run before choosing";
  std::cout << "OrigIndex: " << q.d_originalIndex << " Chosen: " << chosen_index << " with prob: " << d_probabilities[chosen_index] << std::endl;
  const auto retv = (static_cast<int>(q.d_originalIndex) == chosen_index);

  return retv;
}


void HardThresholdQuantifierSampler::close()
{
  d_closed = true;

}
//
bool HardThresholdQuantifierSampler::choose(std::mt19937& mt, const QuantifierInfo& q)
{
  Assert(d_closed) << "close method needs to be run before choosing";
////  const Options& options = QuantifierLogger::s_logger->options();
////  const auto epsilon = options.quantifiers.qselEpsilon;
  const auto& pred = q.d_prediction;
  const auto retv =  pred > 0.00001;
////      (std::fpclassify(epsilon) != FP_ZERO
////       && (*d_distributionEpsilon)(mt) < epsilon)
////      || (std::fpclassify(pred) != FP_ZERO && (*d_distributionSel)(mt) < pred);
  Trace("quant-sampler") << "[quant-sample] " << q.d_quantifier << " : "
                         << (retv ? "YES" : "NO") << std::endl;
  return retv;
}

}  // namespace quantifiers
}  // namespace theory
}  // namespace cvc5
