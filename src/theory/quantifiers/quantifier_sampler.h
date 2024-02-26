/*
 * File:  quantifier_sampler.h
 * Author:  mikolas
 * Created on:  Thu Jan 27 10:58:00 CET 2022
 * Copyright (C) 2022, Mikolas Janota
 */
#ifndef QUANTIFIER_SAMPLER_H_24262
#define QUANTIFIER_SAMPLER_H_24262
#include <random>
#include <vector>

#include "options/quantifiers_options.h"
#include "theory/quantifiers/instantiate.h"
#include "theory/quantifiers/quantifier_logger.h"
#include "theory/quantifiers/relevant_domain.h"
#include "theory/quantifiers/term_database.h"
#include "theory/quantifiers/term_tuple_enumerator.h"
//#include "theory/quantifiers/term_tuple_enumerator_ml.h"
#include "theory/quantifiers/term_util.h"

using namespace cvc5::internal::kind;
using namespace cvc5::context;

namespace cvc5::internal {
namespace theory {
namespace quantifiers {

class IQuantifierSampler
{
 public:
  virtual ~IQuantifierSampler() {}
  struct QuantifierInfo
  {
    size_t d_originalIndex;
    size_t d_localIndex;
    Node d_quantifier;
    double d_prediction;
  };

  const std::vector<QuantifierInfo>& get_quantifiers() const
  {
    return d_quantifiers;
  }

  void addQuantifier(size_t index, Node quantifier, double prediction)
  {
    d_quantifiers.push_back(
        {index, d_quantifiers.size(), quantifier, prediction});
  }

  virtual bool choose(std::mt19937& mt, const QuantifierInfo& q) = 0;
  virtual void close() = 0;

 protected:
  bool d_closed = false;
  std::vector<QuantifierInfo> d_quantifiers;
};

class TrivialQuantifierSampler : public IQuantifierSampler
{
 public:
  virtual bool choose(std::mt19937&, const QuantifierInfo&) override
  {
    Assert(d_closed);
    return true;
  }
  virtual void close() override { d_closed = true; }
};

class QuantifierSampler : public IQuantifierSampler
{
 public:
  virtual bool choose(std::mt19937& mt, const QuantifierInfo& q) override;
  virtual void close() override;

 protected:
  bool d_closed = false;
  double d_maxPrediction;
  typedef std::uniform_real_distribution<double> DistributionType;
  std::unique_ptr<DistributionType> d_distributionSel;
  std::unique_ptr<DistributionType> d_distributionEpsilon;
};

class SoftmaxQuantifierSampler : public IQuantifierSampler
{
 public:
  virtual bool choose(std::mt19937& mt, const QuantifierInfo& q) override;
  virtual void close() override;

  std::vector<double> d_probabilities;

 protected:
  bool d_closed = false;

  std::unique_ptr<std::discrete_distribution<int>> d_distributionSel;
  int chosen_index;


  //  std::unique_ptr<DistributionType> d_distributionEpsilon;
};

class MaxQuantifierChooser : public IQuantifierSampler
{
 public:
  virtual bool choose(std::mt19937& mt, const QuantifierInfo& q) override;
  virtual void close() override;

  std::vector<double> d_probabilities;

 protected:
  bool d_closed = false;

//  std::unique_ptr<std::discrete_distribution<int>> d_distributionSel;
  int chosen_index;


  //  std::unique_ptr<DistributionType> d_distributionEpsilon;
};



//
class HardThresholdQuantifierSampler : public IQuantifierSampler
{
 public:
  virtual bool choose(std::mt19937& mt, const QuantifierInfo& q) override;
  virtual void close() override;

 protected:
  bool d_closed = false;
//  double d_maxPrediction;
//  typedef std::uniform_real_distribution<double> DistributionType;
//  std::unique_ptr<DistributionType> d_distributionSel;
//  std::unique_ptr<DistributionType> d_distributionEpsilon;
};

}  // namespace quantifiers
}  // namespace theory
}  // namespace cvc5
#endif /* QUANTIFIER_SAMPLER_H_24262 */
