/*
 * File:  src/theory/quantifiers/ml.h
 * Author:  mikolas
 * Created on:  Thu Jan 14 11:55:09 CET 2021
 * Copyright (C) 2021, Mikolas Janota
 */

#ifndef CVC4__ML
#define CVC4__ML

//#include <cmath>
//#include <memory>
#include <string>
#include <vector>

//#include "base/check.h"
//#include "embedding_matrices.h"
//#include "expr/node.h"
#include "lightgbm.h"

//#ifdef CVC5_TCPML
//class tcp_client_t;
//
//class TCPClient
//{
// public:
//  static TCPClient s_client;
//  virtual ~TCPClient();
//  static int sopen() { return s_client.open(); }
//  static void sclose() { return s_client.close(); }
//  static int send(const char* message)
//  {
//    return s_client.send_internal(message);
//  }
//  static std::string receive() { return s_client.receive_internal(); }
//
// private:
//  TCPClient();
//  const unsigned short d_port = 8080;
//  const char d_server[255] = "127.0.0.1";  // server host name or IP
//  bool d_isOpen = false;
//  std::unique_ptr<tcp_client_t> d_client;
//  int open();
//  void close();
//  int send_internal(const char* message);
//  std::string receive_internal();
//};
//
//class TCPReader
//{
// public:
//  TCPReader() {}
//  virtual ~TCPReader() {}
//  int predict(const char* message);
//  const std::vector<std::vector<double> >& predictions() const
//  {
//    return d_predictions;
//  }
//
// private:
//  std::vector<std::vector<double> > d_predictions;
//  int receive();
//};
//#endif

namespace cvc5::internal {
class PredictorInterface
{
 public:
  const std::string d_modelFile;
  PredictorInterface(const std::string& modelFile) : d_modelFile(modelFile) {}
  virtual ~PredictorInterface() {}
  virtual double predict(const std::vector<float>& features) = 0;
  virtual size_t numberOfFeatures() const = 0;
};

//class Sigmoid : public PredictorInterface
//{
// public:
//  Sigmoid(const std::string& modelFile);
//  virtual ~Sigmoid() {}
//
//  inline double sigmoid(double x)
//  {
//    if (x < 0)
//    {
//      const double expx = std::exp(x);
//      return expx / (1 + expx);
//    }
//    return 1 / (1 + std::exp(-x));
//  }
//
//  virtual double predict(const std::vector<float>& features) override
//  {
//    double exponent =
//        d_coefficients[0];  //  assuming intercept on the first position
//    for (size_t i = 1; i < d_coefficients.size(); i++)
//    {
//      exponent += d_coefficients[i] * features[i - 1];
//    }
//    return sigmoid(exponent);
//  }
//
//  virtual size_t numberOfFeatures() const override
//  {
//    return d_coefficients.size() - 1;
//  }
//
// protected:
//  std::vector<double> d_coefficients;
//};
//
//class WeightsWrapper : public PredictorInterface
//{
// public:
//  WeightsWrapper(const std::string& modelFile, const Matrices* matrices)
//      : PredictorInterface(modelFile), d_matrices(matrices)
//  {
//  }
//  virtual double predict(const std::vector<float>& features) override;
//  virtual ~WeightsWrapper() {}
//
//  virtual size_t numberOfFeatures() const override;
//
// protected:
//  const Matrices* d_matrices;
//  int d_numIterations;
//};
//
//#ifdef CVC5_LIGHTGBM
class LightGBMWrapper : public PredictorInterface
{
 public:
  LightGBMWrapper(const std::string& modelFile);
  virtual double predict(const std::vector<float>& features) override;
  virtual ~LightGBMWrapper();

  virtual size_t numberOfFeatures() const override;

 protected:
  BoosterHandle d_handle;
  FastConfigHandle d_configHandle;
  int d_numIterations = -1;
  int d_numFeatures = -1;
  int d_predictType;
};
//#else
//class LightGBMWrapper : public PredictorInterface
//{
// public:
//  LightGBMWrapper(const std::string&)
//  {
//    AlwaysAssert(0) << "not compiled with LightGBM support\n";
//  }
//  virtual double predict(const std::vector<float>&) override
//  {
//    AlwaysAssert(0) << "not compiled with LightGBM support\n";
//    return 0;
//  }
//  virtual ~LightGBMWrapper() {}
//
//  virtual size_t numberOfFeatures() const override
//  {
//    AlwaysAssert(0) << "not compiled with LightGBM support\n";
//    return 0;
//  }
//};
//#endif  // CVC5_LIGHTGBM
}  // namespace cvc5

#endif
