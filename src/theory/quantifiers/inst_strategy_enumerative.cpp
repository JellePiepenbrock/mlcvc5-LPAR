/******************************************************************************
 * Top contributors (to current version):
 *   Andrew Reynolds, Mikolas Janota, Mathias Preiner
 *
 * This file is part of the cvc5 project.
 *
 * Copyright (c) 2009-2022 by the authors listed in the file AUTHORS
 * in the top-level source directory and their institutional affiliations.
 * All rights reserved.  See the file COPYING in the top-level source
 * directory for licensing information.
 * ****************************************************************************
 *
 * Implementation of an enumerative instantiation strategy.
 */

#include "theory/quantifiers/inst_strategy_enumerative.h"

#include "options/quantifiers_options.h"
#include "theory/quantifiers/instantiate.h"
#include "theory/quantifiers/featurize.h"
#include "theory/quantifiers/quantifier_logger.h"
#include "theory/quantifiers/quantifier_sampler.h"
#include "theory/quantifiers/relevant_domain.h"
#include "theory/quantifiers/term_database.h"
#include "theory/quantifiers/term_tuple_enumerator.h"
#include "theory/quantifiers/term_util.h"
#include "theory/quantifiers/pytorch_wrapper.h"

using namespace cvc5::internal::kind;
using namespace cvc5::context;

namespace cvc5::internal {
namespace theory {
namespace quantifiers {


InstStrategyEnum::InstStrategyEnum(Env& env,
                                   QuantifiersState& qs,
                                   QuantifiersInferenceManager& qim,
                                   QuantifiersRegistry& qr,
                                   TermRegistry& tr,
                                   RelevantDomain* rd)
    : QuantifiersModule(env, qs, qim, qr, tr), d_rd(rd), d_enumInstLimit(-1)
{
  QSelFeatureProperties::Config qsel_config;
  qsel_config.d_formulaBOW = options().quantifiers.qselFormulaContext;
  QuantifierLogger::s_logger->d_qselFeatureProperties.reset(
      new QSelFeatureProperties(qsel_config));
  QuantifierLogger::s_logger->d_qselFeatureProperties->initialize();

//  const Options& options = QuantifierLogger::s_logger->options();
  const bool randomize = options().quantifiers.mlRandomizeTermsBase;
  if (randomize) {
    QuantifierLogger::s_logger->d_term_rng.seed(std::time(0));

  }


  if (options().quantifiers.lightGBModelQuantifiersWasSetByUser)
  {
    d_learningContext.d_quantifierPredictor.reset(
        new LightGBMWrapper(options().quantifiers.lightGBModelQuantifiers));
  }
  else if (options().quantifiers.GNNModelQuantifiersWasSetByUser) {
    Trace("quant-sampler") << "Loading model from " << options().quantifiers.GNNModelQuantifiers << std::endl;
    std::cout << "Model" << std::endl;
    // std::string hardcoded_location = "/home/jelle/projects/learning_cvc5/build/gnn.pt";
    d_learningContext.d_graphquantifierPredictor.reset(new PyTorchWrapper(options().quantifiers.GNNModelQuantifiers));
    Trace("quant-sampler") << "Model successfully loaded from user-specified location" << std::endl;
  }

  else if (options().quantifiers.GNNModelTermsWasSetByUser) {
    Trace("quant-sampler") << "Loading term model from " << options().quantifiers.GNNModelTerms << std::endl;
    std::cout << "Model" << std::endl;
    d_learningContext.d_graphtermPredictor.reset(new PyTorchWrapper(options().quantifiers.GNNModelTerms));
    Trace("quant-sampler") << "Term Model successfully loaded from user-specified location" << std::endl;
    std::cout << "Term Model successfully loaded from user-specified location" << std::endl;
  }

  if (options().quantifiers.QuantifierRandomSeedWasSetByUser) {

    d_learningContext.d_mt.seed(options().quantifiers.QuantifierRandomSeed);

  }


  // add GNN option - probably easiest to have another object controlled by different options
  // like before

  if (d_learningContext.d_quantifierPredictor)
  {
    std::cout << "; Loaded quantifier ML with "
              << d_learningContext.d_quantifierPredictor->numberOfFeatures()
              << " features from: "
              << d_learningContext.d_quantifierPredictor->d_modelFile
              << std::endl;
  }

}
void InstStrategyEnum::presolve()
{
  d_enumInstLimit = options().quantifiers.enumInstLimit;
}
bool InstStrategyEnum::needsCheck(Theory::Effort e)
{
  if (d_enumInstLimit == 0)
  {
    return false;
  }
  if (options().quantifiers.enumInstInterleave)
  {
    // if interleaved, we run at the same time as E-matching
    if (d_qstate.getInstWhenNeedsCheck(e))
    {
      return true;
    }
  }
  if (options().quantifiers.enumInst)
  {
    if (e >= Theory::EFFORT_LAST_CALL)
    {
      return true;
    }
  }
  return false;
}

void InstStrategyEnum::reset_round(Theory::Effort e) {}

void InstStrategyEnum::check(Theory::Effort e, QEffort quant_e)
{
  ++(d_qim.getInstantiate()->d_statistics.d_enum_check_calls);
  if (options().quantifiers.qloggingActive || options().quantifiers.mlBoostActives)
  {
    QuantifierLogger::s_logger->updateActiveQuantifiers(
      d_qstate.getActiveFormulas());
  }

  bool doCheck = false;
  bool fullEffort = false;
  if (d_enumInstLimit != 0)
  {
    if (options().quantifiers.enumInstInterleave)
    {
      // we only add when interleaved with other strategies
      doCheck = quant_e == QEFFORT_STANDARD && d_qim.hasPendingLemma();
    }
    if (options().quantifiers.enumInst && !doCheck)
    {
      if (!d_qstate.getValuation().needCheck())
      {
        doCheck = quant_e == QEFFORT_LAST_CALL;
        fullEffort = true;
      }
    }
  }
  if (!doCheck)
  {
    return;
  }
  ++(d_qim.getInstantiate()->d_statistics.d_enum_check_done);
  Assert(!d_qstate.isInConflict());
  double clSet = 0;
  if (TraceIsOn("enum-engine"))
  {
    clSet = double(clock()) / double(CLOCKS_PER_SEC);
    Trace("enum-engine") << "---Full Saturation Round, effort = " << e << "---"
                         << std::endl;
  }
  unsigned rstart = options().quantifiers.enumInstRd ? 0 : 1;
  unsigned rend = fullEffort ? 1 : rstart;
  unsigned addedLemmas = 0;
  // First try in relevant domain of all quantified formulas, if no
  // instantiations exist, try arbitrary ground terms.
  // Notice that this stratification of effort levels makes it so that some
  // quantified formulas may not be instantiated (if they have no instances
  // at effort level r=0 but another quantified formula does). We prefer
  // this stratification since effort level r=1 may be highly expensive in the
  // case where we have a quantified formula with many entailed instances.
  FirstOrderModel* fm = d_treg.getModel();
  unsigned nquant = fm->getNumAssertedQuantifiers();
  std::map<Node, bool> alreadyProc;
  if (options().quantifiers.qlogging)
  {
    QuantifierLogger::s_logger->registerInstantiationRound(d_treg);
//    QuantifierLogger::s_logger->registerTermRegistry(d_treg);
  }
  for (unsigned r = rstart; r <= rend; r++)
  {
    if (d_rd || r > 0)
    {
      if (r == 0)
      {
        Trace("inst-alg") << "-> Relevant domain instantiate..." << std::endl;
        Trace("inst-alg-debug") << "Compute relevant domain..." << std::endl;
        d_rd->compute();
        Trace("inst-alg-debug") << "...finished" << std::endl;
      }
      else
      {
        Trace("inst-alg") << "-> Ground term instantiate..." << std::endl;
      }

      // >>>

      // set runPredictions also if graph predictor
      // old
//      const bool runPredictions = true;
//      const bool runSampler =
//          runPredictions || options().quantifiers.qselEpsilonWasSetByUser;
//
//
//      std::unique_ptr<IQuantifierSampler> sampler(
//          runSampler ? static_cast<IQuantifierSampler*>(new QuantifierSampler())
//                     : new TrivialQuantifierSampler());
//
      // new
      const bool runPredictions = true;
      const bool runSampler = true;

//      std::unique_ptr<QuantifierSampler> sampler((new QuantifierSampler));
//      std::unique_ptr<TrivialQuantifierSampler> sampler((new TrivialQuantifierSampler));
      std::unique_ptr<HardThresholdQuantifierSampler> sampler((new HardThresholdQuantifierSampler));
//      std::unique_ptr<MaxQuantifierChooser> sampler((new MaxQuantifierChooser));
      //
      // use the QListGraphList featurizer here.

      QSelFeaturizerBOW featurizer(
         QuantifierLogger::s_logger->d_qselFeatureProperties.get());

      d_learningContext.d_time_spent_process_list.clear();


      RLFeaturizerGraph rl_featurizer;
      RLFeaturizerGraph term_featurizer;
      bool term_rl_samples = true;

      if (options().quantifiers.qselFormulaContext)
      {
        featurizer.addContext(QuantifierLogger::s_logger->assertionsBOW());
      }
      //
      // prepare quantifiers to be run

      // for graph network, I need the whole list of qs at once; think about
      // how to support both ml predictors on 1 branch.

      // if statement outside of this for loop for gnn
      std::vector<Node> quantifier_list;
      std::vector<int> quantifier_todo_list;
      std::vector<int> normalized_quantifier_list;
      std::vector<TNode> as_list;
      int to_process_quants = 0;
      std::vector<Node> inst_list;
      std::vector<Node> term_list;
      std::map<TypeNode, std::vector<Node>> available_terms_per_type;

      std::vector<int> var_index_list;
      std::vector<Node> var_glob_list;
      std::vector<double> quantifier_scores;
      std::map<std::pair<int,int>, float> scores;
      std::vector<std::vector<int>> term_index_list;

//      if (d_learningContext.d_graphquantifierPredictor){
//
//        // process as a list all at once with gnn
//
//        // add qs to vector if they need to be processed
//
//        for (unsigned i = 0; i < nquant; i++) {
//          const Node q = fm->getAssertedQuantifier(i, true);
//          QuantifierLogger::s_logger->getQuantifierInfo(q);
//          bool doProcess = d_qreg.hasOwnership(q, this)
//                           && fm->isQuantifierActive(q)
//                           && alreadyProc.find(q) == alreadyProc.end();
//          if (doProcess) {
//            quantifier_list.push_back(q);
//          }
//        }
//
//        as_list = QuantifierLogger::s_logger->getAssertionList();
//        if (options().quantifiers.QuantifierRLSamples)
//        {
//          auto instantiation_list =
//              QuantifierLogger::s_logger->getInstantiations();
//          for (auto& inst_info : instantiation_list)
//          {
//            inst_list.push_back(inst_info.d_body);
//          }
//
//          std::set<TypeNode> type_set;
//          for (auto& quant : quantifier_list) {
//
//            auto num_vars = quant[0].getNumChildren();
//            for (int varindex = 0; varindex < static_cast<int>(num_vars); ++varindex) {
//              type_set.insert(quant[0][varindex].getType());
//            }
//          }
//
//          auto termdb = d_treg.getTermDatabase();
//          for (auto& type : type_set) {
//            const size_t ground_terms_count = termdb->getNumTypeGroundTerms(type);
//
//            for (size_t j = 0; j < ground_terms_count; j++) {
//              //            std::cout << " " << std::endl;
//              Node gt = termdb->getTypeGroundTerm(type, j);
//              term_list.push_back(gt);
//            }
//          }
//          //        QuantifierListFeaturizerGraph gr_featurizer(quantifier_list);
//          //        QuantifierPlusAssertionsListFeaturizerGraph gr_featurizer(as_list, quantifier_list);
//
//          rl_featurizer.init(as_list, quantifier_list, inst_list, term_list);
//          rl_featurizer.run();
//          const auto predictions =
//              d_learningContext.d_graphquantifierPredictor->predict(
//                  rl_featurizer.nodeInits(),
//                  rl_featurizer.graphEdges(),
//                  rl_featurizer.getNormalizedTargets());
//          Assert(predictions.size() == quantifier_list.size());
//
//          for (int i = 0; i < static_cast<int>(quantifier_list.size()); ++i) {
//
//            std::cout << predictions[i] << "--" << quantifier_list[i].getId() << std::endl;
//
//            sampler->addQuantifier(i, quantifier_list[i], predictions[i]);
//          }
//          //          std::cout << "---------------END---------------" << std::endl;
//        }
//
//
//        else {
//
//          if (options().quantifiers.QuantifierListUseAssertions) {
//            // for the term prediction, run a second predictor here and store the
//            // embedding in PyTorchWrapper variable?
//
//            QuantifierPlusAssertionsListFeaturizerGraph gr_featurizer(as_list, quantifier_list);
//            gr_featurizer.run();
//            const auto predictions =
//                d_learningContext.d_graphquantifierPredictor->predict(
//                    gr_featurizer.nodeInits(),
//                    gr_featurizer.graphEdges(),
//                    gr_featurizer.getNormalizedTargets());
//
//            Assert(predictions.size() == quantifier_list.size());
//
//            for (int i = 0; i < static_cast<int>(quantifier_list.size()); ++i) {
//
//              std::cout << predictions[i] << "--" << quantifier_list[i].getId() << std::endl;
//
//              sampler->addQuantifier(i, quantifier_list[i], predictions[i]);
//            }
//            //            std::cout << "---------------END---------------" << std::endl;
//          }
//          else{
//            QuantifierListFeaturizerGraph gr_featurizer(quantifier_list);
//            gr_featurizer.run();
//            const auto predictions =
//                d_learningContext.d_graphquantifierPredictor->predict(
//                    gr_featurizer.nodeInits(),
//                    gr_featurizer.graphEdges(),
//                    gr_featurizer.getNormalizedTargets());
//
//            Assert(predictions.size() == quantifier_list.size());
//
//            for (int i = 0; i < static_cast<int>(quantifier_list.size()); ++i) {
//
//              std::cout << predictions[i] << "--" << quantifier_list[i].getId() << std::endl;
//
//              sampler->addQuantifier(i, quantifier_list[i], predictions[i]);
//            }
//            std::cout << "---------------END---------------" << std::endl;
//          }
//        }
//
//      }

      // NO graphquantifierpredictor
      // process one at a time

      // collecting RL style samples when we are running the base solver
      if (term_rl_samples)
      {
        as_list = QuantifierLogger::s_logger->getAssertionList();
        auto instantiation_list =
            QuantifierLogger::s_logger->getInstantiations();
        for (auto& inst_info : instantiation_list)
        {
          inst_list.push_back(inst_info.d_body);
        }
      }

      std::vector<unsigned> qs_indices;
      for (unsigned i = 0; i < nquant; i++)
      {
        Node q = fm->getAssertedQuantifier(i, true);
        QuantifierLogger::s_logger->getQuantifierInfo(q);
        bool doProcess = d_qreg.hasOwnership(q, this)
                         && fm->isQuantifierActive(q)
                         && alreadyProc.find(q) == alreadyProc.end();
        if (doProcess)
        {
          quantifier_list.push_back(q);
          qs_indices.push_back(i);
          quantifier_todo_list.push_back(1);
        }
        else {
          quantifier_todo_list.push_back(0);
        }
      }

      bool produce_graph_term_predictor_data;
      bool run_predictor;
      bool interleave = false;
      bool print_rl_samples = false;


      if (interleave)
      {
        //        if ((d_learningContext.d_round_counter % 2) == 1)
        if ((d_learningContext.d_round_counter > 4))
        {
          produce_graph_term_predictor_data = false;
          run_predictor = false;
        }
        else
        {
          produce_graph_term_predictor_data = true;
          run_predictor = true;
        }
      }
      else {
        produce_graph_term_predictor_data = true;
        run_predictor = true;
      }
      //

      print_rl_samples = false;
      produce_graph_term_predictor_data = true;
      run_predictor = true;

      if (produce_graph_term_predictor_data) {
        std::chrono::time_point<std::chrono::high_resolution_clock> start_dfs = std::chrono::high_resolution_clock::now();
        // Get the graph representation and get the embeddings;
        as_list = QuantifierLogger::s_logger->getAssertionList();
        auto instantiation_list =
            QuantifierLogger::s_logger->getInstantiations();
        for (auto& inst_info : instantiation_list)
        {
          inst_list.push_back(inst_info.d_body);
        }



        // to get all the terms of a certain type, first need to know the types
        // of the variables;
        std::set<TypeNode> type_set;
        for (auto& quant : quantifier_list) {
          //          std::cout << "QB: " << quant.getId() << std::endl;
          auto num_vars = quant[0].getNumChildren();
          for (int varindex = 0; varindex < static_cast<int>(num_vars); ++varindex) {
            type_set.insert(quant[0][varindex].getType());
            //            d_learningContext.d_persistent_type_set.insert(quant[0][varindex].getType());

            var_index_list.push_back(quant[0][varindex].getId());
            var_glob_list.push_back(quant[0][varindex]);
          }

        }
        //        std::vector<Node> term_list;

        auto termdb = d_treg.getTermDatabase();
        if (options().quantifiers.qlogging)
        {
          //          QuantifierLogger::s_logger->registerInstantiationRound(d_treg);
          //              QuantifierLogger::s_logger->registerTermRegistry(d_treg);
          //              QuantifierLogger::s_logger->d_term_registry_store.reset(new TermRegistry(d_treg));
          QuantifierLogger::s_logger->d_term_database_store = termdb;
        }


        // This has some repetition; this goes to sample
        bool just_keep_representatives = true;
        std::map<Node, Node> repsFound;
        for (auto& type : type_set) {

          const size_t ground_terms_count = termdb->getNumTypeGroundTerms(type);
          std::vector<Node> type_terms;
          for (size_t j = 0; j < ground_terms_count; j++) {
            //            std::cout << " " << std::endl;
            Node gt = termdb->getTypeGroundTerm(type, j);
            term_list.push_back(gt);
            d_learningContext.d_persistent_term_set.insert(gt);

            if (! just_keep_representatives) {
              type_terms.push_back(gt);
            }
            else {
              if (!quantifiers::TermUtil::hasInstConstAttr(gt))
              {
                Node rep = d_qstate.getRepresentative(gt);

                // d_qs should probably be d_qstate here
                if (repsFound.find(rep) == repsFound.end())
                {
                  repsFound[rep] = gt;
                  type_terms.push_back(gt);
                }
              }

            }





          }
          available_terms_per_type[type] = type_terms;
        }

        // How to deal with representatives? Could shave off some computation..
        // Perhaps I could mark the representative?
        // Re repetition: this goes to network here.
        bool add_non_reps = false;

        int termcounter = 0;
        for (auto& var : var_glob_list)
        {

          int nonrepincludetermcounter = 0;
          int reptermcounter = 0;

          std::vector<int> current_var_terms;
          auto type_node = var.getType();
          const size_t ground_terms_count =
              termdb->getNumTypeGroundTerms(type_node);

          std::map<Node, Node> repsFound;

          for (size_t j = 0; j < ground_terms_count; j++)
          {
            //            std::cout << " " << std::endl;

            if (add_non_reps)
            {
              Node gt = termdb->getTypeGroundTerm(type_node, j);

              current_var_terms.push_back(gt.getId());
              termcounter += 1;
            }
            else {
              Node gt = termdb->getTypeGroundTerm(type_node, j);
              nonrepincludetermcounter += 1;
              if (!quantifiers::TermUtil::hasInstConstAttr(gt))
              {
                Node rep = d_qstate.getRepresentative(gt);

                // d_qs should probably be d_qstate here
                if (repsFound.find(rep) == repsFound.end())
                {
                  repsFound[rep] = gt;
                  current_var_terms.push_back(gt.getId());
                  termcounter += 1;
                  reptermcounter += 1;
                }
              }
            }
          }
          std::cout << "VAR has " << reptermcounter << " repterms and " << nonrepincludetermcounter << " otherwise" << std::endl;
          term_index_list.push_back(current_var_terms);
        }


        std::cout << "Number of ground terms available in total over all variables: " << term_list.size() << std::endl;
        //        for (auto& tt: term_list) {
        //          std::cout << tt << std::endl;
        //        }
        //        std::vector<Node> term_list(d_learningContext.d_persistent_term_set.begin(), d_learningContext.d_persistent_term_set.end());
        term_featurizer.init(as_list, quantifier_list, inst_list, term_list);
        term_featurizer.run();
        // store the node mapping for later use.

        // TODO
        d_learningContext.d_node_cvc_graph_index.reset(new std::unordered_map<int,int>(term_featurizer.getNodeNormalizationMap()));
        std::cout << "quantifiers list size " << quantifier_list.size() << std::endl;

        std::cout << "mapsizes: learncont" << d_learningContext.d_node_cvc_graph_index->size() << std::endl;

        std::unordered_map<int,int> reversemap;
        //        std::cout << "REVERSEMAP CONSTRUCTION" << std::endl;
        for (auto [key, value] : term_featurizer.getNodeNormalizationMap()) {
          //          std::cout << "pair [" << key << "->" << value << "]" << " to " << "[" << value << "->" << key << "]"<< std::endl;
          reversemap[value] = key;
        }
        d_learningContext.d_node_cvc_graph_index_reverse.reset(new std::unordered_map<int,int>(reversemap));

        std::chrono::time_point<std::chrono::high_resolution_clock> end_dfs = std::chrono::high_resolution_clock::now();

        std::cout << "cost of dfs: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_dfs - start_dfs).count() << std::endl;
        //
        for (auto& qnode : quantifier_list) {
          normalized_quantifier_list.push_back( d_learningContext.d_node_cvc_graph_index->at(qnode.getId()));
        }
        // actually have a predictor?
        if (d_learningContext.d_graphtermPredictor && run_predictor)
        {
          std::chrono::time_point<std::chrono::high_resolution_clock> start_indexconst = std::chrono::high_resolution_clock::now();

          std::vector<int> var_index_list_normalized;
          std::vector<std::vector<int>> term_index_list_normalized;

          for (auto& varid : var_index_list)
          {
            //            std::cout << varid << " Normalized to "
            //                      << d_learningContext.d_node_cvc_graph_index->at(varid)
            //                      << std::endl;
            var_index_list_normalized.push_back(
                d_learningContext.d_node_cvc_graph_index->at(varid));
          }
          //          std::cout << "terms" << std::endl;
          int vc = 0;
          for (auto& vartermlist : term_index_list)
          {
            //            std::cout << "var" << var_index_list[vc] << std::endl;
            vc += 1;
            std::vector<int> normalized_vartermlist;
            //            normalized_vartermlist.reserve(vartermlist.size()); // doesn't help that much.
            for (auto& termid : vartermlist)
            {
              //              std::cout << termid << std::endl;
              normalized_vartermlist.push_back(
                  d_learningContext.d_node_cvc_graph_index->at(termid));
            }
            term_index_list_normalized.push_back(normalized_vartermlist);
          }


          //        var_index_list;
          //        term_index_list;
          //          std::cout << "Computing Embeddings: " << std::endl;
          // TODO also precompute the var vs term matches here.
          // TODO can probably store them in a map<std::pair<varid, termid>, score>
          // TODO which I can then access in the process functions
          //        d_learningContext.d_graphtermPredictor->compute_embeddings(
          //                term_featurizer.nodeInits(),
          //                term_featurizer.graphEdges());

          std::chrono::time_point<std::chrono::high_resolution_clock> end_indexconst = std::chrono::high_resolution_clock::now();

          std::cout << "cost of indexconst: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_indexconst - start_indexconst).count() << std::endl;
          //
          std::chrono::time_point<std::chrono::high_resolution_clock> start_predictor = std::chrono::high_resolution_clock::now();

          std::tie(scores, quantifier_scores) = d_learningContext.d_graphtermPredictor->compute_scores_and_premsel(
              term_featurizer.nodeInits(),
              term_featurizer.graphEdges(),
              normalized_quantifier_list,
              var_index_list_normalized,
              term_index_list_normalized);


          std::cout << "QUANTIFIER SCORES " << quantifier_scores.size() << std::endl;
          for (auto& qs : quantifier_scores) {
            std::cout << qs << " ";
          }
//          sampler->d_probabilities = quantifier_scores;
          std::cout << std::endl;
          std::chrono::time_point<std::chrono::high_resolution_clock> end_predictor = std::chrono::high_resolution_clock::now();

          std::cout << "cost of predictor: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_predictor - start_predictor).count() << std::endl;

          std::chrono::time_point<std::chrono::high_resolution_clock> start_norm= std::chrono::high_resolution_clock::now();


          std::map<std::pair<int, int>, float> normalized_score_map;
          for (auto& [pairkeys, score] : scores)
          {
            int cvc5_varid =
                d_learningContext.d_node_cvc_graph_index_reverse->at(
                    pairkeys.first);
            int cvc5_termid =
                d_learningContext.d_node_cvc_graph_index_reverse->at(
                    pairkeys.second);
            //            std::cout << cvc5_varid << "-" << cvc5_termid << "SC: " << score
            //                      << std::endl;
            auto newkeys = std::make_pair(cvc5_varid, cvc5_termid);
            auto sc = score;
            normalized_score_map.insert(
                std::map<std::pair<int, int>, float>::value_type(newkeys, sc));
          }

          //          std::cout << "sizes: " << normalized_score_map.size() << " "
          //                    << scores.size() << " " << termcounter << std::endl;
          d_learningContext.d_varterm_score_lookup.reset(
              new std::map<std::pair<int, int>, float>(normalized_score_map));
          //        d_term_registry_store = &
          //
          std::chrono::time_point<std::chrono::high_resolution_clock> end_norm = std::chrono::high_resolution_clock::now();

          std::cout << "cost of normalizing map: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_norm - start_norm).count() << std::endl;

        }

      }

      for (unsigned i = 0; i < nquant; i++)
      {
        Node q = quantifier_list[i];
        {
          to_process_quants += 1;

          auto prediction = quantifier_scores[i];
          sampler->addQuantifier(i, q, prediction);





        }
        //

      }
//      std::cout << quantifier_list.size() << " " << quantifier_scores.size() << " " << qs_indices.size() << std::endl;
//      int num_qs_to_process = static_cast<int>(quantifier_list.size());
//      for (int i=0; i < num_qs_to_process; ++i){
//        auto prediction = quantifier_scores[i];
//        auto q = quantifier_list[i];
//        sampler->addQuantifier(qs_indices[i], q, prediction);
//
//      }


      sampler->close();
      Trace("quant-sampler") << "[sampling round]  from: "
                             << sampler->get_quantifiers().size()
                             << " quantifiers" << std::endl;
      std::vector<bool> tried(sampler->get_quantifiers().size(), false);
      std::vector<bool> successful(sampler->get_quantifiers().size(), false);
      std::vector<int> actually_used(sampler->get_quantifiers().size(), 0);

      std::cout << "QS to process: " << to_process_quants << std::endl;
      std::cout << "QS in sampler: " << successful.size() << std::endl;
      size_t totry = sampler->get_quantifiers().size();


//      bool graph_term_predictor = true;
      std::chrono::time_point<std::chrono::high_resolution_clock> start_dataprep = std::chrono::high_resolution_clock::now();

      // TODO migrate all these options to command line options


      std::chrono::time_point<std::chrono::high_resolution_clock> end_dataprep = std::chrono::high_resolution_clock::now();

      std::cout << "cost of prep: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_dataprep - start_dataprep).count() << std::endl;
      // Check if term that are created in the process function are eventually
      // incorporated into the graph
//      if (d_learningContext.d_round_counter > 0) {
//        for (auto& variable_termset: d_learningContext.d_orphan_terms) {
//          for (auto& term: d_learningContext.d_orphan_terms.at(variable_termset.first)){
//            std::cout << "Check if term: " << term << " in " << std::endl;
//            d_learningContext.d_node_cvc_graph_index->at(term);
//          }
//        }
//        d_learningContext.d_orphan_terms.clear();
//      }

//      std::vector<bool> used_quantifiers(sampler->get_quantifiers().size(), false);

      // sample from existing quantifiers
      // if no lemma is added, we continue with sampling because otherwise we
      // get "unknown" results bcs the solver thinks nothing left to do
      std::chrono::time_point<std::chrono::high_resolution_clock> start_sampler = std::chrono::high_resolution_clock::now();

//      std::cout << "cost of prep: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_dataprep - start_dataprep).count() << std::endl;

//      std::vector<int> actual_qs_used;
      int current_q_index = 0;

      do
      {
        Trace("quant-sampler") << (runSampler ? "[sampling round]\n" : "");


        for (const auto& qi : sampler->get_quantifiers())

        // quantifiers should be in same order as they are above, when they go into the gnn
        // therefore,
        {
          if (tried[qi.d_localIndex]
              || !sampler->choose(d_learningContext.d_mt, qi))
          {
            continue;
          }
          Assert(qi.d_localIndex < tried.size());
          Assert(totry > 0);
          tried[qi.d_localIndex] = true;
          totry--;

          // Process function is entrypoint to term generation

          if (process(qi.d_quantifier, fullEffort, r == 0))
          {
            successful[qi.d_localIndex] = true;
            actually_used[qi.d_localIndex] = 1;
            // don't need to mark this if we are not stratifying
            if (!options().quantifiers.enumInstStratify)
            {
              alreadyProc[qi.d_quantifier] = true;
            }
            // added lemma
            addedLemmas++;
          }
          if (d_qstate.isInConflict())
          {
            totry = 0;
            break;
          }

        }
      } while (options().quantifiers.qselFallback && totry > 0 && addedLemmas == 0);

      std::chrono::time_point<std::chrono::high_resolution_clock> end_sampler= std::chrono::high_resolution_clock::now();
      //
      std::cout << "cost of sampler: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_sampler - start_sampler).count() << std::endl;

      // writing down each sample with which qs were actually tried.
//      std::vector<int> used_qs(successful.begin(), successful.end());


      // TODO Modify this to report the actual instantiations
      if (term_rl_samples) {

        if (print_rl_samples)
        {


//          std::chrono::time_point<std::chrono::high_resolution_clock>
//              begin_copy = std::chrono::high_resolution_clock::now();
//          //
//          //
//          //        QuantifierLogger::s_logger.d_d_round_instantiation_store
//
//          // Store stuff for the end instead of printing to disk in the middle of the process
//          QuantifierLogger::s_logger->d_featurizers_store_vector.push_back(term_featurizer);
//          QuantifierLogger::s_logger->d_infos_store_vector.push_back(QuantifierLogger::s_logger->d_infos);
//          QuantifierLogger::s_logger->d_round_instantiation_store_vector.push_back(QuantifierLogger::s_logger->d_round_instantiation_store);
//
//
//          std::chrono::time_point<std::chrono::high_resolution_clock>
//              end_copy = std::chrono::high_resolution_clock::now();

//          std::cout << "cost of copy: "
//                    << std::chrono::duration_cast<std::chrono::milliseconds>(
//                           end_copy - begin_copy)
//                           .count()
//                    << std::endl;

//          bool premsel_data = false;
//          if (premsel_data)
//          {
//            std::vector<int> label_vector;
//
//            for (auto& lab : succesful)
//            {
//              if (lab)
//              {
//                label_vector.push_back(1)
//              }
//              else
//              {
//                label_vector.push_back(0)
//              }
//            }
//          }
          std::chrono::time_point<std::chrono::high_resolution_clock>
              begin_printing = std::chrono::high_resolution_clock::now();

          std::cout << "normquantlistsize: " << normalized_quantifier_list.size() << std::endl;
          QuantifierLogger::s_logger->printTermSamplesRL(std::cout,
                                                         term_featurizer, available_terms_per_type, normalized_quantifier_list, actually_used);


          std::chrono::time_point<std::chrono::high_resolution_clock>
              end_printing = std::chrono::high_resolution_clock::now();
          std::cout << "cost of printing: "
                    << std::chrono::duration_cast<std::chrono::milliseconds>(
                           end_printing - begin_printing)
                           .count()
                    << std::endl;
          // clear map after printing
          QuantifierLogger::s_logger->d_round_instantiation_store.clear();
        }
      }

      /*
      for (unsigned i = 0; i < nquant; i++)
      {
        Node q = fm->getAssertedQuantifier(i, true);
        bool doProcess = d_qreg.hasOwnership(q, this)
                         && fm->isQuantifierActive(q)
                         && alreadyProc.find(q) == alreadyProc.end();
        if (doProcess)
        {
          if (process(q, fullEffort, r == 0))
          {
            // don't need to mark this if we are not stratifying
            if (!options().quantifiers.enumInstStratify)
            {
              alreadyProc[q] = true;
            }
            // added lemma
            addedLemmas++;
          }
          if (d_qstate.isInConflict())
          {
            break;
          }
        }
      }
      */
      // <<<
      if (d_qstate.isInConflict()
          || (addedLemmas > 0 && options().quantifiers.enumInstStratify))
      {
        // we break if we are in conflict, or if we added any lemma at this
        // effort level and we stratify effort levels.
        break;
      }
    }
    if (! (r == 0)){ // count the non-rd rounds
      // TODO use this variable to interleave
      d_learningContext.d_round_counter += 1;

    }
    if ( d_learningContext.d_time_spent_process_list.size() > 0) {
      double mean_duration_process = 0.0;
      double sum_duration_process = 0;
      for (auto& entry : d_learningContext.d_time_spent_process_list){
        sum_duration_process += entry;
      }
      mean_duration_process = sum_duration_process /  static_cast<double>(d_learningContext.d_time_spent_process_list.size());
      std::cout << "Average duration process: " << mean_duration_process << std::endl;
      std::cout << "Sum duration process: " << sum_duration_process << std::endl;
    }


  }
  if (TraceIsOn("enum-engine"))
  {
    Trace("enum-engine") << "Added lemmas = " << addedLemmas << std::endl;
    double clSet2 = double(clock()) / double(CLOCKS_PER_SEC);
    Trace("enum-engine") << "Finished full saturation engine, time = "
                         << (clSet2 - clSet) << std::endl;
  }
  if (d_enumInstLimit > 0)
  {
    d_enumInstLimit--;
  }

}

bool InstStrategyEnum::process(Node quantifier, bool fullEffort, bool isRd)
{
  // ignore if constant true (rare case of non-standard quantifier whose body
  // is rewritten to true)
  if (quantifier[1].isConst() && quantifier[1].getConst<bool>())
  {
    return false;
  }
  std::chrono::time_point<std::chrono::high_resolution_clock> start_process = std::chrono::high_resolution_clock::now();
  // General design questions: how to integrate with GNN here.
  // Instantiation is done per quantifier.
//  std::cout << "NEW QUANTIFIER BEING PROCESSED" << std::endl;
  // quantifier inference manager gives us a pointer to the Instatiate modle
  Instantiate* ie = d_qim.getInstantiate();
  ++(ie->d_statistics.d_enum_process_calls);
  TermTupleEnumeratorEnv ttec;
  ttec.d_fullEffort = fullEffort;
  ttec.d_increaseSum = options().quantifiers.enumInstSum;
  ttec.d_tr = &d_treg;
  // make the enumerator, which is either relevant domain or term database
  // based on the flag isRd.
//  bool mlterms = true;

  std::unique_ptr<TermTupleEnumeratorInterface> enumerator;
  NodeVector completedTerms;
  bool interleave = false;
  bool use_ml_enum = true;
  if (interleave) {
//    if ((d_learningContext.d_round_counter % 2) == 1)
    if ((d_learningContext.d_round_counter > 4))
    {
        use_ml_enum = false;
      }
    else {
      use_ml_enum = true;
    }
  }
  if (!d_learningContext.d_graphtermPredictor || !use_ml_enum) {
    std::cout << "NO ML ENUMERATOR" << std::endl;
    enumerator.reset(
        isRd ? mkTermTupleEnumeratorRd(quantifier, &ttec, d_rd)
             : mkTermTupleEnumerator(quantifier, &ttec, d_qstate));

  }
  else {
    std::cout << "ML ENUMERATOR" << std::endl;
    enumerator.reset(
        isRd ? mkTermTupleEnumeratorRd(quantifier, &ttec, d_rd)
             : mkTermTupleEnumeratorML(quantifier, &ttec, d_qstate, &d_learningContext));

  }


  // I suppose I could replace terms directly; but it could be better to
  // go a bit later so some legal checks are done. Don't want to suggest terms
  // that don't do anything.

  std::vector<Node> terms;
  std::vector<bool> failMask;

  int tuple_counter = 0;

//  int max_insts = 5;
//  int inst_counter = 0;

  for (enumerator->init(); enumerator->hasNext();)
  {
    if (d_qstate.isInConflict())
    {
      // could be conflicting for an internal reason
      return false;
    }

    // TODO: Figure out whether a breaker is useful:
//    if (tuple_counter > 0) {
//      return false;
//    }
    ++(ie->d_statistics.d_enum_process_attempts);
    enumerator->next(terms);

    if (options().quantifiers.qlogging)
    {
       // complete missing terms, TODO: elsewhere?
       std::vector<Node> tmp(terms.size());
       for (size_t vx = terms.size(); vx--;)
       {
          tmp[vx] = terms[vx].isNull()
                  ? d_treg.getTermForType(quantifier[0][vx].getType())
                  : terms[vx];
       }
       // log instantiation attempt
       completedTerms = QuantifierLogger::s_logger->registerInstantiationAttempt(
          quantifier, tmp);
    }
    // try instantiation
    failMask.clear();
    /* if (ie->addInstantiation(quantifier, terms)) */
    const bool successful = ie->addInstantiationExpFail(
            quantifier, terms, failMask, InferenceId::QUANTIFIERS_INST_ENUM);
    if (options().quantifiers.qlogging)
    {
//       QuantifierLogger::s_logger->registerInstantiation(
//          quantifier, successful, completedTerms);
       if (successful) {
         QuantifierLogger::s_logger->registerInstantiationRL(
             quantifier, completedTerms
         );
//         std::cout << "RL registration" << std::endl;
       }


    }
    if (successful)
    {
      Trace("inst-alg-rd") << "Success!" << std::endl;
//      std::cout << "succesful" << std::endl;
      ++(ie->d_statistics.d_enum_process_success);
//      std::cout << "QSUCCESS: " << quantifier.getId() << std::endl;
      // TODO Log the termvectors here
//      std::cout << "Q " << quantifier.getId() << " failed " << tuple_counter << " times before success." << std::endl;
//      std::chrono::time_point<std::chrono::high_resolution_clock> end_process = std::chrono::high_resolution_clock::now();
//      std::cout << "cost of this process: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_process - start_process).count() << std::endl;
      return true;
    }
    else
    {
      enumerator->failureReason(failMask);
      tuple_counter += 1;
//      std::cout << "QFAILURE: " << quantifier.getId() << std::endl;
    }
  }
//  std::cout << "Q " << quantifier.getId() << " failed " << tuple_counter << " times before failure." << std::endl;
  std::chrono::time_point<std::chrono::high_resolution_clock> end_process = std::chrono::high_resolution_clock::now();
//  std::cout << "cost of this process: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_process - start_process).count() << std::endl;
  // convert to doubles
//  std::chrono::duration<int, std::micro> startpoint_dur = start_process;
//  std::chrono::duration<int, std::micro> endpoint_dur = end_process;
//  int startpoint = startpoint_dur.count();
//  int endpoint = endpoint_dur.count();
  d_learningContext.d_time_spent_process_list.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(end_process - start_process).count());
//  int micros = dur.count();
  return false;
  // TODO : term enumerator instantiation?
}

}  // namespace quantifiers
}  // namespace theory
}  // namespace cvc5::internal
