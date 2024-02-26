#include "theory/quantifiers/pytorch_wrapper.h"
#include <bits/stdint-intn.h>
#include <torch/torch.h>
#include <torchscatter/scatter.h>

#include <iostream>

struct GraphNet : torch::nn::Module {

  bool initted = false;
  int netsize = 64;
  //  int netsize = 128;
  torch::nn::Linear graph_conv;
  torch::nn::Linear predictor;
  torch::nn::Linear predictor2;
  torch::nn::Linear predictor3;
  torch::nn::ParameterListImpl nb;
  torch::nn::ParameterListImpl eb;
  std::vector<std::shared_ptr<torch::nn::LinearImpl>> conv_vector;
  //torch::nn::ParameterList node_inits;

  //torch::nn::Parameter node_embeddings;
  //Initialize the node type embeddings
  GraphNet() : graph_conv(netsize, netsize), predictor(netsize, netsize), predictor2(netsize, netsize), predictor3(netsize, 1)

  {
    register_module("graph_conv", graph_conv);
    register_module("predictor", predictor);
    register_module("predictor2", predictor2);
    register_module("predictor3", predictor3);

    //  //    std::string param_name = "NodeType";
    //  //    param_name += std::to_string(i);
    //  //    std::cout << param_name << std::endl;
    //  //    register_parameter(param_name, torch::randn(16), true);
    //  //}

    for (int j=0; j < 10; ++j) {

      conv_vector.push_back(register_module("layer_" + std::to_string(j), torch::nn::Linear(2*netsize, netsize)));

    }
    nb.append(register_parameter("nb_0", torch::randn({400, netsize}), true));
    eb.append(register_parameter("eb_0", torch::randn({20, netsize}), true));

  }

};


// ORACLE
//std::vector<std::string> split_string(
//    std::string str,
//    std::string delim)
//{
//  std::vector<std::string> splittedStrings = {};
//  size_t pos = 0;
//
//  while ((pos = str.find(delim)) != std::string::npos)
//  {
//    std::string token = str.substr(0, pos);
//    if (token.length() > 0)
//      splittedStrings.push_back(token);
//    str.erase(0, pos + delim.length());
//  }
//
//  if (str.length() > 0)
//    splittedStrings.push_back(str);
//  return splittedStrings;
//}
//
//std::vector<std::string> split_string_by_newline(const std::string& str)
//{
//  auto result = std::vector<std::string>{};
//  auto ss = std::stringstream{str};
//
//  for (std::string line; std::getline(ss, line, '\n');)
//    result.push_back(line);
//
//  return result;
//}
//
//std::vector<std::vector<int>> extract_int_vector_paired(std::string& int_line) {
//
//  std::vector<std::vector<int>> created_vector;
//  std::vector<std::string> splitted;
//
//  std::string del = " ";
//
//  splitted = SplitString(int_line, del);
//
//  std::vector<int> temp_vector;
//  int counter = 0;
//  for (auto& integer : splitted) {
//    //        std::cout << integer << std::endl;
//
//
//    temp_vector.push_back(stoi(integer));
//    counter += 1
//    if ((counter % 2 == 0) and (counter != 0)) {
//      created_vector.push_back(temp_vector);
//      temp_vector.clear();
//    }
//  }
//  return created_vector;
//}
//
//struct Oracle {
//
//  std::vector<std::map<int, int>> history;
//
//  Oracle(std::string location) {
//
//    // file that is pairs of V T each line.
//    //
//
//    std::ifstream tin(location);
//    std::string proof_trace((std::istreambuf_iterator<char>(tin)),
//                                  std::istreambuf_iterator<char>())
//    std::vector<std::string> linelist;
//
//    linelist = split_string_by_newline(proof_trace);
//
//
//    for (auto&round : linelist) {
//
//      std::map<int, int> round_map;
//      std::vector<std::vector<int>> round_vector = extract_int_vector_paired(round);
//      for (auto& [v, t] : round_vector) {
//        round_map[v] = t;
//      }
//      history.push_back(round_map);
//    }
//  }
//
//
//};
// ENDORACLE

//
//std::vector<double> PyTorchWrapper::predict(const std::vector<int>& node_kinds, const std::vector<std::vector<int>>& edges, const std::vector<int>& targets) {
//  torch::DeviceType device_type;
//  device_type = torch::kCUDA;
//  torch::Device device(device_type);
//  //std::cout << "Try to cast" << std::endl;
//  GraphNet* gnn_t = static_cast<GraphNet*> (this->gnn);
//  //std::cout << "Cast successful " << std::endl;
//
//  //std::cout << "Making some tensors" << std::endl;
//  torch::Tensor index_tensor = torch::tensor(node_kinds, device);
//  torch::Tensor quantifier_index_tensor = torch::tensor(targets, device);
//  torch::Tensor node_embs;
//  //std::cout << "Tensors created" << std::endl;
//
//  node_embs = gnn_t->nb[0].index_select(0, index_tensor);
//  //std::cout << "Indexing into embds" << std::endl;
//  torch::Tensor source_nodes;
//
//  std::vector<int> source_vector;
//  std::vector<int> target_vector;
//
//  for (auto& e : edges) {
//    source_vector.push_back(e[0]);
//    target_vector.push_back(e[1]);
//
//  }
//
//  torch::Tensor source_index_tensor = torch::tensor(source_vector, device);
//  torch::Tensor target_index_tensor = torch::tensor(target_vector, device);
//  int message_passes = 10;
//
//  for (int l=0; l < message_passes; ++l) {
//
//    source_nodes = node_embs.index_select(0, source_index_tensor);
//    int64_t dim = 0;
//    torch::optional<torch::Tensor> optional_out;
//    torch::optional<int64_t> dim_size;
//
//    torch::Tensor results;
//    results = gnn_t->graph_conv->forward(scatter_sum(source_nodes, target_index_tensor, dim, optional_out, dim_size));
//
//    node_embs = node_embs + torch::relu(results);
//
//    //node_embs = gnn_t->graph_conv->forward(node_embs);
//
//    node_embs = torch::nn::functional::layer_norm(node_embs, torch::nn::functional::LayerNormFuncOptions({64}).eps(2e-5));
//
//
//    //node_embs = torch::relu(node_embs);
//
//  }
//
//
//  torch::Tensor target_outputs;
//
//  target_outputs = node_embs.index_select(0, quantifier_index_tensor);
//
//  std::vector<double> results_vector;
//  torch::Tensor results;
//
//  results = torch::sigmoid(gnn_t->predictor2->forward(torch::relu(gnn_t->predictor->forward(target_outputs))));
//  for (int i = 0; i < torch::_shape_as_tensor(results)[0].item<int>(); ++i) {
//
//    results_vector.push_back(results[i].item<double>());
//  }
//
//  return results_vector;
//  //return torch::sigmoid(gnn_t->predictor2->forward(torch::relu(gnn_t->predictor->forward(torch::sum(node_embs, 0))))).item<double>();
//}
//
//std::vector<double> PyTorchWrapper::rank_terms(const std::vector<int>& variables, const std::vector<int>& terms){
//
//  // for now, assume there is only 1 variable vector
//
//  // shapes (1, 64)
//  // shapes (#t, 64)
//  // this function needs to take the node indices for a variable and several terms.
//  std::cout << "In the rank terms function" << std::endl;
////  std::cout << variables << std::endl;
////  std::cout << terms << std::endl;
//  // let's store the precomputed node embeddings somewhere;
//  torch::DeviceType device_type;
//  device_type = torch::kCUDA;
//  torch::Device device(device_type);
//
//  GraphNet* gnn_t = static_cast<GraphNet*> (this->gnn);
//
//  torch::Tensor* precomputed_embeddings = static_cast<torch::Tensor*> (this->computed_embeddings);
//
//  torch::Tensor variable_index_tensor = torch::tensor(variables, device);
//  torch::Tensor term_index_tensor = torch::tensor(terms, device);
//  std::cout << variable_index_tensor << std::endl;
//  std::cout << term_index_tensor << std::endl;
//  // now i have tensors to index with
//
//  torch::Tensor variable_embeddings;
//  torch::Tensor term_embeddings;
//
//  variable_embeddings = gnn_t->predictor->forward(precomputed_embeddings->index_select(0, variable_index_tensor));
//
//  term_embeddings = gnn_t->predictor2->forward(precomputed_embeddings->index_select(0, term_index_tensor));
//
//  torch::Tensor variable_stack;
//
//  variable_stack = variable_embeddings.repeat({static_cast<int>(terms.size()), 1});
//
//  torch::Tensor results;
//
//  std::vector<torch::Tensor> tensorvector;
//
//  tensorvector.push_back(variable_stack);
//  tensorvector.push_back(term_embeddings);
//  std::cout << variable_stack.sizes() << std::endl;
//  std::cout << term_embeddings.sizes() << std::endl;
//  torch::TensorList tenslist = torch::TensorList(tensorvector);
////  einsum_inputs =
//  //https://discuss.pytorch.org/t/libtorch-how-to-convert-std-vector-torch-tensor-to-torch-tensorlist/106828/2
//  results = torch::einsum("ij,ij->i",tenslist);
//
//  // result is #terms length of floats
//
//  std::cout << results << std::endl;
//
//  // this should be refactored, slow;
//  std::vector<double> results_vector;
//
//  for (int i = 0; i < torch::_shape_as_tensor(results)[0].item<int>(); ++i) {
//
//    results_vector.push_back(results[i].item<double>());
//  }
//  return results_vector;
//}

//std::vector<float> sample(std::vector<int> ages) {
//
//    torch::Tensor ages_tensor = torch::tensor(ages)
//
//    int universe_flattening_constant = -0.1;
//
//    ages_tensor = ages_tensor * universe_flattening_constant;
//
//    // noise scale
//    float beta = 1.0;
//    auto random_noise = torch::rand_like(ages_tensor);
//    noise = beta * - torch::log(- torch::log(random_noise));
//
//    torch::Tensor noisy_outcome = ages_tensor + noise
//    std::vector<float> noisy_outcome_vector(noisy_outcome.data_ptr<float>(), noisy_outcome.data_ptr<float>() + noisy_outcome.numel());
//    return noisy_outcome_vector;
//
//
//  }

std::vector<int> linearize(const std::vector<std::vector<int>>& vec_vec) {
  std::vector<int> vec;
  for (const auto& v : vec_vec) {
    for (auto d : v) {
      vec.push_back(d);
    }
  }
  return vec;
}



// TODO figure out how to support different architectures.
std::map<std::pair<int,int>, float> PyTorchWrapper::compute_scores(const std::vector<int>& node_kinds, const std::vector<std::vector<int>>& edges, const std::vector<int>& variables, const std::vector<std::vector<int>>& terms) {
  // what should this function do.



  // compute the embeddings, do the predictors and then do the big einsum.

  // how to store?

  //
  std::chrono::time_point<std::chrono::high_resolution_clock> start_compute_scores = std::chrono::high_resolution_clock::now();


  torch::DeviceType device_type;
  device_type = torch::kCUDA;
//  device_type = torch::kCPU;
  torch::Device device(device_type);

  GraphNet* gnn_t = static_cast<GraphNet*> (this->gnn);

  // TODO set model to eval mode
  torch::NoGradGuard no_grad;
  torch::Tensor index_tensor = torch::tensor(node_kinds, device);

  torch::Tensor node_embs;
  node_embs = gnn_t->nb[0].index_select(0, index_tensor);

  // TODO Add age?

  torch::Tensor source_nodes;

  std::vector<int> source_vector;
  std::vector<int> target_vector;
  std::vector<int> argument_index_vector;
  torch::Tensor edge_embs;

  // TODO investigate if subsampling edges could lead to speedup without too much performance hit
  for (auto& e : edges) {
    source_vector.push_back(e[0]);
    target_vector.push_back(e[1]);
    argument_index_vector.push_back(e[2]);

  }

  torch::Tensor source_index_tensor = torch::tensor(source_vector, device);
  torch::Tensor target_index_tensor = torch::tensor(target_vector, device);
  torch::Tensor argument_index_tensor = torch::tensor(argument_index_vector, device);

  int message_passes = 10;
  edge_embs = gnn_t->eb[0].index_select(0, argument_index_tensor);
  std::chrono::time_point<std::chrono::high_resolution_clock> end_setup = std::chrono::high_resolution_clock::now();

  std::cout << "cost of nn setup: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_setup - start_compute_scores).count() << std::endl;
  std::chrono::time_point<std::chrono::high_resolution_clock> start_mp = std::chrono::high_resolution_clock::now();

  for (int l=0; l < message_passes; ++l) {

    std::chrono::time_point<std::chrono::high_resolution_clock> start_index_select = std::chrono::high_resolution_clock::now();
    source_nodes = node_embs.index_select(0, source_index_tensor);


    std::chrono::time_point<std::chrono::high_resolution_clock> end_index_select = std::chrono::high_resolution_clock::now();
    source_nodes = source_nodes + edge_embs;
    std::cout << "cost of index select: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_index_select - start_index_select).count() << std::endl;

    int64_t dim = 0;
    torch::optional<torch::Tensor> optional_out;
    torch::optional<int64_t> dim_size;

    torch::Tensor results;
    torch::Tensor results_max;


//    results = gnn_t->graph_conv->forward(scatter_mean(source_nodes, target_index_tensor, dim, optional_out, dim_size));

//    std::cout << source_nodes.sizes() << std::endl;
//    std::cout << target_index_tensor.sizes() << std::endl;

    // NORMAL
//    results = gnn_t->conv_vector[l]->forward(scatter_mean(source_nodes, target_index_tensor, dim, optional_out, dim_size));
//    std::cout << "convolution done" << std::endl;
//    std::cout << node_embs.sizes() << std::endl;

//    std::cout << results.sizes() << std::endl;

    // MEAN + MAX

    results = scatter_mean(source_nodes, target_index_tensor, dim, optional_out, dim_size);
    results_max = std::get<0>(scatter_max(source_nodes, target_index_tensor, dim, optional_out, dim_size));
    //
    torch::Tensor multi_agg_results;
    //      std::cout << results.sizes() << " " << results_max.sizes() << std::endl;
    multi_agg_results = torch::cat({results, results_max}, 1);

//          results = graph_conv->forward(multi_agg_results);
    results = gnn_t->conv_vector[l]->forward(multi_agg_results);

    node_embs = node_embs + torch::relu(results);

    node_embs = torch::nn::functional::layer_norm(node_embs, torch::nn::functional::LayerNormFuncOptions({64}).eps(2e-5));

  }

  std::chrono::time_point<std::chrono::high_resolution_clock> end_mp = std::chrono::high_resolution_clock::now();

  std::cout << "cost of mp: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_mp - start_mp).count() << std::endl;

  std::cout << "Embedding Computed of shape: " << node_embs.sizes() << std::endl;

  std::chrono::time_point<std::chrono::high_resolution_clock> start_scores = std::chrono::high_resolution_clock::now();

  std::chrono::time_point<std::chrono::high_resolution_clock> start_projection = std::chrono::high_resolution_clock::now();

  torch::Tensor variable_index_tensor = torch::tensor(variables, device);
  // duplication of terms, not optimal.
  torch::Tensor term_index_tensor = torch::tensor(linearize(terms), device);
  torch::Tensor variable_embeddings;
  torch::Tensor term_embeddings;

  //TODO add quantifier indexing and embeddings for premise selection.
  std::cout << "Indexing into node_embs with variable_index_tensor" << std::endl;
  variable_embeddings = gnn_t->predictor->forward(node_embs.index_select(0, variable_index_tensor));
  std::cout << "Indexing into node_embs with term_index_tensor" << std::endl;
  term_embeddings = gnn_t->predictor2->forward(node_embs.index_select(0, term_index_tensor));

  std::chrono::time_point<std::chrono::high_resolution_clock> end_projection = std::chrono::high_resolution_clock::now();

  std::cout << "cost of projectors: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_projection - start_projection).count() << std::endl;

//  variable_stack = variable_embeddings.repeat({static_cast<int>(terms.size()), 1});
  std::chrono::time_point<std::chrono::high_resolution_clock> start_dot = std::chrono::high_resolution_clock::now();


  std::vector<int> duplication_list;
  int varcounter = 0;
  for (auto& termlist : terms) {
//    duplication_list.push_back(static_cast<int>(termlist.size()))
    for (auto& term : termlist) {
      duplication_list.push_back(varcounter);
    }
    varcounter += 1;
  }

  torch::Tensor rep_var_index = torch::tensor(duplication_list, device);
  torch::Tensor replicated_variable_tensor;
  std::cout << "indexing into variable embeddings" << std::endl;
  replicated_variable_tensor = variable_embeddings.index_select(0, rep_var_index);
  std::vector<torch::Tensor> tensorvector;

  tensorvector.push_back(replicated_variable_tensor);
  tensorvector.push_back(term_embeddings);
  std::cout << replicated_variable_tensor.sizes() << std::endl;
  std::cout << term_embeddings.sizes() << std::endl;
  torch::TensorList tenslist = torch::TensorList(tensorvector);
  //  einsum_inputs =
  //https://discuss.pytorch.org/t/libtorch-how-to-convert-std-vector-torch-tensor-to-torch-tensorlist/106828/2
  torch::Tensor results;
  results = torch::einsum("ij,ij->i",tenslist);
  std::chrono::time_point<std::chrono::high_resolution_clock> end_dot = std::chrono::high_resolution_clock::now();

  std::cout << "cost of dot: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_dot - start_dot).count() << std::endl;
  bool sample = true;
  int beta = 10;
  if (sample) {
    auto random_noise = torch::rand_like(results, device);
    // gumbel
//    std::cout << "Random noise" << std::endl;
//    std::cout << random_noise.index({torch::indexing::Slice(0, 10)}) << std::endl;
    random_noise = beta * - torch::log(- torch::log(random_noise));
//    std::cout << "Log noise" << std::endl;
//    std::cout << random_noise.index({torch::indexing::Slice(0, 10)}) << std::endl;
//    std::cout << "Results" << std::endl;
//    std::cout << results.index({torch::indexing::Slice(0, 10)}) << std::endl;
    results = results + random_noise;
//    std::cout << "Noise added" << std::endl;
//    std::cout << results.index({torch::indexing::Slice(0, 10)}) << std::endl;
  }

  // now construct the map
  std::chrono::time_point<std::chrono::high_resolution_clock> start_mapconst = std::chrono::high_resolution_clock::now();

  std::map<std::pair<int, int>, float> return_map;
  std::map<int, std::map<int, float>> nested_return_map;
//  std::vector<double> results_vector;
  // TODO make this faster;
  std::chrono::time_point<std::chrono::high_resolution_clock> start_linearize = std::chrono::high_resolution_clock::now();

  std::vector<int> linearized_terms = linearize(terms);
  std::chrono::time_point<std::chrono::high_resolution_clock> end_linearize = std::chrono::high_resolution_clock::now();

  std::cout << "cost of last linearize: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_linearize - start_linearize).count() << std::endl;

  std::cout << "costinfo: " << torch::_shape_as_tensor(results)[0] << "device: " << results.device() <<  std::endl;
  std::chrono::time_point<std::chrono::high_resolution_clock> start_cpucopy = std::chrono::high_resolution_clock::now();
  torch::DeviceType device_type_cpu;
  device_type_cpu = torch::kCPU;
  torch::Device cpu_device(device_type_cpu);

  results = results.to(cpu_device);
  std::chrono::time_point<std::chrono::high_resolution_clock> end_cpucopy = std::chrono::high_resolution_clock::now();

  std::cout << "cost of cpu copy: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_cpucopy - start_cpucopy).count() << std::endl;
  std::cout << "costinfo: " << torch::_shape_as_tensor(results)[0] << "device: " << results.device() <<  std::endl;
//  results =


  std::chrono::time_point<std::chrono::high_resolution_clock> start_resultvec = std::chrono::high_resolution_clock::now();
  std::vector<float> resultsvector(results.data_ptr<float>(), results.data_ptr<float>() + results.numel());
  std::chrono::time_point<std::chrono::high_resolution_clock> end_resultvec = std::chrono::high_resolution_clock::now();
  std::cout << "cost of resultvec copy: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_resultvec - start_resultvec).count() << std::endl;
  std::cout << "costinfo: veclen " << resultsvector.size() << std::endl;

  // TODO when trained with Softmax, somehow normalize here before we're back in CVC5. Then we can PL sample there.


//  std::chrono::time_point<std::chrono::high_resolution_clock> start_loop_time = std::chrono::high_resolution_clock::now();
//  for (int i = 0; i < torch::_shape_as_tensor(results)[0].item<int>(); ++i) {
//    int var_id = variables[duplication_list[i]];
//    int t_id = linearized_terms[i];
//    float term_prediction = resultsvector[i];
//
//
//  }
//  std::chrono::time_point<std::chrono::high_resolution_clock> end_loop_time = std::chrono::high_resolution_clock::now();
//  std::cout << "cost of looping over results: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_loop_time - start_loop_time).count() << std::endl;
//
//  std::chrono::time_point<std::chrono::high_resolution_clock> start_loop_time_c = std::chrono::high_resolution_clock::now();
//  auto length_of_results = torch::_shape_as_tensor(results)[0].item<int>();
//
//  for (int i = 0; i < length_of_results; ++i) {
//    int var_id = variables[duplication_list[i]];
//    int t_id = linearized_terms[i];
//    float term_prediction = resultsvector[i];
//
//
//  }
//  std::chrono::time_point<std::chrono::high_resolution_clock> end_loop_time_c = std::chrono::high_resolution_clock::now();
//  std::cout << "cost of looping over results (precompute end condition): " << std::chrono::duration_cast<std::chrono::milliseconds>(end_loop_time_c - start_loop_time_c).count() << std::endl;
//  std::chrono::time_point<std::chrono::high_resolution_clock> start_nested_map = std::chrono::high_resolution_clock::now();
//  for (int i = 0; i < torch::_shape_as_tensor(results)[0].item<int>(); ++i) {
//    int var_id = variables[duplication_list[i]];
//    int t_id = linearized_terms[i];
//    float term_prediction = resultsvector[i];
//
//    if (nested_return_map.find(var_id) == nested_return_map.end()) {
//      std::map<int,float> new_inner_map;
//      nested_return_map[var_id] = new_inner_map;
//      nested_return_map[var_id][t_id] = term_prediction;
//    }
//    else {
//      nested_return_map[var_id][t_id] = term_prediction;
//    }
//
//  }
//
//  std::chrono::time_point<std::chrono::high_resolution_clock> end_nested_map = std::chrono::high_resolution_clock::now();
//
//  std::chrono::time_point<std::chrono::high_resolution_clock> start_innermapconst = std::chrono::high_resolution_clock::now();
//  for (int i = 0; i < torch::_shape_as_tensor(results)[0].item<int>(); ++i) {
//    auto key = std::make_pair(variables[duplication_list[i]], linearized_terms[i]);
//    auto value = resultsvector[i];
//    return_map.insert(std::map<std::pair<int,int>, float>::value_type(key,value));
//  }
//
//
//  std::chrono::time_point<std::chrono::high_resolution_clock> end_mapconst = std::chrono::high_resolution_clock::now();
//
//  std::cout << "cost of map construction: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_mapconst - start_mapconst).count() << std::endl;
//
//  std::cout << "cost of nested map construction: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_nested_map - start_nested_map).count() << std::endl;
//  std::cout << "cost of flat map construction: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_mapconst - start_innermapconst).count() << std::endl;

//
  std::chrono::time_point<std::chrono::high_resolution_clock> start_innermap_c = std::chrono::high_resolution_clock::now();
  auto length_of_results = torch::_shape_as_tensor(results)[0].item<int>();
  for (int i = 0; i < length_of_results; ++i) {
    auto key = std::make_pair(variables[duplication_list[i]], linearized_terms[i]);
    auto value = resultsvector[i];
    return_map.insert(std::map<std::pair<int,int>, float>::value_type(key,value));
  }
  std::chrono::time_point<std::chrono::high_resolution_clock> end_innermap_c = std::chrono::high_resolution_clock::now();
  std::cout << "cost of flat map construction (precompute end condition): " << std::chrono::duration_cast<std::chrono::milliseconds>(end_innermap_c - start_innermap_c).count() << std::endl;
  std::chrono::time_point<std::chrono::high_resolution_clock> end_scores = std::chrono::high_resolution_clock::now();

//  std::cout << "cost of scores: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_scores - start_scores).count() << std::endl;

  return return_map;
}

std::tuple<std::map<std::pair<int,int>, float>, std::vector<double>> PyTorchWrapper::compute_scores_and_premsel(const std::vector<int>& node_kinds, const std::vector<std::vector<int>>& edges, const std::vector<int>& targets, const std::vector<int>& variables, const std::vector<std::vector<int>>& terms) {
  // what should this function do.



  // compute the embeddings, do the predictors and then do the big einsum.

  // how to store?

  //
  std::chrono::time_point<std::chrono::high_resolution_clock> start_compute_scores = std::chrono::high_resolution_clock::now();


  torch::DeviceType device_type;
//  device_type = torch::kCUDA;
  device_type = torch::kCPU;


  torch::Device device(device_type);

  GraphNet* gnn_t = static_cast<GraphNet*> (this->gnn);

  // TODO set model to eval mode
  torch::NoGradGuard no_grad;
  torch::Tensor index_tensor = torch::tensor(node_kinds, device);

  torch::Tensor q_index_tensor = torch::tensor(targets, device);

  torch::Tensor node_embs;
  node_embs = gnn_t->nb[0].index_select(0, index_tensor);

  // TODO Add age?

  torch::Tensor source_nodes;

  std::vector<int> source_vector;
  std::vector<int> target_vector;
  std::vector<int> argument_index_vector;
  torch::Tensor edge_embs;

  // TODO investigate if subsampling edges could lead to speedup without too much performance hit
  for (auto& e : edges) {
    source_vector.push_back(e[0]);
    target_vector.push_back(e[1]);
    argument_index_vector.push_back(e[2]);

  }

  torch::Tensor source_index_tensor = torch::tensor(source_vector, device);
  torch::Tensor target_index_tensor = torch::tensor(target_vector, device);
  torch::Tensor argument_index_tensor = torch::tensor(argument_index_vector, device);

  int message_passes = 10;
  edge_embs = gnn_t->eb[0].index_select(0, argument_index_tensor);
  std::chrono::time_point<std::chrono::high_resolution_clock> end_setup = std::chrono::high_resolution_clock::now();

  std::cout << "cost of nn setup: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_setup - start_compute_scores).count() << std::endl;
  std::chrono::time_point<std::chrono::high_resolution_clock> start_mp = std::chrono::high_resolution_clock::now();

  for (int l=0; l < message_passes; ++l) {

    std::chrono::time_point<std::chrono::high_resolution_clock> start_index_select = std::chrono::high_resolution_clock::now();
    source_nodes = node_embs.index_select(0, source_index_tensor);


    std::chrono::time_point<std::chrono::high_resolution_clock> end_index_select = std::chrono::high_resolution_clock::now();
    source_nodes = source_nodes + edge_embs;
    std::cout << "cost of index select: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_index_select - start_index_select).count() << std::endl;

    int64_t dim = 0;
    torch::optional<torch::Tensor> optional_out;
    torch::optional<int64_t> dim_size;

    torch::Tensor results;
    torch::Tensor results_max;


    //    results = gnn_t->graph_conv->forward(scatter_mean(source_nodes, target_index_tensor, dim, optional_out, dim_size));

    //    std::cout << source_nodes.sizes() << std::endl;
    //    std::cout << target_index_tensor.sizes() << std::endl;

    // NORMAL
    //    results = gnn_t->conv_vector[l]->forward(scatter_mean(source_nodes, target_index_tensor, dim, optional_out, dim_size));
    //    std::cout << "convolution done" << std::endl;
    //    std::cout << node_embs.sizes() << std::endl;

    //    std::cout << results.sizes() << std::endl;

    // MEAN + MAX

    results = scatter_mean(source_nodes, target_index_tensor, dim, optional_out, dim_size);
    results_max = std::get<0>(scatter_max(source_nodes, target_index_tensor, dim, optional_out, dim_size));
    //
    torch::Tensor multi_agg_results;
    //      std::cout << results.sizes() << " " << results_max.sizes() << std::endl;
    multi_agg_results = torch::cat({results, results_max}, 1);

    //          results = graph_conv->forward(multi_agg_results);
    results = gnn_t->conv_vector[l]->forward(multi_agg_results);

    node_embs = node_embs + torch::relu(results);

    node_embs = torch::nn::functional::layer_norm(node_embs, torch::nn::functional::LayerNormFuncOptions({64}).eps(2e-5));

  }

  std::chrono::time_point<std::chrono::high_resolution_clock> end_mp = std::chrono::high_resolution_clock::now();

  std::cout << "cost of mp: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_mp - start_mp).count() << std::endl;

  std::cout << "Embedding Computed of shape: " << node_embs.sizes() << std::endl;

  std::chrono::time_point<std::chrono::high_resolution_clock> start_scores = std::chrono::high_resolution_clock::now();

  std::chrono::time_point<std::chrono::high_resolution_clock> start_projection = std::chrono::high_resolution_clock::now();

  torch::Tensor variable_index_tensor = torch::tensor(variables, device);
  // duplication of terms, not optimal.
  torch::Tensor term_index_tensor = torch::tensor(linearize(terms), device);
  torch::Tensor variable_embeddings;
  torch::Tensor term_embeddings;

  torch::Tensor target_scores;

  //TODO add quantifier indexing and embeddings for premise selection.
  std::cout << "Indexing into node_embs with variable_index_tensor" << std::endl;
  variable_embeddings = gnn_t->predictor->forward(node_embs.index_select(0, variable_index_tensor));
  std::cout << "Indexing into node_embs with term_index_tensor" << std::endl;
  term_embeddings = gnn_t->predictor2->forward(node_embs.index_select(0, term_index_tensor));

  target_scores = gnn_t->predictor3->forward(node_embs.index_select(0, q_index_tensor));

  std::chrono::time_point<std::chrono::high_resolution_clock> end_projection = std::chrono::high_resolution_clock::now();

  std::cout << "cost of projectors: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_projection - start_projection).count() << std::endl;

  //  variable_stack = variable_embeddings.repeat({static_cast<int>(terms.size()), 1});
  std::chrono::time_point<std::chrono::high_resolution_clock> start_dot = std::chrono::high_resolution_clock::now();


  std::vector<int> duplication_list;
  int varcounter = 0;
  for (auto& termlist : terms) {
    //    duplication_list.push_back(static_cast<int>(termlist.size()))
    for (auto& term : termlist) {
      duplication_list.push_back(varcounter);
    }
    varcounter += 1;
  }

  torch::Tensor rep_var_index = torch::tensor(duplication_list, device);
  torch::Tensor replicated_variable_tensor;
  std::cout << "indexing into variable embeddings" << std::endl;
  replicated_variable_tensor = variable_embeddings.index_select(0, rep_var_index);
  std::vector<torch::Tensor> tensorvector;

  tensorvector.push_back(replicated_variable_tensor);
  tensorvector.push_back(term_embeddings);
  std::cout << replicated_variable_tensor.sizes() << std::endl;
  std::cout << term_embeddings.sizes() << std::endl;
  torch::TensorList tenslist = torch::TensorList(tensorvector);
  //  einsum_inputs =
  //https://discuss.pytorch.org/t/libtorch-how-to-convert-std-vector-torch-tensor-to-torch-tensorlist/106828/2
  torch::Tensor results;
  results = torch::einsum("ij,ij->i",tenslist);
  std::chrono::time_point<std::chrono::high_resolution_clock> end_dot = std::chrono::high_resolution_clock::now();

  std::cout << "cost of dot: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_dot - start_dot).count() << std::endl;
//  compute_scores_and_premsel
  //
  bool sample = false;
  int beta = 1;
  if (sample) {
    auto random_noise = torch::rand_like(results, device);
    // gumbel
    //    std::cout << "Random noise" << std::endl;
    //    std::cout << random_noise.index({torch::indexing::Slice(0, 10)}) << std::endl;
    random_noise = beta * - torch::log(- torch::log(random_noise));
    //    std::cout << "Log noise" << std::endl;
    //    std::cout << random_noise.index({torch::indexing::Slice(0, 10)}) << std::endl;
    //    std::cout << "Results" << std::endl;
    //    std::cout << results.index({torch::indexing::Slice(0, 10)}) << std::endl;
    results = results + random_noise;
    //    std::cout << "Noise added" << std::endl;
    //    std::cout << results.index({torch::indexing::Slice(0, 10)}) << std::endl;
  }

  // now construct the map
  std::chrono::time_point<std::chrono::high_resolution_clock> start_mapconst = std::chrono::high_resolution_clock::now();

  std::map<std::pair<int, int>, float> return_map;
  std::map<int, std::map<int, float>> nested_return_map;
  //  std::vector<double> results_vector;
  // TODO make this faster;
  std::chrono::time_point<std::chrono::high_resolution_clock> start_linearize = std::chrono::high_resolution_clock::now();

  std::vector<int> linearized_terms = linearize(terms);
  std::chrono::time_point<std::chrono::high_resolution_clock> end_linearize = std::chrono::high_resolution_clock::now();

  std::cout << "cost of last linearize: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_linearize - start_linearize).count() << std::endl;

  std::cout << "costinfo: " << torch::_shape_as_tensor(results)[0] << "device: " << results.device() <<  std::endl;
  std::chrono::time_point<std::chrono::high_resolution_clock> start_cpucopy = std::chrono::high_resolution_clock::now();
  torch::DeviceType device_type_cpu;
  device_type_cpu = torch::kCPU;
  torch::Device cpu_device(device_type_cpu);

  results = results.to(cpu_device);
  std::chrono::time_point<std::chrono::high_resolution_clock> end_cpucopy = std::chrono::high_resolution_clock::now();

  std::cout << "cost of cpu copy: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_cpucopy - start_cpucopy).count() << std::endl;
  std::cout << "costinfo: " << torch::_shape_as_tensor(results)[0] << "device: " << results.device() <<  std::endl;
  //  results =


  std::chrono::time_point<std::chrono::high_resolution_clock> start_resultvec = std::chrono::high_resolution_clock::now();
  std::vector<float> resultsvector(results.data_ptr<float>(), results.data_ptr<float>() + results.numel());
  std::chrono::time_point<std::chrono::high_resolution_clock> end_resultvec = std::chrono::high_resolution_clock::now();
  std::cout << "cost of resultvec copy: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_resultvec - start_resultvec).count() << std::endl;
  std::cout << "costinfo: veclen " << resultsvector.size() << std::endl;

  // TODO when trained with Softmax, somehow normalize here before we're back in CVC5. Then we can PL sample there.


  //  std::chrono::time_point<std::chrono::high_resolution_clock> start_loop_time = std::chrono::high_resolution_clock::now();
  //  for (int i = 0; i < torch::_shape_as_tensor(results)[0].item<int>(); ++i) {
  //    int var_id = variables[duplication_list[i]];
  //    int t_id = linearized_terms[i];
  //    float term_prediction = resultsvector[i];
  //
  //
  //  }
  //  std::chrono::time_point<std::chrono::high_resolution_clock> end_loop_time = std::chrono::high_resolution_clock::now();
  //  std::cout << "cost of looping over results: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_loop_time - start_loop_time).count() << std::endl;
  //
  //  std::chrono::time_point<std::chrono::high_resolution_clock> start_loop_time_c = std::chrono::high_resolution_clock::now();
  //  auto length_of_results = torch::_shape_as_tensor(results)[0].item<int>();
  //
  //  for (int i = 0; i < length_of_results; ++i) {
  //    int var_id = variables[duplication_list[i]];
  //    int t_id = linearized_terms[i];
  //    float term_prediction = resultsvector[i];
  //
  //
  //  }
  //  std::chrono::time_point<std::chrono::high_resolution_clock> end_loop_time_c = std::chrono::high_resolution_clock::now();
  //  std::cout << "cost of looping over results (precompute end condition): " << std::chrono::duration_cast<std::chrono::milliseconds>(end_loop_time_c - start_loop_time_c).count() << std::endl;
  //  std::chrono::time_point<std::chrono::high_resolution_clock> start_nested_map = std::chrono::high_resolution_clock::now();
  //  for (int i = 0; i < torch::_shape_as_tensor(results)[0].item<int>(); ++i) {
  //    int var_id = variables[duplication_list[i]];
  //    int t_id = linearized_terms[i];
  //    float term_prediction = resultsvector[i];
  //
  //    if (nested_return_map.find(var_id) == nested_return_map.end()) {
  //      std::map<int,float> new_inner_map;
  //      nested_return_map[var_id] = new_inner_map;
  //      nested_return_map[var_id][t_id] = term_prediction;
  //    }
  //    else {
  //      nested_return_map[var_id][t_id] = term_prediction;
  //    }
  //
  //  }
  //
  //  std::chrono::time_point<std::chrono::high_resolution_clock> end_nested_map = std::chrono::high_resolution_clock::now();
  //
  //  std::chrono::time_point<std::chrono::high_resolution_clock> start_innermapconst = std::chrono::high_resolution_clock::now();
  //  for (int i = 0; i < torch::_shape_as_tensor(results)[0].item<int>(); ++i) {
  //    auto key = std::make_pair(variables[duplication_list[i]], linearized_terms[i]);
  //    auto value = resultsvector[i];
  //    return_map.insert(std::map<std::pair<int,int>, float>::value_type(key,value));
  //  }
  //
  //
  //  std::chrono::time_point<std::chrono::high_resolution_clock> end_mapconst = std::chrono::high_resolution_clock::now();
  //
  //  std::cout << "cost of map construction: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_mapconst - start_mapconst).count() << std::endl;
  //
  //  std::cout << "cost of nested map construction: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_nested_map - start_nested_map).count() << std::endl;
  //  std::cout << "cost of flat map construction: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_mapconst - start_innermapconst).count() << std::endl;

  //
  std::chrono::time_point<std::chrono::high_resolution_clock> start_innermap_c = std::chrono::high_resolution_clock::now();
  auto length_of_results = torch::_shape_as_tensor(results)[0].item<int>();
  for (int i = 0; i < length_of_results; ++i) {
    auto key = std::make_pair(variables[duplication_list[i]], linearized_terms[i]);
    auto value = resultsvector[i];
    return_map.insert(std::map<std::pair<int,int>, float>::value_type(key,value));
  }
  std::chrono::time_point<std::chrono::high_resolution_clock> end_innermap_c = std::chrono::high_resolution_clock::now();
  std::cout << "cost of flat map construction (precompute end condition): " << std::chrono::duration_cast<std::chrono::milliseconds>(end_innermap_c - start_innermap_c).count() << std::endl;
  std::chrono::time_point<std::chrono::high_resolution_clock> end_scores = std::chrono::high_resolution_clock::now();

  //  std::cout << "cost of scores: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_scores - start_scores).count() << std::endl;
  std::cout << target_scores.sizes() << std::endl;

  // implement a temperature here

//  float temperature = 0.1;

  target_scores = torch::sigmoid(target_scores);
  target_scores = target_scores.to(cpu_device);
  std::vector<float> target_scores_vector(target_scores.data_ptr<float>(), target_scores.data_ptr<float>() + target_scores.numel());
  std::vector<double> target_scores_vector_doubles(target_scores_vector.begin(), target_scores_vector.end());
  std::cout << "target scores vector made" << std::endl;
  return std::make_tuple(return_map, target_scores_vector_doubles);
}

void PyTorchWrapper::compute_embeddings(const std::vector<int>& node_kinds, const std::vector<std::vector<int>>& edges) {
  // this function just runs the gnn and stores the results in this->computed_embeddings

  torch::DeviceType device_type;
  device_type = torch::kCUDA;
  torch::Device device(device_type);

  //std::cout << "Try to cast" << std::endl;
  GraphNet* gnn_t = static_cast<GraphNet*> (this->gnn);
  //std::cout << "Cast successful " << std::endl;

  //std::cout << "Making some tensors" << std::endl;
  torch::Tensor index_tensor = torch::tensor(node_kinds, device);
//  torch::Tensor quantifier_index_tensor = torch::tensor(targets, device);
  torch::Tensor node_embs;
  //std::cout << "Tensors created" << std::endl;

  node_embs = gnn_t->nb[0].index_select(0, index_tensor);
  //std::cout << "Indexing into embds" << std::endl;
  torch::Tensor source_nodes;

  std::vector<int> source_vector;
  std::vector<int> target_vector;

  for (auto& e : edges) {
    source_vector.push_back(e[0]);
    target_vector.push_back(e[1]);

  }

  torch::Tensor source_index_tensor = torch::tensor(source_vector, device);
  torch::Tensor target_index_tensor = torch::tensor(target_vector, device);
  int message_passes = 10;

  for (int l=0; l < message_passes; ++l) {

    source_nodes = node_embs.index_select(0, source_index_tensor);
    int64_t dim = 0;
    torch::optional<torch::Tensor> optional_out;
    torch::optional<int64_t> dim_size;

    torch::Tensor results;
    results = gnn_t->graph_conv->forward(scatter_sum(source_nodes, target_index_tensor, dim, optional_out, dim_size));

    node_embs = node_embs + torch::relu(results);

    //node_embs = gnn_t->graph_conv->forward(node_embs);

    node_embs = torch::nn::functional::layer_norm(node_embs, torch::nn::functional::LayerNormFuncOptions({64}).eps(2e-5));


    //node_embs = torch::relu(node_embs);

  }

  // store them in the wrapper. for now, it's quite inefficient, but don't want
  // to rewrite half the solver logic to stack the computations at the moment.
  std::cout << "Embedding Computed of shape: " << node_embs.sizes() << std::endl;
  this->computed_embeddings = new torch::Tensor(node_embs);

}

PyTorchWrapper::PyTorchWrapper(std::string model_location)

{
  std::chrono::time_point<std::chrono::high_resolution_clock> start_model_load = std::chrono::high_resolution_clock::now();
  std::cout << "PyTorchWrapper constructor" << std::endl;
  //GraphNet* gnn_t = static_cast<GraphNet*> (gnn);
  std::cout << "After wacky move" << std::endl;
  torch::serialize::InputArchive archive;
  std::string file(model_location);

  // Load from GPU (or actually from the device it was saved on)
//  archive.load_from(file);

  // Loading to CPU
  torch::DeviceType device_type;
  device_type = torch::kCPU;
  torch::Device device(device_type);
  archive.load_from(file, device);

  //
  std::cout << "loading archive" << std::endl;
  std::cout << model_location << std::endl;

  GraphNet* gnn_pl = new GraphNet;

  gnn_pl->load(archive);

  std::cout << "loading graphnet" << std::endl;
  at::set_num_threads(1);
  at::set_num_interop_threads(1);

  auto parameters = gnn_pl->named_parameters();
  auto keys = parameters.keys();
  auto val = parameters.values();

  for (auto v : keys) {
    std::cout << v << std::endl;
  }
  std::cout << "Layer9" << std::endl;
  std::cout << gnn_pl->conv_vector[9]->weight.index({torch::indexing::Slice(0, 2)}) << std::endl;
  std::cout << "Node INITS" << std::endl;
  std::cout << gnn_pl->eb[0].index({torch::indexing::Slice(0, 2)}) << std::endl;
  std::cout << "Edge INITS" << std::endl;
  std::cout << gnn_pl->nb[0].index({torch::indexing::Slice(0, 2)}) << std::endl;

  gnn_pl->initted = true;

  std::cout << "Saved Model: " << std::endl;
  std::cout << c10::str(gnn_pl) << std::endl;


  this->gnn = gnn_pl;
  std::cout << "Model loaded" << std::endl;
  std::chrono::time_point<std::chrono::high_resolution_clock> end_model_load = std::chrono::high_resolution_clock::now();
  std::cout << "cost of model load: " << std::chrono::duration_cast<std::chrono::milliseconds>(end_model_load - start_model_load).count() << std::endl;

}

