#ifndef PYTORCH_WRAPPER_H_11629
#define PYTORCH_WRAPPER_H_11629

#include <vector>
#include <string>
#include <map>


// this line is wrong. hide torch from CVC5 codebase.
//#include <torch/torch.h>
// pointer to mystery class
    //

//struct GraphNet {

                  //};

//struct hash_intpair {
//    auto operator()(const std::pair<int, int> &p) const -> size_t {
//    return std::hash<int>{}(p.first) ^ std::hash<int>{}(p.second);
//  }
//};
//
//struct hash_intpair_cantor {
//    auto operator()(const std::pair<int, int> &p) const -> size_t {
//
//      size_t x = std::hash<int>{}(p.first)
//      size_t y = std::hash<int>{}(p.second)
//    return  (x*x + x + 2*x*y + 3*y + y*y) / 2;
//  }
//};

class PyTorchWrapper {
public:
    int foobar();
std::vector<double> predict(const std::vector<int>& node_kinds, const std::vector<std::vector<int>>& edges, const std::vector<int>& targets);
std::vector<double> rank_terms(const std::vector<int>& variables, const std::vector<int>& terms);
std::map<std::pair<int,int>, float> compute_scores(const std::vector<int>& node_kinds, const std::vector<std::vector<int>>& edges, const std::vector<int>& variables, const std::vector<std::vector<int>>& terms);


std::tuple<std::map<std::pair<int,int>, float>, std::vector<double>> compute_scores_and_premsel(const std::vector<int>& node_kinds, const std::vector<std::vector<int>>& edges, const std::vector<int>& targets, const std::vector<int>& variables, const std::vector<std::vector<int>>& terms);

void compute_embeddings(const std::vector<int>& node_kinds, const std::vector<std::vector<int>>& edges);
    // Don't talk about any torch type, because it requires me to pull the torch.h into here, which causes a macro conflict for "Warning"
void* gnn;
void* computed_embeddings;


PyTorchWrapper(std::string model_location);


};

#endif /* PYTORCH_WRAPPER_H_11629 */