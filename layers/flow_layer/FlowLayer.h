//
// Created by Matthew Prytula on 07.05.2023.
//

#ifndef DEEPDENDRO_FLOWLAYER_H
#define DEEPDENDRO_FLOWLAYER_H

#include <tbb/concurrent_vector.h>
#include <tbb/concurrent_unordered_map.h>
#include <tbb/concurrent_queue.h>
#include <atomic>
#include "Layer.h"
#include "activationFuncs.h"

class FlowLayer : public Layer {
    ActivationFunc activ_func;
    tbb::concurrent_vector<MatrixXd> weight_stash; // for weight stashing of different micro-batches
    tbb::concurrent_vector<VectorXd> bias_stash; // for bias stashing of different micro-batches
    tbb::concurrent_unordered_map<size_t, size_t> stash_map; // for keeping correspondence between micro-batch and its weights
    std::atomic<size_t> micro_batch_num;

    MatrixXd a_value;
    MatrixXd z_value;
    tbb::concurrent_queue<MatrixXd> z_values; // for dZ computing

    // TODO: maybe change to regular queue, or other as it will be averaged anyway
    tbb::concurrent_queue<MatrixXd> dz_values; // for weight update

    // TODO: maybe change to regular queue, or other as it will be averaged anyway
    tbb::concurrent_queue<MatrixXd> received_activations; // for weight update.

public:
    FlowLayer(const int curr_neurons, Shape input_shape, ActivationFunc activation):
    activ_func(activation), micro_batch_num(0)
    {
        weight_stash.emplace_back(MatrixXd::Random(curr_neurons, input_shape.first)/ sqrt(input_shape.first));
        bias_stash.emplace_back(VectorXd::Zero(curr_neurons));
        shape.first = curr_neurons;
        shape.second = input_shape.second;
    }

    void forward_prop(const MatrixXd &prev_a_values) {
        // save received activations for weight update
        received_activations.push(prev_a_values);

        // always use the latest version of weights for forward prop
        z_value = weight_stash.back() * prev_a_values;
        z_value.colwise() += bias_stash.back();
        a_value = activ_func(z_value);

        // stash weights
        stash_map[micro_batch_num++] = weight_stash.size();


    }
};


#endif //DEEPDENDRO_FLOWLAYER_H
