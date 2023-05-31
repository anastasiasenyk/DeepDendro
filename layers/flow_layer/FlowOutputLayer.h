//
// Created by Matthew Prytula on 20.05.2023.
//

#ifndef DEEPDENDRO_FLOWOUTPUTLAYER_H
#define DEEPDENDRO_FLOWOUTPUTLAYER_H

#include <iostream>
#include "FlowLayer.h"
#include "lossFunc.h"

class FlowOutputLayer : public FlowLayer {
    tbb::concurrent_unordered_map<size_t, MatrixXd> a_value_stash;  // stash a_value for each micro-batch
    tbb::concurrent_queue<MatrixXd> labels;
public:
    FlowOutputLayer(const int curr_neurons, Shape input_shape, ActivationFunc activation, size_t update_num) :
            FlowLayer(curr_neurons, input_shape, activation, update_num)
    { }

    void forward_prop(const MatrixXd &prev_a_values, bool is_first=false) override {
        FlowLayer::forward_prop(prev_a_values);
        a_value_stash[micro_batch_num_forw] = a_value;
    }

    MatrixXd calc_first_back_prop(const MatrixXd &prev_a_values) {
        MatrixXd curr_labels;
        if (labels.try_pop(curr_labels)) {
            dz_value = prev_a_values - curr_labels;
            std::cout << lossFunc().categoryCrossEntropy(prev_a_values, curr_labels) << "\n";
            dz_values.emplace_back(dz_value); // store for weight updates
            return calc_gradient();
        }
    }

    void set_labels(const MatrixXd& micro_label) {
//        std::cout << "RECEIVED LABELS\n";
        labels.emplace(micro_label);
    }
};


#endif //DEEPDENDRO_FLOWOUTPUTLAYER_H
