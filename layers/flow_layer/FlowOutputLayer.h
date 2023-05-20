//
// Created by Matthew Prytula on 20.05.2023.
//

#ifndef DEEPDENDRO_FLOWOUTPUTLAYER_H
#define DEEPDENDRO_FLOWOUTPUTLAYER_H
#include "FlowLayer.h"

class FlowOutputLayer : public FlowLayer {
    tbb::concurrent_unordered_map<size_t, MatrixXd> a_value_stash;  // stash a_value for each micro-batch

public:
    FlowOutputLayer(const int curr_neurons, Shape input_shape, ActivationFunc activation, size_t update_num) :
            FlowLayer(curr_neurons, input_shape, activation, update_num)
    { }

    void forward_prop(const MatrixXd &prev_a_values) override {
        FlowLayer::forward_prop(prev_a_values);
        a_value_stash[micro_batch_num_forw] = a_value;
    }

    MatrixXd calc_first_back_prop(const MatrixXd &labels, size_t micro_batch_num) {
        dz_value = a_value_stash[micro_batch_num] - labels;

        dz_values.emplace_back(dz_value); // store for weight updates
        // todo: may cause UB. Erase is unsafe
        a_value_stash.unsafe_erase(micro_batch_num);
        return calc_gradient();
    }
};


#endif //DEEPDENDRO_FLOWOUTPUTLAYER_H
