//
// Created by Matthew Prytula on 07.05.2023.
//

#ifndef DEEPDENDRO_FLOWLAYER_H
#define DEEPDENDRO_FLOWLAYER_H

#include <tbb/concurrent_vector.h>
#include <tbb/concurrent_unordered_map.h>
#include <tbb/concurrent_queue.h>
#include <atomic>
#include <queue>
#include "Layer.h"
#include "activationFuncs.h"
#include "activationDerivative.h"

class FlowLayer : public Layer {
    ActivationFunc activ_func;
    tbb::concurrent_vector<MatrixXd> weight_stash; // for weight stashing of different micro-batches
    tbb::concurrent_vector<VectorXd> bias_stash; // for bias stashing of different micro-batches
    tbb::concurrent_unordered_map<size_t, size_t> stash_map; // for keeping correspondence between micro-batch and its weights

    MatrixXd z_value;

    std::queue<MatrixXd> z_values; // for dZ computing



    std::queue<MatrixXd> received_activations; // for weight update.

    size_t update_after;
protected:

    std::atomic<size_t> micro_batch_num_forw;
    std::atomic<size_t> micro_batch_num_back;
    std::vector<MatrixXd> dz_values; // for weight update
    MatrixXd dz_value;
    MatrixXd a_value;

public:
    FlowLayer(const int curr_neurons, Shape input_shape, ActivationFunc activation, size_t update_num):
            activ_func(activation), micro_batch_num_forw(0), micro_batch_num_back(0),update_after(update_num)
    {
        weight_stash.emplace_back(MatrixXd::Random(curr_neurons, input_shape.first)/ sqrt(input_shape.first));
        bias_stash.emplace_back(VectorXd::Zero(curr_neurons));
        shape.first = curr_neurons;
        shape.second = input_shape.second;
    }

    virtual void forward_prop(const MatrixXd &prev_a_values) {
        // save received activations for weight update
        received_activations.push(prev_a_values);

        // always use the latest version of weights for forward prop
        z_value = weight_stash.back() * prev_a_values;
        z_value.colwise() += bias_stash.back();
        a_value = activ_func(z_value);

        // stash weights
        stash_map[micro_batch_num_forw++] = weight_stash.size();

    }

    inline MatrixXd calc_gradient() {
        return weight_stash[stash_map[micro_batch_num_back++]].transpose() * dz_value;
    }

    MatrixXd back_prop (const MatrixXd &gradient) {
        // there's no way, one micro-batch overtakes another,
        // so their backprop will be calculated in the same order as forward prop
        // therefore, queue is the best choice here
        MatrixXd act_func_der = find_activation_der(activ_func)(z_values.front());
        z_values.pop(); // pop front value, basically the oldest one in the queue

        dz_value = gradient.cwiseProduct(act_func_der);
        dz_values.emplace_back(dz_value); // store for weight updates
        return calc_gradient();
    }

    MatrixXd calc_first_back_prop(const MatrixXd &labels) {
        dz_value = a_value - labels;
        dz_values.emplace_back(dz_value); // store for weight updates
        return calc_gradient();
    }

    void update_weights(double learning_rate) {
        // Only update weights every m micro-batches
        if (micro_batch_num_back < update_after) {
            return;
        }

        MatrixXd res_weights = MatrixXd::Zero(dz_values.front().rows(), dz_values.front().cols());
        VectorXd res_biases = VectorXd::Zero(dz_values.front().rows());

        // Compute the average of the gradients over m micro-batches
        for (int i = 0; i < update_after; ++i) {
            res_weights += dz_values[i];
            res_biases += dz_values[i].colwise().sum();  // Bias gradient is the sum across the columns
        }
        res_weights /= static_cast<double>(update_after);
        res_biases /= static_cast<double>(update_after);

        // Perform the weight and bias updates using the average gradients
        // todo: run across the weights version used for current m micro-batches and delete those
        // weights from the weight_stash, as they require a lot of space
        weight_stash.back() -= learning_rate * res_weights * received_activations.front().transpose();
        bias_stash.back() -= learning_rate * res_biases;
        received_activations.pop(); // you need to pop the used activations as well

        // Remove the used gradients
        dz_values.erase(dz_values.begin(), dz_values.begin() + update_after);
    }

};


#endif //DEEPDENDRO_FLOWLAYER_H
