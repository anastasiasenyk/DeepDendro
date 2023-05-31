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
#include <iostream>
#include "Layer.h"
#include "activationFuncs.h"
#include "activationDerivative.h"

class FlowLayer : public Layer {
    ActivationFunc activ_func;
    tbb::concurrent_vector<MatrixXd> weight_stash; // for weight stashing of different micro-batches
    // TODO: bias is just the same number in a vector. No need to store it in a vector, just one value is enough
    tbb::concurrent_vector<VectorXd> bias_stash; // for bias stashing of different micro-batches
    tbb::concurrent_unordered_map<size_t, size_t> stash_map; // for keeping correspondence between micro-batch and its weights

    MatrixXd z_value;

    tbb::concurrent_queue<MatrixXd> z_values; // for dZ computing



    std::queue<MatrixXd> received_activations; // for weight update.

    size_t update_after;
    std::mutex mtx;
protected:

    std::atomic<size_t> micro_batch_num_forw;
    std::atomic<size_t> micro_batch_num_back;
    std::vector<MatrixXd> dz_values; // for weight update
    MatrixXd dz_value;
    MatrixXd a_value;
public:
    const MatrixXd &getAValue() const {
        return a_value;
    }

    FlowLayer(const int curr_neurons, Shape input_shape, ActivationFunc activation, size_t update_num):
            activ_func(activation), micro_batch_num_forw(0), micro_batch_num_back(0),update_after(update_num)
    {
        weight_stash.emplace_back(MatrixXd::Random(curr_neurons, input_shape.first)/ sqrt(input_shape.first));
        bias_stash.emplace_back(VectorXd::Zero(curr_neurons));
        shape.first = curr_neurons;
        shape.second = input_shape.second;
    }

    virtual void forward_prop(const MatrixXd &prev_a_values, bool is_first=false) {
        std::lock_guard<std::mutex> lock(mtx);

        // save received activations for weight update
        received_activations.push(prev_a_values);

        // always use the latest version of weights for forward prop
        z_value = weight_stash.back() * prev_a_values;
//        for (int i = 0; i < weight_stash.back().rows(); ++i) {
//            for (int j = 0; j <weight_stash.back().cols(); ++j) {
//                std::cout << weight_stash.back()(i, j) << " ";
//            }
//            std::cout << "\n";
//        }

//        if(is_first) std::cout << weight_stash.back().rows() << " " << weight_stash.back().cols() << " | " << prev_a_values.rows() << " " << prev_a_values.cols() << " | " << z_value.rows() << " " << z_value.cols() << " | " << bias_stash.back().rows() << " " << bias_stash.back().cols() <<"\n";
        z_value.colwise() += bias_stash.back();
        z_values.emplace(z_value); // store for backprop
//        if (is_first) {
//            for (int i = 0; i < bias_stash[0].size(); ++i) {
//                std::cout << bias_stash[bias_stash.size()-1][i] << " ";
//            }
//            std::cout << "\n";
//        }

        a_value = activ_func(z_value);

        // stash weights
        stash_map[micro_batch_num_forw++] = weight_stash.size()-1;

    }

    inline MatrixXd calc_gradient() {
        return weight_stash[stash_map[micro_batch_num_back++]].transpose() * dz_value;
    }

    MatrixXd back_prop (const MatrixXd &gradient) {
        std::lock_guard<std::mutex> lock(mtx);
        // there's no way, one micro-batch overtakes another,
        // so their backprop will be calculated in the same order as forward prop
        // therefore, queue is the best choice here
        if(z_values.try_pop(z_value)) {
            MatrixXd act_func_der = find_activation_der(activ_func)(z_value);
            dz_value = gradient.cwiseProduct(act_func_der);
            dz_values.emplace_back(dz_value); // store for weight updates
            return calc_gradient();
        }
    }

//    MatrixXd calc_first_back_prop(const MatrixXd &labels) {
//        dz_value = a_value - labels;
//        dz_values.emplace_back(dz_value); // store for weight updates
//        return calc_gradient();
//    }

    void update_weights(double learning_rate) {
        std::lock_guard<std::mutex> lock(mtx);
        // Only update weights every m micro-batches

        MatrixXd res_weights = MatrixXd::Zero(dz_values.front().rows(), dz_values.front().cols());
        VectorXd res_biases = VectorXd::Zero(dz_values.front().rows());

        // Compute the average of the gradients over m micro-batches
        MatrixXd temp;
        for (int i = 0; i < update_after; ++i) {
            res_weights += dz_values[i];
            res_biases += dz_values[i].rowwise().sum();  // Bias gradient is the sum across the columns
        }
        res_weights /= static_cast<double>(update_after);
        res_biases /= static_cast<double>(update_after);

        // Perform the weight and bias updates using the average gradients
        // todo: run across the weights version used for current m micro-batches and delete those
        // weights from the weight_stash, as they require a lot of space
        MatrixXd tmp = weight_stash.back() - learning_rate * res_weights * received_activations.front().transpose();
        MatrixXd tmp2 = bias_stash.back() - learning_rate * res_biases;
//        weight_stash.back() -= learning_rate * res_weights * received_activations.front().transpose();
        weight_stash.emplace_back(tmp);
//        bias_stash.back() -= learning_rate * res_biases;
        bias_stash.emplace_back(tmp2);
        received_activations.pop(); // you need to pop the used activations as well
        // Remove the used gradients
        dz_values.erase(dz_values.begin(), dz_values.begin() + update_after);
    }

};


#endif //DEEPDENDRO_FLOWLAYER_H
