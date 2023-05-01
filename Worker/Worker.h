//
// Created by Matthew Prytula on 19.04.2023.
//

#ifndef DEEPDENDRO_WORKER_H
#define DEEPDENDRO_WORKER_H

#include <tbb/concurrent_queue.h>
#include <vector>
#include "HiddenLayer.h"

using namespace tbb;

class Worker {
    HiddenLayer& layer_;
    int worker_id_;
    size_t batch_size;
    tbb::concurrent_queue<MatrixXd> activationsQ;
    tbb::concurrent_queue<MatrixXd> deltaWQ;
    std::vector<MatrixXd> activationsContainer;
    std::vector<MatrixXd> zValuesContainer;
public:
    Worker(int worker_id, HiddenLayer& layer): worker_id_(worker_id), layer_(layer) {}

    void operator()(concurrent_queue<MatrixXd>& nextActivations, concurrent_queue<MatrixXd>& prevDeltaW) {
        while(true) {
            MatrixXd activations(layer_.shape.first, layer_.shape.second);
            if(activationsQ.try_pop(activations)) {
                // for computing gradient store the activations from the prev layer
                activationsContainer.emplace_back(activations);
                layer_.forward_prop(activations);
                zValuesContainer.emplace_back(layer_.getZValues());
                nextActivations.push(layer_.getAValues());
                // here, activation is just gradient
            } else if (deltaWQ.try_pop(activations)) {
                // TODO: backpropagation computation for current microbatch
                layer_.setZValues(zValuesContainer.back());
                zValuesContainer.pop_back();

                activations = layer_.calc_back_prop(activations);
                // TODO: push result to the prevDeltaW
                prevDeltaW.push(activations);
            }
        }
    }
};




#endif //DEEPDENDRO_WORKER_H
