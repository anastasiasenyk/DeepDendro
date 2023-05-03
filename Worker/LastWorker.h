//
// Created by Matthew Prytula on 03.05.2023.
//

#ifndef DEEPDENDRO_LASTWORKER_H
#define DEEPDENDRO_LASTWORKER_H

#include <tbb/concurrent_queue.h>
#include <vector>
#include "HiddenLayer.h"
#include "lossFunc.h"

using namespace tbb;

class LastWorker {
    HiddenLayer layer_;
    int worker_id_;
    std::vector<MatrixXd> dzContainer;
    std::vector<MatrixXd> activationsContainer;
    int batch_size;

public:
    tbb::concurrent_queue<std::pair<MatrixXd, MatrixXd>> activationsQ;

    LastWorker(int worker_id, HiddenLayer &layer, int num_microbatches) : worker_id_(worker_id), layer_(layer),
                                                                          batch_size(num_microbatches) {}

    void operator()(concurrent_queue<MatrixXd> &prevDeltaW, std::atomic<size_t> &gp, double learning_rate) {
#ifdef DEBUG
        std::cout << worker_id_ << ": Starting.... \n";
#endif
        int processed = 0;
        std::pair<MatrixXd, MatrixXd> activations;
        MatrixXd gradient;
        int numUpdates = 0;
        while (true) {
            // TODO: numUpdates == 75 - magic number, change!
            if (processed == batch_size && numUpdates == 75) {
#ifdef DEBUG
                std::cout << worker_id_ << ": " << processed <<"\n";
                std::cout << worker_id_ << ": DONE! \n";
#endif
                break;
            }
            if (activationsQ.try_pop(activations)) {
                if (activations.first.cols() == 1 && activations.first.rows() == 1 && activations.first(0, 0) == -2.0 &&
                    activations.second.cols() == 1 && activations.second.rows() == 1 &&
                    activations.second(0, 0) == -2.0) {
                    MatrixXd resWeights = MatrixXd::Zero(layer_.getWeightsShape().first, layer_.getWeightsShape().second);
                    MatrixXd resBiases = MatrixXd::Zero(layer_.getBiasesShape().first, layer_.getBiasesShape().second);
                    int toUpdate = dzContainer.size();
                    for (int i = 0; i < toUpdate; ++i) {
                        layer_.setDelta(dzContainer.back());
                        dzContainer.pop_back();
                        layer_.apply_back_prop(learning_rate, activationsContainer[i]);
                        resWeights += layer_.getWeights();
                        resBiases += layer_.getBiases();
                    }
                    resWeights/=toUpdate;
                    resBiases/=toUpdate;
                    numUpdates++;
                    layer_.setWeights(resWeights);
                    layer_.setBiases(resBiases);
                    activationsContainer.clear();
                    std::cout << "UPDATED LAST\n";
                } else if (processed < batch_size) {
                    // it's not a poison pill, so process it
#ifdef DEBUG
                    std::cout << worker_id_ << ": got activations \n";
#endif
                    activationsContainer.emplace_back(activations.first);

                    layer_.forward_prop(activations.first);

                    std::cout << lossFunc().categoryCrossEntropy(layer_.getAValues(), activations.second) << "\n";

                    gradient = layer_.calc_first_back_prop(activations.second);
                    prevDeltaW.push(gradient);
                    gp++;
                    // for weights update
                    dzContainer.emplace_back(layer_.getDelta());
                    processed++;
#ifdef DEBUG
                    std::cout << worker_id_ << ": computed gradient, pushed it \n";

#endif
                }
            }

        }
    }
};


#endif //DEEPDENDRO_LASTWORKER_H
