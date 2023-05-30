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
    HiddenLayer layer_;
    int worker_id_;
    int batch_size;
    std::vector<MatrixXd> zValuesContainer;
    std::vector<MatrixXd> dzContainer;
    std::vector<MatrixXd> activationsContainer;
public:
    std::atomic<size_t> gradients_pushed_w2 = 0;
    tbb::concurrent_queue<std::pair<MatrixXd, MatrixXd>> activationsQ;
    tbb::concurrent_queue<MatrixXd> deltaWQ;
    Worker(int worker_id, HiddenLayer& layer, int num_microbatches): worker_id_(worker_id), layer_(layer),
    batch_size(num_microbatches){}

    void operator()(concurrent_queue<std::pair<MatrixXd, MatrixXd>>& nextActivations, concurrent_queue<MatrixXd>& prevDeltaW, std::atomic<size_t>& gp,
             int update_after, double learning_rate) {
        int forward_processed = 0;
        int backward_processed = 0;
        int num_updates = 0;
        while(true) {
//            MatrixXd activations(layer_.shape.first, layer_.shape.second);
            std::pair<MatrixXd, MatrixXd> activations;
            MatrixXd gradient;
            if (activationsQ.try_pop(activations)) {
                if (activations.first.cols() == 1 && activations.first.rows() == 1 && activations.first(0, 0) == -2.0 &&
                    activations.second.cols() == 1 && activations.second.rows() == 1 && activations.second(0, 0) == -2.0){
                    // push so-called poison pill to start weights update
                    nextActivations.push(activations);

                    // Start weight updates
                    // TODO: maybe move to different function
                    MatrixXd resWeights = MatrixXd::Zero(layer_.getWeightsShape().first, layer_.getWeightsShape().second);
                    MatrixXd resBiases = MatrixXd::Zero(layer_.getBiasesShape().first, layer_.getBiasesShape().second);
                    int toUpdate = dzContainer.size();
                    for (int i = 0; i < toUpdate; ++i) {
                        // get the collected gradient
                        layer_.setDelta(dzContainer[i]);
//                        dzContainer.pop();
                        layer_.apply_back_prop(learning_rate, activationsContainer[i]);
//                        activationsContainer.pop_back();
                        resWeights += layer_.getWeights();
                        resBiases += layer_.getBiases();
                    }
                    resWeights/=toUpdate;
                    resBiases/=toUpdate;
                    layer_.setWeights(resWeights);
                    layer_.setBiases(resBiases);
                    num_updates++;
                    if(resWeights == MatrixXd::Zero(resWeights.rows(), resWeights.cols())){
                        std::cout << "ZERO WEIGHTS!\n";
                    }
                    if(resBiases == MatrixXd::Zero(resBiases.rows(), resBiases.cols())){
                        std::cout << "ZERO BIASES!\n";
                    }
#ifdef DEBUG
                    std::cout << worker_id_ << ": WEIGHTS UPDATED\n" << std::endl;
#endif
                    gradients_pushed_w2 = 0;
                    dzContainer.clear();
                    activationsContainer.clear();

                }
                else if (forward_processed < batch_size) {


                    layer_.forward_prop(activations.first);
                    zValuesContainer.emplace_back(layer_.getZValues());
                    activationsContainer.emplace_back(activations.first);
                    nextActivations.push(std::make_pair(layer_.getAValues(), activations.second));

#ifdef DEBUG
                    std::cout << worker_id_ << ": forward \n";
#endif
                    forward_processed++;
                }
            }
            else if (backward_processed<batch_size && deltaWQ.try_pop(gradient)) {

                layer_.setZValues(zValuesContainer.front());
                zValuesContainer.erase(zValuesContainer.begin());
                gradient = layer_.calc_back_prop(gradient);
                dzContainer.emplace_back(layer_.getDelta());
                prevDeltaW.push(gradient);
                gp++;
#ifdef DEBUG
                std::cout << worker_id_ << ": backward \n";
#endif
                backward_processed++;
                // TODO: num_updates == 75 - magic number, change!
            } else if (forward_processed==batch_size && backward_processed==batch_size && num_updates == 75) {
#ifdef DEBUG
                std::cout << worker_id_ << ": " << forward_processed << "/" << backward_processed << "\n";
                std::cout << worker_id_ << ": DONE! \n";
#endif
                break;
            }
        }
    }
};




#endif //DEEPDENDRO_WORKER_H
