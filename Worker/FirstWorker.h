//
// Created by Matthew Prytula on 03.05.2023.
//

#ifndef DEEPDENDRO_FIRSTWORKER_H
#define DEEPDENDRO_FIRSTWORKER_H

#include <tbb/concurrent_queue.h>
#include <vector>
#include "HiddenLayer.h"

using namespace tbb;

class FirstWorker {
    HiddenLayer layer_;
    int worker_id_;

    std::vector<MatrixXd> zValuesContainer;
    std::vector<MatrixXd> activationsContainer;
public:
    std::atomic<size_t> gradients_pushed = 0;
    tbb::concurrent_queue<MatrixXd> deltaWQ;

    FirstWorker(int worker_id, HiddenLayer &layer) : worker_id_(worker_id), layer_(layer) {}

    void operator()(int microbatch_size, tbb::concurrent_queue<std::pair<MatrixXd, MatrixXd>> &queue,
                    concurrent_queue<std::pair<MatrixXd, MatrixXd>> &nextActivations, int update_after,
                    double learning_rate) {
        std::pair<MatrixXd, MatrixXd> batch;
        int batches_processed = 0;
        MatrixXd activations(layer_.shape.first, layer_.shape.second);
        while (true) {
            MatrixXd gradient;
            if (queue.try_pop(batch)) {
                // check whether a poison pill
                if (batch.first.cols() == 1 && batch.first.rows() == 1 && batch.first(0, 0) == -1.0 &&
                    batch.second.cols() == 1 && batch.second.rows() == 1 && batch.second(0, 0) == -1.0) {
                    break;
#ifdef DEBUG
                    std::cout << worker_id_ << ": received poison pill\n";
#endif
                } else {
#ifdef DEBUG
                    std::cout << worker_id_ << ": got minibatch\n";
#endif
                    // divide into microbatches, and start the pipeline by
                    // calculating forward prop for the first microbatch
                    // and push its result to nextActivations
                    for (int sample = 0; sample < batch.first.cols(); sample += microbatch_size) {
                        MatrixXd microbatchData = batch.first.block(0, sample, 784,
                                                                    std::min(microbatch_size,
                                                                             static_cast<int>(batch.first.cols()) -
                                                                             sample));
                        MatrixXd microbatchLabels = batch.second.block(0, sample, 10,
                                                                       std::min(microbatch_size,
                                                                                static_cast<int>(batch.second.cols()) -
                                                                                sample));
                        layer_.forward_prop(microbatchData);
                        activationsContainer.emplace_back(microbatchData);
                        zValuesContainer.emplace_back(layer_.getZValues());
                        nextActivations.push(std::make_pair(layer_.getAValues(), microbatchLabels));
#ifdef DEBUG
                        std::cout << worker_id_ << ": pushed microbatch \n";
#endif
                    }
                    batches_processed++;
                    if (batches_processed == update_after) {

                        // wait for pipeline to process all the given batches
                        while (gradients_pushed != update_after * (batch.first.cols() / microbatch_size)) {
//                            std::cout << "Gradients received: " << deltaWQ.unsafe_size() << "\n";
                        }
                        // push so-called poison pill to start weights update
                        nextActivations.push(
                                std::make_pair(MatrixXd::Constant(1, 1, -2.0), MatrixXd::Constant(1, 1, -2.0)));
                        MatrixXd resWeights = MatrixXd::Zero(layer_.getWeightsShape().first,
                                                             layer_.getWeightsShape().second);
                        MatrixXd resBiases = MatrixXd::Zero(layer_.getBiasesShape().first,
                                                            layer_.getBiasesShape().second);
                        for (int i = 0; i < gradients_pushed; ++i) {
                            if (deltaWQ.try_pop(gradient)) {
                                layer_.setZValues(zValuesContainer.back());
                                zValuesContainer.pop_back();
                                gradient = layer_.calc_back_prop(gradient);
                                layer_.apply_back_prop(learning_rate, activationsContainer[i]);
                                resWeights += layer_.getWeights();
                                resBiases += layer_.getBiases();
                            } else {
                                // that means something wrong is up
                                // because all the needed gradients should
                                // already be available
                                std::cout << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n";
                            }
                        }
                        // synchronize weights and biases through number of processed micro-batches
                        resWeights /= gradients_pushed;
                        resBiases /= gradients_pushed;
                        layer_.setWeights(resWeights);
                        layer_.setBiases(resBiases);
                        if (resWeights == MatrixXd::Zero(resWeights.rows(), resWeights.cols())) {
                            std::cout << "ZERO WEIGHTS!\n";
                        }
                        if (resBiases == MatrixXd::Zero(resBiases.rows(), resBiases.cols())) {
                            std::cout << "ZERO BIASES!\n";
                        }

#ifdef DEBUG
                        std::cout << worker_id_ << ": WEIGHTS UPDATED\n" << std::endl;
#endif

                        activationsContainer.clear();


                        batches_processed = 0;
                        gradients_pushed = 0;
                    }
                }
            }
        }
    }


};


#endif //DEEPDENDRO_FIRSTWORKER_H
