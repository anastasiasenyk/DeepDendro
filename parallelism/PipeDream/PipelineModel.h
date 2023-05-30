//
// Created by Matthew Prytula on 20.05.2023.
//

#ifndef DEEPDENDRO_PIPELINEMODEL_H
#define DEEPDENDRO_PIPELINEMODEL_H

#include <tbb/tbb.h>
#include <algorithm>
#include <iostream>
#include "FlowLayer.h"
#include "FlowOutputLayer.h"
#include "SourceNode/MicrobatchSourceBody.h"


class PipelineModel {
    int microbatch_size;
    std::vector<std::shared_ptr<FlowLayer>> layers;
    std::shared_ptr<FlowOutputLayer> outputLayer;
    tbb::flow::graph g;
    void forward_propagate(std::pair<MatrixXd, MatrixXd>& microbatch, int layerIndex) {
        layers[layerIndex]->forward_prop(microbatch.first);
    }

    // Backward propagation through a layer
    void backward_propagate(std::pair<MatrixXd, MatrixXd>& gradient, int layerIndex) {
        layers[layerIndex]->back_prop(gradient.first);
    }
public:
    PipelineModel(int micro_batch_size): microbatch_size(micro_batch_size) {};

    void run_pipeline(tbb::concurrent_queue<std::pair<MatrixXd, MatrixXd>> &queue) {
        size_t microBatchCounter = 0;


        MicrobatchSourceBody body(queue, microbatch_size);

        tbb::flow::input_node<std::pair<MatrixXd, MatrixXd>> input(g,
            [&body, &microBatchCounter](tbb::flow_control &fc) -> std::pair<MatrixXd, MatrixXd> {
            std::cout << "LAUNCHED INPUT NODE" << std::endl;
                std::pair<MatrixXd, MatrixXd> microbatch;
                if (body(microbatch)) {
                    std::cout << "Sending microbatch: " << ++microBatchCounter << std::endl;
                    if (microbatch.first.cols() == 1 && microbatch.first.rows() == 1 && microbatch.first(0, 0) == -1.0 &&
                            microbatch.second.cols() == 1 && microbatch.second.rows() == 1 && microbatch.second(0, 0) == -1.0) {
                        fc.stop();
                    }
                    return microbatch;
                }
            });

        tbb::flow::function_node<std::pair<MatrixXd, MatrixXd>, std::pair<MatrixXd, MatrixXd>> func1( g, tbb::flow::unlimited, []( std::pair<MatrixXd, MatrixXd> m ) -> std::pair<MatrixXd, MatrixXd> {
//            std::cout << m.first << "\n";
            std::cout << "RECEIVED MICROBATCH: " << m.first.cols() << " | " << m.first.rows() << std::endl;
            return m;
        } );

        /* continue defining your network... */

        input.activate();
        make_edge( input, func1 );
        g.wait_for_all();
    }
};


#endif //DEEPDENDRO_PIPELINEMODEL_H
