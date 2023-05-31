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
#include "MNISTProcess.h"


class PipelineModel {
    int microbatch_size;
    std::vector<std::shared_ptr<FlowLayer>> layers;
    std::shared_ptr<FlowOutputLayer> outputLayer;
    tbb::flow::graph g;
    std::mutex mtx1;
    std::mutex mtx2;
    std::mutex mtx3;
    std::mutex mtx4;
    std::mutex mtx5;
    std::mutex mtx6;
    std::mutex mtx7;
    std::mutex mtx8;
    std::mutex mtx9;

    void forward_propagate(std::pair<MatrixXd, MatrixXd>& microbatch, int layerIndex) {
        layers[layerIndex]->forward_prop(microbatch.first);
    }

    // Backward propagation through a layer
    void backward_propagate(std::pair<MatrixXd, MatrixXd>& gradient, int layerIndex) {
        layers[layerIndex]->back_prop(gradient.first);
    }
public:
    PipelineModel(int micro_batch_size): microbatch_size(micro_batch_size) {
        outputLayer = std::make_shared<FlowOutputLayer>(10, Shape(8, 8), find_activation_func(softmax), 25);
    };

    void run_pipeline(tbb::concurrent_queue<std::pair<MatrixXd, MatrixXd>> &queue) {
        size_t microBatchCounter1 = 0;
        double learning_rate = 0.001;

        FlowLayer flowLayer1(16, {784, 8}, find_activation_func(relu), 25);

        FlowLayer flowLayer2(8, {16, 8}, find_activation_func(relu), 25);

        tbb::flow::function_node<bool, bool> weight_update1( g, tbb::flow::unlimited, [learning_rate, &flowLayer1, this](bool m) -> bool {
            std::lock_guard<std::mutex> lock(mtx7);
//             std::cout << "weight_update1 \n";
            flowLayer1.update_weights(learning_rate);
            return true;
        } );

        tbb::flow::function_node<bool, bool> weight_update2( g, tbb::flow::unlimited, [learning_rate, &flowLayer2, this](bool m) -> bool {
            std::lock_guard<std::mutex> lock(mtx8);
//             std::cout << "weight_update2 \n";
            flowLayer2.update_weights(learning_rate);
            return true;
        } );

        tbb::flow::function_node<bool, bool> weight_update3( g, tbb::flow::unlimited, [learning_rate, this](bool m) -> bool {
            std::lock_guard<std::mutex> lock(mtx9);
//             std::cout << "weight_update3 \n";
            outputLayer->update_weights(learning_rate);
            return true;
        } );


        MicrobatchSourceBody body(queue, microbatch_size);

        tbb::flow::input_node<MatrixXd> input(g,
            [this, &body](tbb::flow_control &fc) -> MatrixXd {
                std::pair<MatrixXd, MatrixXd> microbatch;
                if (body(microbatch)) {
                    if (microbatch.first.cols() == 1 && microbatch.first.rows() == 1 && microbatch.first(0, 0) == -1.0 &&
                            microbatch.second.cols() == 1 && microbatch.second.rows() == 1 && microbatch.second(0, 0) == -1.0) {
                        fc.stop();
                    }else {
                        this->outputLayer->set_labels(microbatch.second);
                    }
                    return microbatch.first;
                }
            });


        tbb::flow::function_node<MatrixXd, MatrixXd> func1( g, tbb::flow::unlimited, [this, &flowLayer1]( MatrixXd m ) -> MatrixXd {
            if (m.cols() == 1 && m.rows() == 1 && m(0, 0) == -1.0) {
                return m; // Return immediately poison pill
            }
            std::lock_guard<std::mutex> lock(mtx1);

//            std::cout << "func1 receiving a_value from input\n";
            flowLayer1.forward_prop(m, true);
            return flowLayer1.getAValue();
        } );



        tbb::flow::function_node<MatrixXd, MatrixXd> func2( g, tbb::flow::unlimited, [this, &flowLayer2]( MatrixXd m ) -> MatrixXd {
            if (m.cols() == 1 && m.rows() == 1 && m(0, 0) == -1.0) {
                return m; // Return immediately poison pill
            }
            std::lock_guard<std::mutex> lock(mtx2);
//            std::cout << "func2 receiving a_value from func1\n";
            flowLayer2.forward_prop(m);
            return flowLayer2.getAValue();
        } );

        tbb::flow::function_node<MatrixXd, MatrixXd> func3( g, tbb::flow::unlimited, [this]( MatrixXd m ) -> MatrixXd {
            if (m.cols() == 1 && m.rows() == 1 && m(0, 0) == -1.0) {
                return m; // Return immediately poison pill
            }
            std::lock_guard<std::mutex> lock(mtx3);
//            std::cout << "func3 receiving a_value from func2\n";
            this->outputLayer->forward_prop(m);
            return outputLayer->getAValue();
        } );

        tbb::flow::function_node<MatrixXd, MatrixXd> back_func3( g, tbb::flow::unlimited, [this]( MatrixXd m ) -> MatrixXd {
            if (m.cols() == 1 && m.rows() == 1 && m(0, 0) == -1.0) {
                return m; // Return immediately poison pill
            }
            std::lock_guard<std::mutex> lock(mtx4);
//            std::cout << "back_prop3 receiving a_value from func3\n";
            MatrixXd grad = this->outputLayer->calc_first_back_prop(m);
            return grad;
        } );

        tbb::flow::function_node<MatrixXd, MatrixXd> back_func2( g, tbb::flow::unlimited, [this, &flowLayer2]( MatrixXd m ) -> MatrixXd {
            if (m.cols() == 1 && m.rows() == 1 && m(0, 0) == -1.0) {
                return m; // Return immediately poison pill
            }
            std::lock_guard<std::mutex> lock(mtx5);
//            std::cout << "back_prop2 receiving a_value from back_prop3\n";
            MatrixXd grad = flowLayer2.back_prop(m);
            return grad;
        } );

        tbb::flow::function_node<MatrixXd, MatrixXd > back_func1( g, tbb::flow::unlimited, [this, &flowLayer1, &microBatchCounter1, &weight_update1]( MatrixXd m ) -> MatrixXd {
            if (m.cols() == 1 && m.rows() == 1 && m(0, 0) == -1.0) {
                return m; // Return immediately poison pill
            }
            std::lock_guard<std::mutex> lock(mtx6);
            MatrixXd grad = flowLayer1.back_prop(m);
            microBatchCounter1++;
            if (microBatchCounter1 == 25) {
//                std::cout << "func1 sending true to weight_update1\n";
                microBatchCounter1 = 0;
                weight_update1.try_put(true);
            }
//            std::cout << "back_prop1 receiving a_value from back_prop2\n";
            return grad;
        } );



        make_edge( input, func1 );
        make_edge( func1, func2 );
        make_edge( func2, func3 );
        make_edge( func3, back_func3 );
        make_edge( back_func3, back_func2 );
        make_edge( back_func2, back_func1 );
        make_edge( weight_update1, weight_update2 );
        make_edge( weight_update2, weight_update3 );
        for (int i = 0; i < 1; ++i) {
            std::cout << "Epoch: " << i << "\n";
            input.activate();
            g.wait_for_all();
            g.reset();
            MNISTProcess mnistProcessTrain = MNISTProcess();
            std::string pathToMNIST = "../MNIST_ORG";
            mnistProcessTrain.enqueueMiniBatches(32, queue, pathToMNIST);
        }
//        input.activate();
//        g.wait_for_all();
    }
};


#endif //DEEPDENDRO_PIPELINEMODEL_H
