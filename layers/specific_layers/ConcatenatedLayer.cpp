//
// Created by Anastasiaa on 04.05.2023.
//

#include "ConcatenatedLayer.h"

ConcatenatedLayer::ConcatenatedLayer(std::shared_ptr<Layer> layer1, std::shared_ptr<Layer> layer2) :
        Layer(layer1->get_shape()[0]+layer2->get_shape()[0])
        {
        MatrixXd a_layer1 = layer1->getAValues();
        MatrixXd a_layer2 = layer2->getAValues();
        Eigen::MatrixXd mat_concat(a_layer1.rows(), a_layer1.cols() + a_layer2.cols());
        mat_concat.block(0, 0, a_layer1.rows(), a_layer1.cols()) = a_layer1;
        mat_concat.block(0, a_layer1.cols(), a_layer2.rows(), a_layer2.cols()) = a_layer2;

        a_values = mat_concat;

        add_parent(layer1);
        add_parent(layer2);
}


MatrixXd ConcatenatedLayer::getAValues() const {
    return a_values;
}
//
//MatrixXd calc_gradient(){
//
//};
//
//MatrixXd calc_back_prop(const MatrixXd &gradient){
//
//};

//void apply_back_prop(double learning_rate){
//
//};