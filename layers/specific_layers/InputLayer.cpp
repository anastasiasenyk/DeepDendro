//
// Created by Anastasiaa on 03.05.2023.
//

#include "InputLayer.h"

InputLayer::InputLayer(MatrixXd &data) :
    a_values(data),
    Layer({data.rows(), data.cols()}){
}

void InputLayer::parameters_init() {
    if (!get_parents().empty()) {
        throw std::logic_error("An input layer must not have a parent.");
    }
}