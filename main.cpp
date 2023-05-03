//
// Created by Yaroslav Korch on 30.03.2023.
//

#include <iostream>

#include "Layer.h"
#include "Model.h"
#include "MNISTProcess.h"


int main() {
    typedef std::shared_ptr<Layer> shared_layer;

    shared_layer first_layer = std::make_shared<Layer>(1, 1);
    shared_layer second_layer = std::make_shared<Layer>(2, 2);
    shared_layer third_layer = std::make_shared<Layer>(3, 3);
    shared_layer output_1 = std::make_shared<Layer>(4, 4);
    shared_layer output_2 = std::make_shared<Layer>(5, 5);

    (*second_layer)(first_layer);
    (*third_layer)(first_layer);
    (*output_1)(third_layer);
    (*output_2)(second_layer);

    Model model = Model();
    model.save(first_layer, {output_1, output_2});

    model.train();

    return 0;
}