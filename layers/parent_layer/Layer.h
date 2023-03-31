//
// Created by Yaroslav Korch on 30.03.2023.
//

#ifndef DEEPDENDRO_LAYER_H
#define DEEPDENDRO_LAYER_H


#include <Eigen/Dense>
using Eigen::MatrixXd;
using Eigen::VectorXd;

typedef std::pair<int, int> Shape;

class Layer {
public:
    Shape shape;
};


#endif //DEEPDENDRO_LAYER_H
