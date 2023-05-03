//
// Created by Yaroslav Korch on 30.03.2023.
//

#ifndef DEEPDENDRO_LAYER_H
#define DEEPDENDRO_LAYER_H

#include "iostream"
#include <memory>
#include <utility>
#include <vector>
#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;


class Layer : public std::enable_shared_from_this<Layer> {
private:
    std::vector<long> shape;
    std::vector<std::shared_ptr<Layer>> parent_layers_;
    std::vector<std::shared_ptr<Layer>> child_layers_;

public:
    explicit Layer(long num_neurons) {
        shape.push_back(num_neurons);
    }

    explicit Layer(std::vector<long> layer_shape) {
        shape.insert(shape.end(), layer_shape.begin(), layer_shape.end());
    }

    std::vector<std::shared_ptr<Layer>> get_parents() {
        return parent_layers_;
    }

    std::vector<std::shared_ptr<Layer>> get_children() {
        return child_layers_;
    }

    std::vector<long> get_shape() {
        return shape;
    }

    Layer& add_child(const std::shared_ptr<Layer> &child) {
        child_layers_.push_back(child);
        return *this;
    }

    Layer& operator()(const std::shared_ptr<Layer> &parent) {
        parent_layers_.push_back(parent);
        parent->add_child(shared_from_this());
        return *this;
    }

    virtual void parameters_init(){};
};


#endif //DEEPDENDRO_LAYER_H
