//
// Created by Yaroslav Korch on 30.03.2023.
//

#ifndef DEEPDENDRO_LAYER_H
#define DEEPDENDRO_LAYER_H


#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;

#include "iostream"
#include <memory>
#include <utility>
#include <vector>


class Layer : public std::enable_shared_from_this<Layer> {
private:
    std::pair<size_t, size_t> shape;
    std::vector<std::shared_ptr<Layer>> parent_layers_;
    std::vector<std::shared_ptr<Layer>> child_layers_;

public:
    Layer(size_t shape_x, size_t shape_y) : shape(std::make_pair(shape_x, shape_y)) {}

    std::vector<std::shared_ptr<Layer>> get_parents() {
        return parent_layers_;
    }

    std::vector<std::shared_ptr<Layer>> get_children() {
        return child_layers_;
    }

    Layer add_child(const std::shared_ptr<Layer> &child) {
        child_layers_.push_back(child);
        return *this;
    }

    Layer operator()(const std::shared_ptr<Layer> &parent) {
        parent_layers_.push_back(parent);
        parent->add_child(shared_from_this());
        return *this;
    }

    void print_structure_TEMP() {
        std::cout << "PARENT: ";
        for (const auto &el: parent_layers_) {
            std::cout << "(" << el->shape.first << ", " << el->shape.second << "), ";
        }
        std::cout << std::endl;

        std::cout << "CHILD: ";
        for (const auto &el: child_layers_) {
            std::cout << "(" << el->shape.first << ", " << el->shape.second << "), ";
        }
        std::cout << std::endl;
    }

    void forward_prop() {} // TODO
    void back_prop() {} // TODO

};


#endif //DEEPDENDRO_LAYER_H
