//
// Created by Yaroslav Korch on 30.03.2023.
//

#ifndef DEEPDENDRO_MODEL_H
#define DEEPDENDRO_MODEL_H

#include <iostream>

#include "Layers.h"
#include "vector"
#include "activationFuncs.h"
#include "lossFunc.h"
#include "logging.h"

#include "Layer.h"

using LayerPtr = std::shared_ptr<Layer>;

class Model {
private:
    std::vector<LayerPtr> inputs;
    std::vector<LayerPtr> outputs;
    std::vector<LayerPtr> layers;

    void reset_input_output();
    std::vector<LayerPtr> get_all_layers();
    std::vector<LayerPtr> toposort();
public:
    Model();

    void save(LayerPtr &in_layer, LayerPtr &out_layer);
    void save(LayerPtr &in_layer,std::vector<LayerPtr> &&out_layers);
    void save(std::vector<LayerPtr> &in_layers, LayerPtr &out_layer);
    void save(std::vector<LayerPtr> &in_layers, std::vector<LayerPtr> &out_layers);

    void forward_prop(); // TODO
    void back_prop(); // TODO

    void compile();
    void train();
};


#endif //DEEPDENDRO_MODEL_H
