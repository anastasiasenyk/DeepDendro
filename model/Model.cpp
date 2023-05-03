//
// Created by Yaroslav Korch on 30.03.2023.
//

#include "Model.h"


#include <unordered_map>
#include <unordered_set>
#include <queue>

Model::Model() : inputs(), outputs() {}

void Model::reset_input_output() {
    inputs.clear();
    outputs.clear();
}

void Model::save(LayerPtr &in_layer, LayerPtr &out_layer) {
    reset_input_output();
    inputs.push_back(in_layer);
    outputs.push_back(out_layer);
}

void Model::save(LayerPtr &in_layer, std::vector<LayerPtr> &&out_layers) {
    reset_input_output();
    inputs.push_back(in_layer);
    outputs.insert(outputs.end(), out_layers.begin(), out_layers.end());
}

void Model::save(std::vector<LayerPtr> &in_layers, LayerPtr &out_layer) {
    reset_input_output();
    inputs.insert(inputs.end(), in_layers.begin(), in_layers.end());
    outputs.push_back(out_layer);
}

void Model::save(std::vector<LayerPtr> &in_layers, std::vector<LayerPtr> &out_layers) {
    reset_input_output();
    inputs.insert(inputs.end(), in_layers.begin(), in_layers.end());
    outputs.insert(outputs.end(), out_layers.begin(), out_layers.end());
}

std::vector<LayerPtr> Model::toposort() {
    std::unordered_map<LayerPtr, size_t> in_degree;
    std::vector<LayerPtr> layers = get_all_layers();

    for (const auto& layer : layers) in_degree[layer] = layer->get_parents().size();

    std::queue<LayerPtr> layer_queue;
    // Add nodes with zero in-degree to the queue
    for (const auto& layer : layers) {
        if (in_degree[layer] == 0) {
            layer_queue.push(layer);
        }
    }

    // Process nodes in the queue and update in-degree of their children
    std::vector<LayerPtr> sorted_layers;

    while (!layer_queue.empty()) {
        const auto current_layer = layer_queue.front();
        layer_queue.pop();
        sorted_layers.push_back(current_layer);
        std::vector<std::shared_ptr<Layer>> childs = current_layer->get_children();
        for (const auto& child : childs) {
            in_degree[child]--;
            if (in_degree[child] == 0) {
                layer_queue.push(child);
            }
        }
    }

    if (sorted_layers.size() != layers.size()) {
        throw std::runtime_error("Graph contains a cycle");
    }

    return sorted_layers;
}

std::vector<LayerPtr> Model::get_all_layers() {
    std::vector<LayerPtr> result;
    std::unordered_set<LayerPtr> visited;
    std::queue<LayerPtr> q;

    // add input layers to queue
    for (const auto& input : inputs) {
        q.push(input);
        visited.insert(input);
    }

    // BFS
    while (!q.empty()) {
        auto layer = q.front();
        q.pop();
        result.push_back(layer);
        for (const auto& child : layer->get_children()) {
            if (!visited.count(child)) {
                q.push(child);
                visited.insert(child);
            }
        }
    }
    return result;
}

void Model::forward_prop() {
}

void Model::back_prop() {
}

void Model::train() {
    std::vector<LayerPtr> layers = toposort();

    for (const LayerPtr& el: layers) {
        el->print_structure_TEMP();
        std::cout << std:: endl;
    }
}