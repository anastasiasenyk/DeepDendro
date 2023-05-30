//
// Created by Yaroslav Korch on 30.05.2023.
//

#ifndef DEEPDENDRO_FLATTENINGLAYER_H
#define DEEPDENDRO_FLATTENINGLAYER_H

#include <unsupported/Eigen/CXX11/Tensor>

template<size_t Dimension>
class FlatteningLayerBase {
    using Shape = Eigen::array<Eigen::Index, Dimension>;
protected:
    Shape before_flattening_shape;
public:
    Eigen::VectorXd flatten(const Eigen::Tensor<double, Dimension> &tensor);


    Eigen::Tensor<double, Dimension>
    reshape(const Eigen::VectorXd &vec, const Eigen::array<Eigen::Index, Dimension> &shape);

    Eigen::Tensor<double, Dimension>
    back_to_tensor(const Eigen::VectorXd &vec) {
        return reshape(vec, before_flattening_shape);
    }

};

template<size_t Dimension>
Eigen::VectorXd FlatteningLayerBase<Dimension>::flatten(const Eigen::Tensor<double, Dimension> &tensor) {
    before_flattening_shape = tensor.dimensions();

    Eigen::TensorMap<const Eigen::Tensor<double, 1>> flattened_tensor(tensor.data(), tensor.size());
    Eigen::Map<const Eigen::VectorXd> vector(flattened_tensor.data(), tensor.size());
    return vector;
}

template<size_t Dimension>
Eigen::Tensor<double, Dimension> FlatteningLayerBase<Dimension>::reshape(const Eigen::VectorXd &vec,
                                                                         const Eigen::array<Eigen::Index, Dimension> &shape) {
    Eigen::TensorMap<const Eigen::Tensor<double, 1>> tensor_map(vec.data(), vec.size());
    return tensor_map.reshape(shape);
}

class FlatteningLayer2D : public FlatteningLayerBase<2> {
};

class FlatteningLayer3D : public FlatteningLayerBase<3> {
};


#endif //DEEPDENDRO_FLATTENINGLAYER_H
