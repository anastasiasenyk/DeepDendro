//
// Created by Yaroslav Korch on 30.03.2023.
//

#include "dataProcessing.h"

MatrixXd DataProcessing::flatten(const MatrixXd &data){
    return data.reshaped();
}