#pragma once

#include <Eigen/Dense>

#include "algebra.hpp"


Vector get_flip_vector()
{
    Vector flip_vector(2);
    flip_vector.vector << 1.0, -1.0;
    return flip_vector;
}


Matrix get_base_matrix()
{
    Matrix base_matrix(2,2);
    base_matrix.matrix << 1.0, -1.0, 1.0, -1.0; 
    return base_matrix;
} 
