#include <vector>
#include <Eigen/Dense>
#include <complex>

#include "base_class.hpp"
#include "algebra.hpp"
#include "helpers.hpp"

Vector flip_vector = get_flip_vector();

class expectZ : public base {
    public:

    using base::base;

    std::vector<std::complex<double>> readout(Matrix& probs, int shots = 0, int rseed = 0)
    {
        int nT = probs.matrix.rows();
        int NE = probs.matrix.cols();
        std::vector<std::complex<double>> expZ;

        if (shots == 0) {
            this->shots = shots;
        }
        /* if (!rseed == 0) {
            //TODO: Random seed
            
        } */

        std::vector<Vector> rows = get_matrix_rows(probs);
        
        Vector flips;
        for (int i = 0; i < (this->nE) - 1; i++){
            flips = vectorKroneckerProduct(flips, flip_vector);
        }

        for (auto& row : rows) {
            expZ.push_back(flips.dot(row));
        }

        if (shots = 0) {
            return expZ;
        } else {
            //TODO
            return expZ;
        }
        

    }
};