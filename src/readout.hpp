#include <vector>

#include "base_class.hpp"
#include "helpers.hpp"
#include "algebra.hpp"
#include "constants.hpp"


class expectZ : public base {
    public:

    std::vector<double> readout(Matrix<>& probs, int shots = 0, int rseed = 0)
    {
        int nT = probs.matrix.rows();
        int NE = probs.matrix.cols();
        std::vector<double> expZ;

        if (shots == 0) {
            this->shots = shots;
        }
        if (!rseed == 0) {
            //TODO: Random seed
            break;
        }

        std::vector<Vector<>> rows = get_matrix_rows(probs);
        
        Vector<> flips;
        for (int i = 0; i < (this->nE) - 1; i++){
            flips = vectorkroneckerProduct(flips, flip_vector);
        }

        for (auto& item : rows) {
            expZ.push_back(flips.dot(item));
        }

        if (shots = 0) {
            return expZ;
        } else {
            break;
        }
        

    }
};