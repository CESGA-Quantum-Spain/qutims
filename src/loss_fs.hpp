#include <cstdlib>
#include <vector>
#include <cmath>
#include <numeric>

 #include "helpers.hpp"



class mse : public base {
    public:
        double loss(std::vector<double> yb, std::vector<double> y, int Nwout)
        {
            double res;
            std::vector<double> v = {}; 

            for (int i = yb.size() - Nwout; i < yb.size(); i++){
                v.push_back(pow(yb[i]-y[i], 2));
            }

            res = (1.0/Nwout) * std::accumulate(v.begin(), v.end(), 0);
            return res;

        }

        double loss_deriv(std::vector<double> yb, std::vector<double> y, std::vector<double> partials, int Nwout) 
        {
            double res;
            std::vector<double> v = {};

            for (int i = yb.size() - Nwout; i < yb.size(); i++){
                v.push_back(2.0 * (yb[i] - y[i]) * partials[i]);
            }

            res = (1.0/Nwout) * std::accumulate(v.begin(), v.end(), 0);
            return res;
        }

};