#pragma once

#include <Eigen/Dense>

#include "base_class.hpp"
#include "algebra.hpp"
#include "qualgebra.hpp"

class encodeP2 : public base {
    public:

        using base::base;
        //WORKS
        Vector encode(Vector& xt, std::vector<double>& theta)
        {
            Matrix encgat;
            Matrix Uin;
            std::vector<double> sub_theta;
            Vector sA;
            sA.vector << 1.0;
            Vector sAj(2);
            int j = 0;



            for(auto& xj : xt.vector) {
                encgat = Ry(xj.real());
                Uin = encgat;
                
                for(int repi = 0; repi < (this->nx); repi++) {
                    sub_theta.assign(theta.begin() + 2*(this->nx)*j + 2*repi, theta.begin() + 2*(this->nx)*j + 2*repi + 2);
                    
                    Uin = encgat*U2(sub_theta)*Uin;
                    sAj.vector << Uin.matrix(0,0), Uin.matrix(1,0);
                    
                    sA = vectorKroneckerProduct(sA, sAj);
                }
                j++;
            }


            return sA;

        };
    
        int encode_Nparams()
        {
            return 2*(this->nE)*(this->nx);
        }

};

/* class encodeP3 : public base {
    public:
        Vector<double> encode()
        {
            Matrix<double> encgat;
            Matrix<double> Uin;
            std::vector<double> sub_theta = {};
            Vector<double> sA;
            sA.vector = {1.0};
            Vector<double> sAj;
            int j = 0;

            for(auto& xj : xt) {
                encgat = Ry(xj);
                Uin = encgat;
                for(int repi = 0; repi < (this->nx); repi++) {
                    .assign(theta.begin() + 3*(this->nx)*j + 3*repi, theta.begin() + 3*(this->nx)*j + 3*repi + 3);

                    Uin = encgat*U3(sub_theta)*Uin;
                    sAj.vector(0) = Uin.matrix(0,0);
                    sAj.vector(1) = Uin.matrix(1,0);
                    sA = vectorKroneckerProduct<>(sA, sAj);
                }
                j++;
            }

            return sA;

        };
    
        int encode_Nparams()
        {
            return 3*(this->nE)*(this->nx);
        }

}; */