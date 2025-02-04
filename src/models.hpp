#pragma once

#include <vector>
#include <memory>
#include <fstream>
#include <iostream>
#include <Eigen/Dense>

#include "base_class.hpp"
#include "algebra.hpp"
#include "qualgebra.hpp"
#include "encoders.hpp"
#include "helpers.hpp"

const Matrix base_matrix =  get_base_matrix();


class CZladder2p1 : public base {
    public:
        encodeP2 encoder;

        CZladder2p1(encodeP2& encoder, int nT, int nE, int nM, int nL, int nx, int shots=0, int rseed=0) : encoder{encoder}, base{nT, nE, nM, nL, nx, shots, rseed}
        {}
        //WORKS
        std::vector<std::vector<Matrix>> evolve(std::vector<double>& theta)
        {
            int preindex = encoder.encode_Nparams();
            int findex = preindex * 2*(this->nE + this->nM)*this->nL;
            std::vector<double> sub_theta;
            std::vector<Vector> m_rows;
            Vector csign;
            //csign.vector << 1.0, 1.0;
            std::vector<std::complex<double>> csign_v = {1.0, 1.0};
            Matrix Ua;
            Ua.matrix << 1.0;
            Matrix Uab;
            Uab.matrix << 1.0;
            std::vector<Vector> Uabcx;
            Matrix InB = Identity(this->NM);
            Matrix Uabcx_M;
            Matrix Uf;
            Matrix Ut;
            std::vector<std::vector<Matrix>> result;
            //Tensor<4> result;

            for (auto _ = 0; _ < (this->nE + this->nM - 1); _++){
                std::vector<std::complex<double>> new_csign_v;
                for (int j = 0; j < 2; j++) {
                    for (int i = 0; i < csign_v.size(); i++) {
                        new_csign_v.push_back(base_matrix.matrix(i%2, j) * csign_v[i]);
                    }
                }
                csign_v = new_csign_v;
            }


            csign.vector.conservativeResize(4*(this->nE + this->nM - 1));

            for (int i = 0; i < this->nE; i++){
                sub_theta.assign(theta.begin() + preindex + 2*i, theta.end() + preindex + 2*i + 2);
                Matrix u2 = U2(sub_theta);
                Uab = matrixKroneckerProduct(Uab, u2);
            }


            

            for (int i = this->nE; i < this->nE + this->nM; i++){
                sub_theta.assign(theta.begin() + preindex + 2*i, theta.end() + preindex + 2*i + 2);
                Matrix u2 = U2(sub_theta);
                Uab = matrixKroneckerProduct(Uab, u2);
            }

            


            m_rows = get_matrix_rows(Uab);
            Uabcx_M.matrix.resize(m_rows[0].vector.size(), m_rows[0].vector.size());




            for (int i = 0; i < m_rows.size(); i++){
                //m_rows[i].vector *= csign.vector(i);

                Uabcx.push_back(m_rows[i]);
                for (int j = 0; j < m_rows[i].vector.size(); j++) {
                    Uabcx_M.matrix(i,j) = m_rows[i].vector(j);
                }

            }



        
            

            
            Ut = Uabcx_M;



            // Loop to apply full operator as many times as nlayers. Parameters are different for each layer

            for (int li = 0; li < this->nL; li++){
                Uab.matrix.resize(1,1);
                Uab.matrix << 1.0;
                Uabcx.clear();
                for (int i = 0; i < this->nE; i++){
                    sub_theta.assign(theta.begin() + preindex + 2*(this->nE + this->nM)*li + 2*i, theta.end() + preindex + 2*(this->nE + this->nM)*li + 2*i + 2);
                    Matrix u2 = U2(sub_theta);
                    Uab = matrixKroneckerProduct(Uab, u2);
                }
                
                

                m_rows = get_matrix_rows(Uab);
                

                for (int i = 0; i < m_rows.size(); i++){
                    m_rows[i].vector *= csign.vector(i);
                    Uabcx.push_back(m_rows[i]);
                    for (int j = 0; j < m_rows[i].vector.size(); j++) {
                        Uabcx_M.matrix(i,j) = m_rows[i].vector(j);
                    }
                }
                
                

                Ut = Uabcx_M * Ut;
                
            }

            

            // Applying Rx rotations over regA before measurement

            Ua.matrix << 1.0;
            for (int i = 0; i < this->nE; i++){
                Matrix rx = Rx(findex + i);
                Ua = matrixKroneckerProduct(Ua, rx);
            } 


            Uf = matrixKroneckerProduct(Ua, InB);


            Ut = Uf * Ut;



            for (int j = 0; j < this->NE; j++){
                Matrix aux_matrix;
                std::vector<Matrix> matrix_vector = {};
                for (int i = 0; i < this->NE; i++){
                    aux_matrix.matrix = Ut.matrix.block((this->NM)*i, (this->NM)*j, (this->NM)-1, (this->NM)-1);
                    matrix_vector.push_back(aux_matrix);
                    
                }
                result.push_back(matrix_vector);
            }

            return result;

        };
    
        int evolve_Nparams()
        {
            return 2*(this->nE + this->nM) * this->nL + this->nE;
        }

};




/* class CZme3 : public base {
    public:
        
        void evolve()
        {

        };

        int evolve_Nparams()
        {
            return 3*(this->nE + this->nM) * this->nL + 3*(this->nE);
        }


}; */


