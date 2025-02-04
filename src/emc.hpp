#include <vector>
#include <ranges>
#include <fstream>
#include <iostream>

#include "algebra.hpp"
#include "qualgebra.hpp"
#include "models.hpp"
#include "loss_fs.hpp"
#include "readout.hpp"


class emulator{
    public:

        CZladder2p1 czl;
        expectZ exp;

        emulator(CZladder2p1& czl, expectZ& exp) : czl(czl), exp(exp)
        {}
        
        std::vector<std::complex<double>> evaluate(std::vector<double>& theta, Matrix& x, int shots = 0, int rseed = 0)
        {

            std::ofstream file("emc.logs");

            file << "Empieza emulator \n";

            std::vector<std::vector<Matrix>> Ut;
            Matrix probs;
            Vector probs_ib;
            Vector sA;
            std::vector<Vector> rows = get_matrix_rows(x);
            Matrix rhoB = zeros(czl.NM, czl.NM); rhoB.matrix(0,0) = 1.0;
            Matrix rhoB_tm1;
            std::vector<Matrix> Eks;
            std::vector<Matrix> EksD;
            Matrix Ek_rhoB_EkD;

            file << "Declaramos variables \n";
            
            Ut = czl.evolve(theta);

            file << "Definimos Ut \n";
            

            

            for (int i = 0; i < rows.size(); i++){
                Eks.clear();
                for (int i = 0; i < czl.NE; i++) {
                    Eks.push_back(zeros(czl.NM, czl.NM));
                    
                }
                
                file << "Acaba el for \n";

                sA = czl.encoder.encode(rows[i], theta);
                rhoB_tm1 = rhoB;
                
                //Segmentation fault aqui
                for (int j = 0; j < sA.vector.size(); j++){
                    file << sA.vector.size() << "\n";
                    file.close();
                    for (int k = 0; k < Ut[j].size(); k++){
                        
                        Ut[j][k].matrix *= sA.vector[j];
                    }
                    
                    for (int k = 0; k < Ut[j].size(); k++){
                        Matrix aux_matrix;
                        aux_matrix.matrix = Ut[j][k].matrix;
                        Eks.push_back(aux_matrix); 
                         
                    }
                    
                    
                    
                }
                
                
                

                for (auto& Eki : Eks){
                    Eki.matrix = (Eki.matrix).adjoint();
                    EksD.push_back(Eki);
                }

                for (int j = 0; j < Eks.size(); j++){
                    Ek_rhoB_EkD = Eks[j]*rhoB_tm1*EksD[j];
                    probs_ib.vector(j) = Ek_rhoB_EkD.matrix.trace().real();
                    rhoB.matrix += Ek_rhoB_EkD.matrix;
                }

                probs.matrix.row(i) = probs_ib.vector;
            }
            

            

            return exp.readout(probs, czl.shots, czl.rseed);
    
        } // End of "evaluate" method

        //TODO
        /* void psr1(std::vector<double>& theta, Matrix& x, int& ish, int shots=0, int rseed=0)
        {
            break;
        } */

        double deriv_BL_fd(mse& mse, std::vector<double>& params, Matrix& xin, std::vector<double>& yin, int& Nwout, int& index, double eps = 1.e-7)
        {
            std::vector<double> params_ps = params; params_ps[index] += eps;
            std::vector<double> sub_params(params_ps.begin() + 1, params_ps.end());
            std::vector<std::complex<double>> evalu;
            std::vector<double> yb;
            double costs_ps;


            evalu = this->evaluate(sub_params, xin);
            for (int j = 0; j < evalu.size(); j++){
                yb.push_back(params_ps[0]);
                yb[j] += evalu[j].real();
            }
        
            costs_ps = mse.loss(yb, yin, Nwout);

            return costs_ps;

        }

        std::vector<std::complex<double>> deriv_fd(std::vector<double>& theta, Matrix& xin, int& index, double eps = 1.e-7)
        {
            std::vector<std::complex<double>> res;
            std::vector<double> theta_ps = theta; theta_ps[index] += eps;
            
            res = this->evaluate(theta_ps, xin);

            return res;

        }

        //TODO: Usa Dask :^o
        /* void grad_fd()
        {
            break;
        } */

        // ¿Esta funcion es necesaria?
        /* void deriv_psr(std::vector<double>& theta, Matrix& x, int& ish, int& shots=0, int& rseed=0)
        {
            void res;

            res = psr1(theta, x, ish);
        } */




};