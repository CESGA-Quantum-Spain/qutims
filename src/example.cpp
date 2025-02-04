#include <iostream>
#include <vector>
#include <complex>
#include <fstream>
#include <string>
#include <sstream>  

#include "algebra.hpp"
#include "qualgebra.hpp"
#include "helpers.hpp" 
#include "models.hpp" 
#include "emc.hpp"




std::vector<double> get_theta_from_file(std::string path)
{
    std::vector<double> theta;
    std::ifstream file(path);
    std::string line;
        while (std::getline(file, line)) {  
            std::istringstream iss(line);
            double value;
            
            while (iss >> value) {  
                theta.push_back(value);
            }
        }

    return theta;
}

Matrix get_data_from_file(std::string path)
{

    std::ifstream file(path);  
    std::vector<double> second_column_values;  
    std::string line;

    while (std::getline(file, line)) {  
        std::istringstream iss(line);
        double first_col, second_col, third_col;

        if (iss >> first_col >> second_col >> third_col) { 
            second_column_values.push_back(second_col);  
        } else {
            std::cerr << "Error: Formato incorrecto en la línea -> " << line << "\n";
        }
    }

    // Crear una Eigen::MatrixXd con una sola columna
    Matrix second_column(second_column_values.size(), 1);
    for (size_t i = 0; i < second_column_values.size(); i++) {
        second_column.matrix(i, 0) = second_column_values[i];
    }

    return second_column;
}





int main(){

    int nT=20, nE = 1, nM = 2, nL = 3, nx = 3;
    std::vector<double> theta = get_theta_from_file("theta.dat");
    Matrix x_data = get_data_from_file("matrix.dat");
    std::vector<std::complex<double>> evalu;

    encodeP2 encoder(nT, nE, nM, nL, nx);

    CZladder2p1 czl(encoder, nT, nE, nM, nL, nx);

    expectZ expz(nT, nE, nM, nL, nx);

    emulator emul(czl, expz); 

    evalu = emul.evaluate(theta, x_data);





    return 0;
}