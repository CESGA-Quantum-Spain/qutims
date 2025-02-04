#pragma once

#include <cmath>
#include <complex>

#include "algebra.hpp"

const std::complex<double> im(0.0, 1.0);

// ------- States -------

Vector zero()
{
    Vector v(2);
    v.vector(0) = 1.0;
    v.vector(1) = 0.0;

    return v;
}

Vector one()
{
    Vector v(2);
    v.vector(0) = 0.0;
    v.vector(1) = 1.0;

    return v;
}

Vector plus()
{
    Vector v(2);
    v.vector(0) = 1.0;
    v.vector(1) = 1.0;
    v.vector = (1.0/sqrt(2.0)) * v.vector;

    return v;
}

Vector minus()
{
    Vector v(2);
    v.vector(0) = 1.0;
    v.vector(1) = -1.0;
    v.vector = (1.0/sqrt(2.0)) * v.vector;

    return v;
}

// ------- Matrices -------

Matrix I()
{
    Matrix M(2,2);
    M.matrix(0,0) = 1.0;
    M.matrix(0,1) = 0.0;
    M.matrix(1,0) = 0.0;
    M.matrix(1,1) = 1.0;

    return M;
}

Matrix H()
{
    Matrix M(2,2);
    M.matrix(0,0) = 1.0;
    M.matrix(0,1) = 1.0;
    M.matrix(1,0) = 1.0;
    M.matrix(1,1) = -1.0;
    M.matrix = (1.0/sqrt(2.0)) * M.matrix;

    return M;
}

Matrix X()
{
    Matrix M(2,2);
    M.matrix(0,0) = 0.0;
    M.matrix(0,1) = 1.0;
    M.matrix(1,0) = 1.0;
    M.matrix(1,1) = 0;
    
    return M;
}

Matrix Rx(double theta)
{
    Matrix M(2,2);
    M.matrix(0,0) = cos(theta/2.0);
    M.matrix(0,1) = std::complex<double>(0.0, sin(theta/2.0));
    M.matrix(1,0) = std::complex<double>(0.0, -sin(theta/2.0));
    M.matrix(1,1) = cos(theta/2.0);
    
    return M;
}

Matrix Ry(double theta)
{
    Matrix M(2,2);
    M.matrix(0,0) = cos(theta/2.0);
    M.matrix(0,1) = sin(theta/2.0);
    M.matrix(1,0) = -sin(theta/2.0);
    M.matrix(1,1) = cos(theta/2.0);
    
    return M;

}

Matrix Rz(double theta)
{
    Matrix M(2,2);
    M.matrix(0,0) = std::complex<double>(cos(theta/2.0), sin(theta/2.0));
    M.matrix(0,1) = 0.0;
    M.matrix(1,0) = 0.0;
    M.matrix(1,1) = std::complex<double>(cos(theta/2.0), sin(theta/2.0));
    
    return M;
}

Matrix U(double theta, double phi, double lambda)
{
    Matrix M(2,2);
    M.matrix(0,0) = cos(theta/2.0);
    M.matrix(0,1) = std::complex<double>(cos(phi), sin(phi) * sin(theta/2.0));
    M.matrix(1,0) = std::complex<double>(0.0, sin(lambda) * sin(theta/2.0));
    M.matrix(1,1) = std::complex<double>(cos(lambda + phi), sin(lambda + phi) * cos(theta/2.0));
    
    return M;
}


Matrix U2(std::vector<double>& sub_thetas)
{
    Matrix res = Rx(sub_thetas[0])*Rz(sub_thetas[1]);
    return res;
}



Matrix U3(std::vector<double>& sub_thetas)
{
    return Rx(sub_thetas[2])*Rz(sub_thetas[1])*Rx(sub_thetas[0]);
}


Matrix hermitian(Matrix& M)
{
    Matrix aM(M.matrix.cols(), M.matrix.rows());

    aM.matrix = aM.matrix.adjoint();
    return aM;
}

Matrix Identity(int& n)
{
    Matrix In(n, n);

    for (int i = 0; i < n; i++) {
        In.matrix(i,i) = 1.0;
    }

    return In;
}