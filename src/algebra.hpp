#pragma once
//#include <armadillo> //No esta en los modulos del qmio
#include <Eigen/Dense>
//#include <unsupported/Eigen/CXX11/Tensor>
#include <type_traits>




class Vector{
    public:
        Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1> vector;

        Vector(int len) : vector{len}
        {}

        Vector() : vector{1}
        {}

        std::complex<double> dot(Vector& v)
        {
            std::complex<double> res;
            res = (this->vector).dot(v.vector);
            return res;
        }

};




class Matrix{
    public:
        Eigen::Matrix<std::complex<double>, Eigen::Dynamic, Eigen::Dynamic> matrix;

        Matrix(int n_rows, int n_cols) : matrix{n_rows, n_cols} 
        {}

        Matrix() : matrix{1,1}
        {}
        
        Matrix operator+(const Matrix& M) 
        {
            Matrix res(M.matrix.rows(), M.matrix.cols());
            res.matrix  = this->matrix + M.matrix;
            return res;
        }

        Matrix operator*(const Matrix& M) 
        {
            Matrix res(this->matrix.rows(), M.matrix.cols());
            res.matrix = this->matrix * M.matrix;
            return res;
        }

        Vector operator*(const Vector& V) 
        {
            Vector res(V.vector.rows());
            res.vector  = this->matrix * V.vector;
            return res;
        }

};

Matrix zeros(int& n_rows, int& n_cols)
{
    Matrix zr(n_rows, n_cols);
    zr.matrix = Eigen::MatrixXd::Zero(n_rows, n_cols);
    return zr;
}

Vector vectorKroneckerProduct(const Vector& v, const Vector& w) {
    int m = v.vector.size();
    int n = w.vector.size();

    // Resultado será un vector de tamaño m * n
    Vector result(m * n);

    // Llenar el vector resultado
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            result.vector(i * n + j) = v.vector(i) * w.vector(j);
        }
    }

    return result;
}


Matrix matrixKroneckerProduct(Matrix& A, Matrix& B) {
    int rowsA = A.matrix.rows(), colsA = A.matrix.cols();
    int rowsB = B.matrix.rows(), colsB = B.matrix.cols();

    Matrix result(rowsA * rowsB, colsA * colsB);

    for (int i = 0; i < rowsA; i++) {
        for (int j = 0; j < colsA; j++) {
            result.matrix.block(i * rowsB, j * colsB, rowsB, colsB) = A.matrix(i, j) * B.matrix;
        }
    }

    return result;
}


std::vector<Vector> get_matrix_rows(Matrix& M)
{
    int n_rows = M.matrix.rows();
    std::vector<Vector> rows;
    for (int i = 0; i < n_rows; i++){
        Vector v;
        v.vector = M.matrix.row(i);
        rows.push_back(v);
    }

    return rows;
}

