#pragma once
//#include <armadillo> //No esta en los modulos del qmio
#include <Eigen/Dense>



template<typename T = double>
class Vector{
    public:
        Eigen::Matrix<T, Eigen::Dynamic, 1> vector;

        Vector(int len) : vector(len)
        {}

        double dot(Vector<T>& v)
        {
            double res;
            res.vector = (this->vector).dot(v.vector);
            return res;
        }

};



template<typename T = double>
class Matrix{
    public:
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> matrix;

        Matrix(int n_rows, int n_cols) : matrix(n_rows, n_cols) 
        {}
        
        Matrix<T> operator+(const Matrix<T>& M) 
        {
            Matrix<T> res(M.matrix.rows(), M.matrix.cols());
            res.matrix  = this->matrix + M.matrix;
            return res;
        }

        Matrix<T> operator*(const Matrix<T>& M) 
        {
            Matrix<T> res(this->matrix.rows(), M.matrix.cols());
            res.matrix = this->matrix * M.matrix;
            return res;
        }

        Vector<T> operator*(const Vector<T>& V) 
        {
            Vector<T> res(V.vector.rows());
            res.vector  = this->matrix * V.vector;
            return res;
        }
};


template<typename T = double>
Vector<T> vectorkroneckerProduct(const Vector<T>& v, const Vector<T>& w) {
    int m = v.vector.size();
    int n = w.vector.size();

    // Resultado será un vector de tamaño m * n
    Vector<T> result(m * n);

    // Llenar el vector resultado
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            result(i * n + j) = v(i) * w(j);
        }
    }

    return result;
}

template<typename T = double>
std::vector<Vector<T>> get_matrix_rows(Matrix<T>& M)
{
    int n_rows = probs.matrix.rows();
    std::vector<Vector<T>> rows;
    for (int i = 0; i < n_rows; i++){
        Vector v;
        v.vector = probs.matrix.row(i);
        rows.push_back(v);
    }

    return rows;
}

