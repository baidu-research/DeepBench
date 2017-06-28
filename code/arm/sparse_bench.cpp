#include <algorithm>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <iomanip>
#include <chrono>

#include <Eigen/Eigen>
#include "sparse_gemm_problems.h"

using namespace Eigen;

#ifndef ITERS
#define ITERS 10
#endif

template<typename T, int U = ColMajor>
Matrix<T, Dynamic, Dynamic, U> generate_random_matrix(int rows, int cols) {
    return Matrix<T, Dynamic, Dynamic, U>::Random(rows, cols);
}

template<typename T> 
Matrix<T, Dynamic, 1> generate_random_vector(int entries) {
    return Matrix<T, Dynamic, 1>::Random(entries);
}

template<typename T, int U>
std::tuple <int, int> time_sparse_gemv(const SparseMatrix<T, U>& sp_A, const Matrix<T, Dynamic, Dynamic, U>& A, const Matrix<T, Dynamic, 1>& B, Matrix<T, Dynamic, 1>& C) {

    // Try dense-dense multiplication
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < ITERS; ++i) {
        C = A * B;
        C[0]++;  // dummy instruction to prevent optimizing away prev line
    }
    auto end = std::chrono::steady_clock::now();
    int d_time = static_cast<int>(std::chrono::duration<double, std::micro>(end - start).count() / ITERS);

    // Try sparse-dense multiplication
    start = std::chrono::steady_clock::now();
    for (int i = 0; i < ITERS; ++i) {
        C = sp_A * B;
        C[0]++;  // dummy instruction to prevent optimizing away prev line
    }
    end = std::chrono::steady_clock::now();
    int sp_time = static_cast<int>(std::chrono::duration<double, std::micro>(end - start).count() / ITERS);
    return std::tuple<int, int>(sp_time, d_time);
}

template<typename T, int U, int V, int W>
std::tuple <int, int> time_sparse_gemm(const SparseMatrix<T, U>& sp_A, const Matrix<T, Dynamic, Dynamic, U>& A, const Matrix<T, Dynamic, Dynamic, V>& B, Matrix<T, Dynamic, Dynamic, W>& C) {

    // Try dense-dense multiplication
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < ITERS; ++i) {
        C = A * B;
        C(0,0)++;  // dummy instruction to prevent optimizing away prev line
    }
    auto end = std::chrono::steady_clock::now();
    int d_time = static_cast<int>(std::chrono::duration<double, std::micro>(end - start).count() / ITERS);

    // Try sparse-dense multiplication
    start = std::chrono::steady_clock::now();
    for (int i = 0; i < ITERS; ++i) {
        C = sp_A * B;
        C(0,0)++;  // dummy instruction to prevent optimizing away prev line
    }
    end = std::chrono::steady_clock::now();
    int sp_time = static_cast<int>(std::chrono::duration<double, std::micro>(end - start).count() / ITERS);
    return std::tuple<int, int>(sp_time, d_time);
}


template<typename T>
std::tuple<int, int> time_sparse_bench_helper(int m, int n, int k, float sparsity, std::default_random_engine & e, std::uniform_real_distribution<double> & rng) {
    
    const int U = RowMajor;

    // Note: We've determined empirically that B,C in ColMajor is best for
    // both sparse and dense gemm implementations.
    const int V = ColMajor;
    const int W = ColMajor;

    auto A = generate_random_matrix<T, U>(m, k);
    for (int j = 0; j < k; ++j) {
        for (int i = 0; i < m; ++i) {
            if (rng(e) < sparsity) {
                A(i, j) = 0;
            }
        }
    }
    SparseMatrix<T, U> sp_A = A.sparseView();

    if (n == 1) {
        auto B = generate_random_vector<T>(k);
        auto C = generate_random_vector<T>(m);
        return time_sparse_gemv<T, U>(sp_A, A, B, C);
    }
    else {
        auto B = generate_random_matrix<T, V>(k, n);
        auto C = generate_random_matrix<T, W>(m, n);
        return time_sparse_gemm<T, U, V, W>(sp_A, A, B, C);
    }
}

int main() {

    // Set up RNG
    std::random_device r;
    std::default_random_engine e(r());
    std::uniform_real_distribution<double> rng(0, 1);

    std::cout << std::setw(30) << "Times" << std::endl;
    std::cout << std::setfill('-') << std::setw(110) << "-" << std::endl;
    std::cout << std::setfill(' ');
    std::cout << "    m       n      k      a_t    b_t    sparsity  precision  sparse time (usec) dense time (usec)   speedup " << std::endl;

    std::vector<std::string> types = {"uint8_t", "float"};

    for (const auto &type_name : types) {

        for (const auto &problem : inference_device_set) {

            int m,n,k;
            bool a_t, b_t;
            float sparsity;
            
            std::tie(m, n, k, a_t, b_t, sparsity) = problem;

            std::cout << std::setw(7) << m;
            std::cout << std::setw(7) << n;
            std::cout << std::setw(7) << k;
            std::cout << std::setw(7) << a_t ? "t" : "n";
            std::cout << std::setw(7) << b_t ? "t" : "n";
            std::cout << std::setw(11) << sparsity;
            std::cout << std::setw(12) << type_name;

            int sp_time, d_time;

            if (type_name == "uint8_t") {
                std::tie(sp_time, d_time) = time_sparse_bench_helper<std::uint8_t>(m, n, k, sparsity, e, rng);
            } else if (type_name == "float") {
                std::tie(sp_time, d_time) = time_sparse_bench_helper<float>(m, n, k, sparsity, e, rng);
            } else {
                throw std::runtime_error("Unsupported type_name");
            }

            std::cout << std::setw(15) << sp_time;
            std::cout << std::setw(15) << d_time;
            std::cout << std::setw(20) << float(d_time)/sp_time;
            std::cout << std::endl;
        }
    }

    return 0;
}
