#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <vector>

#include <cuda.h>
#include <cublas_v2.h>
#include <curand.h>

#include <thrust/device_ptr.h>
#include <thrust/fill.h>

#include "tensor.h"

int time_gemm(Tensor<float> A, Tensor<float> B, Tensor<float> C, bool a_t, bool b_t, cublasHandle_t cublas_handle) {
    const float alpha = 1.f / static_cast<float>(A.dims()[1]);
    const float beta  = 1.f;

    int m = C.dims()[0];
    int k = a_t ? A.dims()[0] : A.dims()[1];
    int n = C.dims()[1];

    int numRepeats = std::max(std::ceil(1e11 / (m * k * n)), 10.);

    // Warm up
    cublasStatus_t stat = cublasSgemm(cublas_handle,
                a_t ? CUBLAS_OP_T : CUBLAS_OP_N,
                b_t ? CUBLAS_OP_T : CUBLAS_OP_N,
                m,
                n,
                k,
                &alpha,
                A.begin(), A.dims()[0],
                B.begin(), B.dims()[0],
                &beta,
                C.begin(), C.dims()[0]);
    if (stat != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error("sgemm failed");
    }

    cudaDeviceSynchronize();

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < numRepeats; ++i) {
        cublasStatus_t stat = cublasSgemm(cublas_handle,
                    a_t ? CUBLAS_OP_T : CUBLAS_OP_N,
                    b_t ? CUBLAS_OP_T : CUBLAS_OP_N,
                    m,
                    n,
                    k,
                    &alpha,
                    A.begin(), A.dims()[0],
                    B.begin(), B.dims()[0],
                    &beta,
                    C.begin(), C.dims()[0]);
        if (stat != CUBLAS_STATUS_SUCCESS) {
            throw std::runtime_error("sgemm failed");
        }
    }
    cudaDeviceSynchronize();

    auto end = std::chrono::steady_clock::now();

    return static_cast<int>(std::chrono::duration<double, std::micro>(end - start).count() / numRepeats);

}

int main(int argc, char **argv) {
    cudaFree(0);


    cublasHandle_t cublas_handle;
    cublasStatus_t status = cublasCreate(&cublas_handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cout << "CUBLAS init failed" << std::endl;
    }

    curandGenerator_t curand_gen;

    curandCreateGenerator(&curand_gen, CURAND_RNG_PSEUDO_DEFAULT);
    curandSetPseudoRandomGeneratorSeed(curand_gen, 123ULL);

    std::vector<std::tuple<int, int, int, bool, bool>> problems  = {
        std::make_tuple(1760, 16, 1760, false, false),
        std::make_tuple(1760, 32, 1760, false, false),
        std::make_tuple(1760, 64, 1760, false, false),
        std::make_tuple(1760, 128, 1760, false, false),
        std::make_tuple(1760, 7000, 1760, false, false),
        std::make_tuple(2048, 16, 2048, false, false),
        std::make_tuple(2048, 32, 2048, false, false),
        std::make_tuple(2048, 64, 2048, false, false),
        std::make_tuple(2048, 128, 2048, false, false),
        std::make_tuple(2048, 7000, 2048, false, false),
        std::make_tuple(2560, 16, 2560, false, false),
        std::make_tuple(2560, 32, 2560, false, false),
        std::make_tuple(2560, 64, 2560, false, false),
        std::make_tuple(2560, 128, 2560, false, false),
        std::make_tuple(2560, 7000, 2560, false, false),
        std::make_tuple(4096, 16, 4096, false, false),
        std::make_tuple(4096, 32, 4096, false, false),
        std::make_tuple(4096, 64, 4096, false, false),
        std::make_tuple(4096, 128, 4096, false, false),
        std::make_tuple(4096, 7000, 4096, false, false),
        std::make_tuple(1760, 16, 1760, true, false),
        std::make_tuple(1760, 32, 1760, true, false),
        std::make_tuple(1760, 64, 1760, true, false),
        std::make_tuple(1760, 128, 1760, true, false),
        std::make_tuple(1760, 7000, 1760, true, false),
        std::make_tuple(2048, 16, 2048, true, false),
        std::make_tuple(2048, 32, 2048, true, false),
        std::make_tuple(2048, 64, 2048, true, false),
        std::make_tuple(2048, 128, 2048, true, false),
        std::make_tuple(2048, 7000, 2048, true, false),
        std::make_tuple(2560, 16, 2560, true, false),
        std::make_tuple(2560, 32, 2560, true, false),
        std::make_tuple(2560, 64, 2560, true, false),
        std::make_tuple(2560, 128, 2560, true, false),
        std::make_tuple(2560, 7000, 2560, true, false),
        std::make_tuple(4096, 16, 4096, true, false),
        std::make_tuple(4096, 32, 4096, true, false),
        std::make_tuple(4096, 64, 4096, true, false),
        std::make_tuple(4096, 128, 4096, true, false),
        std::make_tuple(4096, 7000, 4096, true, false),
        std::make_tuple(1760, 7133, 1760, false, true),
        std::make_tuple(2048, 7133, 2048, false, true),
        std::make_tuple(2560, 7133, 2560, false, true),
        std::make_tuple(4096, 7133, 4096, false, true),
        std::make_tuple(5124, 9124, 1760, false, false),
        std::make_tuple(35, 8457, 1760, false, false),
        std::make_tuple(5124, 9124, 2048, false, false),
        std::make_tuple(35, 8457, 2048, false, false),
        std::make_tuple(5124, 9124, 2560, false, false),
        std::make_tuple(35, 8457, 2560, false, false),
        std::make_tuple(5124, 9124, 4096, false, false),
        std::make_tuple(35, 8457, 4096, false, false),
        std::make_tuple(5124, 9124, 1760, true, false),
        std::make_tuple(35, 8457, 1760, true, false),
        std::make_tuple(5124, 9124, 2048, true, false),
        std::make_tuple(35, 8457, 2048, true, false),
        std::make_tuple(5124, 9124, 2560, true, false),
        std::make_tuple(35, 8457, 2560, true, false),
        std::make_tuple(5124, 9124, 4096, true, false),
        std::make_tuple(35, 8457, 4096, true, false),
        std::make_tuple(7680, 16, 2560, false, false),
        std::make_tuple(7680, 32, 2560, false, false),
        std::make_tuple(7680, 64, 2560, false, false),
        std::make_tuple(7680, 128, 2560, false, false),
        std::make_tuple(7680, 16, 2560, true, false),
        std::make_tuple(7680, 32, 2560, true, false),
        std::make_tuple(7680, 64, 2560, true, false),
        std::make_tuple(7680, 128, 2560, true, false),
        std::make_tuple(3072, 16, 1024, false, false),
        std::make_tuple(3072, 32, 1024, false, false),
        std::make_tuple(3072, 64, 1024, false, false),
        std::make_tuple(3072, 128, 1024, false, false),
        std::make_tuple(3072, 16, 1024, true, false),
        std::make_tuple(3072, 32, 1024, true, false),
        std::make_tuple(3072, 64, 1024, true, false),
        std::make_tuple(3072, 128, 1024, true, false),
        std::make_tuple(3072, 7435, 1024, false, true),
        std::make_tuple(7680, 5481, 2560, false, true)
    };

    std::cout << std::setw(30) << "Times" << std::endl;
    std::cout << std::setfill('-') << std::setw(88) << "-" << std::endl;
    std::cout << std::setfill(' ');
    std::cout << "    m       n      k      a_t     b_t      time (usec) " << std::endl;
    for (const auto &problem : problems) {
        int m, n, k;
        bool a_t, b_t;
        std::tie(m, n, k, a_t, b_t) = problem;

        auto a = rand({a_t ? k : m, a_t ? m : k}, curand_gen);
        auto b = rand({b_t ? n : k, b_t ? k : n}, curand_gen);
        auto c = zeros({m, n});

        std::cout << std::setw(7) << m;
        std::cout << std::setw(7) << n;
        std::cout << std::setw(7) << k;
        std::cout << std::setw(7) << a_t ? "t" : "n";
        std::cout << std::setw(7) << b_t ? "t" : "n";
        std::cout << std::setw(13) << std::setprecision(6) << time_gemm(a, b, c, a_t, b_t, cublas_handle);
        std::cout << std::endl;
    }

    cublasDestroy(cublas_handle);
    curandDestroyGenerator(curand_gen);

    return 0;
}
