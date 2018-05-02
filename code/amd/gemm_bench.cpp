#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <vector>
#include <cmath>
#include <rocblas.h>

#include "tensor.h"
#include "gemm_problems.h"

int time_gemm(Tensor<float> A, Tensor<float> B, Tensor<float> C, bool a_t, bool b_t, rocblas_handle handle) {
    const float alpha = 1.f / static_cast<float>(A.dims()[1]);
    const float beta  = 1.f;

    int m = C.dims()[0];
    int k = a_t ? A.dims()[0] : A.dims()[1];
    int n = C.dims()[1];

    int numRepeats = std::max(std::ceil(1e11 / (m * k * n)), 10.);

    // Warm up
    rocblas_status stat = rocblas_sgemm(
                		handle,
                		a_t ? rocblas_operation_transpose : rocblas_operation_none,
                		b_t ? rocblas_operation_transpose : rocblas_operation_none,
                		m, n, k,
                		&alpha,
                		A.begin(), A.dims()[0],
                		B.begin(), B.dims()[0],
                		&beta,
                    C.begin(), C.dims()[0] );

    if (stat != rocblas_status_success) {
        throw std::runtime_error("sgemm failed");
    }

    hipDeviceSynchronize();

    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < numRepeats; ++i) {
      rocblas_status stat = rocblas_sgemm(
                  		handle,
                		a_t ? rocblas_operation_transpose : rocblas_operation_none,
                		b_t ? rocblas_operation_transpose : rocblas_operation_none,
                  		m, n, k,
                  		&alpha,
                  		A.begin(), A.dims()[0],
                  		B.begin(), B.dims()[0],
                  		&beta,
                      C.begin(), C.dims()[0] );
        if (stat != rocblas_status_success) {
            throw std::runtime_error("sgemm failed");
        }
    }
    hipDeviceSynchronize();

    auto end = std::chrono::steady_clock::now();

    return static_cast<int>(std::chrono::duration<double, std::micro>(end - start).count() / numRepeats);

}

int main(int argc, char **argv) {
    hipFree(0);
    hipSetDevice(1);
    rocblas_handle handle;
    rocblas_create_handle(&handle);


    std::cout << std::setw(30) << "Times" << std::endl;
    std::cout << std::setfill('-') << std::setw(88) << "-" << std::endl;
    std::cout << std::setfill(' ');
    std::cout << "    m       n      k      a_t     b_t      time (usec) " << std::endl;
    for (const auto &problem : training_set) {
        int m, n, k;
        bool a_t, b_t;
        std::tie(m, n, k, a_t, b_t) = problem;

        auto a = rand<float>({a_t ? k : m, a_t ? m : k});
        auto b = rand<float>({b_t ? n : k, b_t ? k : n});
        auto c = zeros<float>({m, n});

        std::cout << std::setw(7) << m;
        std::cout << std::setw(7) << n;
        std::cout << std::setw(7) << k;
        std::cout << std::setw(7) << (a_t ? "t" : "n");
        std::cout << std::setw(7) << (b_t ? "t" : "n");
        std::cout << std::setw(13) << std::setprecision(6) << time_gemm(a, b, c, a_t, b_t, handle);
        std::cout << std::endl;
    }

    rocblas_destroy_handle(handle);
    return 0;
}

