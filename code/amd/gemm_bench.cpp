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

    rocblas_handle handle;
    rocblas_create_handle(&handle);

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

        auto a = rand({a_t ? k : m, a_t ? m : k});
        auto b = rand({b_t ? n : k, b_t ? k : n});
        auto c = zeros({m, n});

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
