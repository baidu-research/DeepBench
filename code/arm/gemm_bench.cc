#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>

#include "gemmlowp/public/gemmlowp.h"
#include "gemmlowp/test/test.h"
#include "gemm_problems.h"

template <bool a_t, bool b_t>
double time_gemm(int m, int n, int k) {
    gemmlowp::GemmContext context;
    context.set_max_num_threads(0);
    
    typedef gemmlowp::MapOrder Order;

    static const Order LhsOrder = a_t ? Order::RowMajor : Order::ColMajor;
    static const Order RhsOrder = b_t ? Order::RowMajor : Order::ColMajor;

    gemmlowp::Matrix<std::uint8_t, LhsOrder> lhs(m, k);
    gemmlowp::Matrix<std::uint8_t, RhsOrder> rhs(k, n);
    gemmlowp::Matrix<std::uint8_t, Order::ColMajor> result(m, n);

    gemmlowp::MakeRandom<typename gemmlowp::OperandRange<0, 255>>(&lhs);
    gemmlowp::MakeRandom<typename gemmlowp::OperandRange<0, 255>>(&rhs);

/** Configuration for low precision shifted matrix multiplication.
 *
 *  The mathematical expression to be computed is the result of the following steps:
 *  1. Cast lhs entries from uint8 to int32 and add lhs_offset to each of them.
 *  2. Cast rhs entries from uint8 to int32 and add rhs_offset to each of them.
 *  3. Compute the int32 matrix product of the resulting lhs times rhs.
 *  4. Add res_offset to each entry of the result.
 *  5. Multiply each entry of the result by the following fraction, and round
 *     to the nearest integer:
 *
 *                         res_mul
 *                       -----------
 *                       2^res_shift
 *
 *  6. Clamp the resulting int32 values to the [0..255] range and cast to uint8.
 *
 *  To summarize:
 *
 *        res_mul
 *  B = ----------- ((A + lhs_offset) * (X + rhs_offset) + res_offset)
 *      2^res_shift
 *
 *  By estimating or observing the range of values of the entries in A, X, and
 *  B matrices, you can determine min_a, max_a, min_x, max_x, min_b, max_b,
 *  which are the minimum/maximum representable float value for your uint8_t
 *  representation for entries in A, X, and B, respectively.
 *
 *  Then the parameters are determined as follows:
 *
 *                  min_a * 256
 *  lhs_offset = ------------------
 *                  max_a - min_a
 *
 *                  min_x * 256
 *  rhs_offset = ------------------
 *                  max_x - min_x
 *
 *                     - min_b * 256 * 256
 *  res_offset = -----------------------------------
 *                (max_a - min_a) * (max_x - min_x)
 *
 *    res_mul     (max_a - min_a) * (max_x - min_x)
 *  ----------- = ---------------------------------
 *  2^res_shift        (max_b - min_b) * 256
 *
 *  The parameters used below correpsonds to:
 *
 *   min_a = -1,  max_a = 1
 *   min_x = 0,   max_x = 16
 *   min_b = -16, max_b = 16
 *
 *  which are tuned to work for our GEMM application.
 *  You should determine your own maximum/minimum that work for your case.
 */

    int lhs_offset = -128;
    int rhs_offset = 0;
    int res_offset = 32768;
    int res_mul = 1;
    int res_shift = 8;

    // warm up
    gemmlowp::Gemm<uint8_t, gemmlowp::DefaultL8R8BitDepthParams>(
        &context,
        lhs.const_map(),
        rhs.const_map(),
        &result,
        lhs_offset,
        rhs_offset,
        res_offset,
        res_mul,
        res_shift);

    int numRepeats = std::max(std::ceil(1e10 / (m * k * n)), 10.);
    auto start = std::chrono::steady_clock::now();

    for (int i = 0; i < numRepeats; ++i) {
        gemmlowp::Gemm<uint8_t, gemmlowp::DefaultL8R8BitDepthParams>(
            &context,
            lhs.const_map(),
            rhs.const_map(),
            &result,
            lhs_offset,
            rhs_offset,
            res_offset,
            res_mul,
            res_shift);
    }
     
    auto end = std::chrono::steady_clock::now();
 
    return std::chrono::duration<double, std::milli>(end - start).count() / numRepeats;
}

double time_gemm_helper(int m, int n, int k, bool a_t, bool b_t) {
#define HANDLE_MATRIX_ORDER(ta, tb)            \
    if (a_t == ta && b_t == tb) {              \
        return time_gemm<ta, tb>(m, n, k);     \
    }

    HANDLE_MATRIX_ORDER(false, false)
    HANDLE_MATRIX_ORDER(false, true)
    HANDLE_MATRIX_ORDER(true, false)
    HANDLE_MATRIX_ORDER(true, true)

#undef HANDLE_MATRIX_ORDER
}

int main(int argc, char** argv) {
    std::cout << std::setw(30) << "Times" << std::endl;
    std::cout << std::setfill('-') << std::setw(88) << "-" << std::endl;
    std::cout << std::setfill(' ');
    std::cout << "    m       n      k      a_t    b_t    time (msec)     GOPS " << std::endl;

    for (const auto &problem : inference_device_set) {
    	int m, n, k;
    	bool a_t, b_t;
        std::tie(m, n, k, a_t, b_t) = problem;

        double time = time_gemm_helper(m, n, k, a_t, b_t);
        double mops = 1e-6 * 2 * m * n * k / time; 

        std::cout << std::setw(7) << m;
        std::cout << std::setw(7) << n;
        std::cout << std::setw(7) << k;
        std::cout << std::setw(7) << (a_t ? "t" : "n");
        std::cout << std::setw(7) << (b_t ? "t" : "n");
        std::cout << std::setw(13) << std::setprecision(6) << time;
        std::cout << std::setw(13) << std::setprecision(6) << mops; 
        std::cout << std::endl;
    }

    return 0;
}
    
		
