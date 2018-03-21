#include <getopt.h>

#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>
#include <tuple>

#include "gemmlowp/public/gemmlowp.h"
#include "gemmlowp/test/test.h"
#include "gemm_problems.h"

template <bool a_t, bool b_t>
double time_gemm(int m, int n, int k, int numRepeats, int num_threads) {
    gemmlowp::GemmContext context;
    context.set_max_num_threads(num_threads);
    
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

    //int numRepeats = std::max(std::ceil(1e10 / (m * k * n)), 10.);
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
 
    return std::chrono::duration<double, std::micro>(end - start).count() / numRepeats;
}

double time_gemm_helper(int m, int n, int k, bool a_t, bool b_t, 
                        int numRepeats, int num_threads) {
#define HANDLE_MATRIX_ORDER(ta, tb, repeat, num_threads) \
    if (a_t == ta && b_t == tb) {                      \
        return time_gemm<ta, tb>(m, n, k, repeat, num_threads); \
    }

    HANDLE_MATRIX_ORDER(false, false, numRepeats, num_threads)
    HANDLE_MATRIX_ORDER(false, true, numRepeats, num_threads)
    HANDLE_MATRIX_ORDER(true, false, numRepeats, num_threads)
    HANDLE_MATRIX_ORDER(true, true, numRepeats, num_threads)

#undef HANDLE_MATRIX_ORDER
}

void print_usage()
{
  std::cout << "<app> <args>" << std::endl;
  std::cout << std::left << std::setw(30) << "\tARGS" << std::endl;
  std::cout << std::left << std::setw(30) << "\t--training|inference|device" << "\tSelect and run the built in input set" << std::endl;
  std::cout << std::left << std::setw(30) << "\t--m" << "\tNum rows matrix A" << std::endl;
  std::cout << std::left << std::setw(30) << "\t--n" << "\tNum cols matrix B" << std::endl;
  std::cout << std::left << std::setw(30) << "\t--k" << "\tNum cols matrix A, rows Matrix B" << std::endl;
  std::cout << std::left << std::setw(30) << "\t--ta" << "\tTranspose A" << std::endl;
  std::cout << std::left << std::setw(30) << "\t--tb" << "\tTranspose B" << std::endl;
  std::cout << std::left << std::setw(30) << "\t--num-threads" << "\tNumber of threads to spread gemm across (default: 0 (all cpus))" << std::endl;
  return;
}

int main(int argc, char** argv) {
    // DEFAULT settings
    int REPEAT = 10;
    // Default matrix test size if we are doing a single test
    int m, n, k;
    m = 128; n = 128; k = 128;
    bool ta, tb;
    ta = false; tb = false;
    int num_threads = 0;
    std::vector<std::tuple<int, int, int, bool, bool>>* p_problem_set = nullptr;

    // Use getopt_long here to allow for either driving the benchmark using
    // built in tests, or make it a gemm tester
    static struct option long_options[] = {
        {"training", no_argument, 0, 0},  // These will run the full tests and override customization
        {"inference", no_argument, 0, 0},
        {"device", no_argument, 0, 0},
        {"repeat", required_argument, 0, 0},
        {"m", required_argument, 0, 0},
        {"n", required_argument, 0, 0},
        {"k", required_argument, 0, 0},
        {"ta", no_argument, 0, 0},
        {"tb", no_argument, 0, 0},
        {"num-threads", required_argument, 0, 0},
        {0, 0, 0, 0}
    };

    int c = 0;
    do {
        int option_index = 0;
        c = getopt_long(argc, argv, "", long_options, &option_index);
        switch (c) {
            case -1:
                break;
            case 0:
                switch (option_index) {
                    case 0:
                        if (p_problem_set == nullptr) {
                            p_problem_set = &training_set;
                            std::cout << "Running the training benchmark set" << std::endl;
                        }
                        break;
                    case 1:
                        if (p_problem_set == nullptr) {
                            p_problem_set = &inference_server_set;
                            std::cout << "Running the inference server set" << std::endl;
                        }
                        break;
                    case 2:
                        if (p_problem_set == nullptr) {
                            p_problem_set = &inference_device_set;
                            std::cout << "Running the inference device set" << std::endl;
                        }
                        break;
                    case 3:
                        REPEAT = std::atoi(optarg);
                        if (REPEAT <= 0) {
                            std::cerr << "Invalid repeat parameter spec'ed" << std::endl;
                            return 0;
                        }
                        break;
                    case 4:
                        m = std::atoi(optarg);
                        if (m <= 0) {
                            std::cerr << "Invalid m parameter spec'ed" << std::endl;
                            return 0;
                        }
                        break;
                    case 5:
                        n = std::atoi(optarg);
                        if (n <= 0) {
                            std::cerr << "Invalid n parameter spec'ed" << std::endl;
                            return 0;
                        }
                        break;
                    case 6:
                        k = std::atoi(optarg);
                        if (k <= 0) {
                            std::cerr << "Invalid k parameter spec'ed" << std::endl;
                            return 0;
                        }
                        break;
                    case 7:
                        ta = true;
                        break;
                    case 8:
                        tb = true;
                        break;
                    case 9:
                        num_threads = std::atoi(optarg);
                        if (num_threads < 0) {
                            std::cerr << "Invalid number of threads defined, must be >= 0" << std::endl;
                        }
                        break;
                    default:
                        break;
                }
                break;
            case '?':
                print_usage();
                break;
            default:
                print_usage();
                break;
        }
    } while (c != -1);

    if (p_problem_set == nullptr) {
        p_problem_set = new std::vector<std::tuple<int, int, int, bool, bool> >();
        p_problem_set->push_back(std::tuple<int, int, int, bool, bool>(m, n, k, ta, tb));
    }

    std::cout << "GEMMOP,TA,TB,M,N,K,USEC,GOP/s" << std::endl;

    for (const auto &problem : *p_problem_set) {
    	int m, n, k;
    	bool a_t, b_t;
        std::tie(m, n, k, a_t, b_t) = problem;

        double time = time_gemm_helper(m, n, k, a_t, b_t, REPEAT, num_threads);
        double gops = 1e-3 * 2. * m * n * k / time;

        std::cout << "GEMMLOWP,";
        std::cout << (a_t ? "true" : "false") << ",";
        std::cout << (b_t ? "true" : "false") << ",";
        std::cout << m << ",";
        std::cout << n << ",";
        std::cout << k << ",";
        std::cout << time << ",";
        std::cout << gops; 
        std::cout << std::endl;
    }

    return 0;
}
    
		
