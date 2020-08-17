/*******************************************************************************
* Copyright 2018 Arm Inc.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#include <assert.h>
#include <getopt.h>
#include <stdlib.h>

#include <iomanip>
#include <memory>
#include <chrono>
#include <vector>
#include <tuple>
#include <iostream>

#include "arm_compute/runtime/NEON/NEFunctions.h"
#include "arm_compute/runtime/Scheduler.h"
#include "arm_compute/core/Types.h"

#include "gemm_problems.h"

using namespace arm_compute;

enum Precision {
        u8,
        f32
};

double time_gemm(int m, int n, int k, bool a_t, bool b_t,
                 int num_repeats, Precision precision, int num_threads) {
    // ArmCL does not support explicit transpose like ArmPL or Gemmlowp
    assert(!a_t && !b_t);

    // setup # of threads here
    Scheduler::set(Scheduler::Type::CPP);
    Scheduler::get().set_num_threads(num_threads);

    // Setup matrices and gemm kernels
    Tensor a, b, c;
    TensorShape a_shape, b_shape, c_shape;

    a_shape = TensorShape(k, m);
    b_shape = TensorShape(n, k);
    c_shape = TensorShape(n, m);

    if (precision == Precision::f32) {
        a.allocator()->init(TensorInfo(a_shape, 1, DataType::F32));
        b.allocator()->init(TensorInfo(b_shape, 1, DataType::F32));
        c.allocator()->init(TensorInfo(c_shape, 1, DataType::F32));
    } else if (precision == Precision::u8) {
        a.allocator()->init(TensorInfo(a_shape, 1, DataType::U8));
        b.allocator()->init(TensorInfo(b_shape, 1, DataType::U8));
        c.allocator()->init(TensorInfo(c_shape, 1, DataType::U32));
    } else {
        std::cerr << "Undefined precision" << std::endl;
        return 0.0;
    }

    NEGEMM sgemm;
    NEGEMMLowpAssemblyMatrixMultiplyCore igemm;  

    // Configure our kernels according to precision
    if (precision == Precision::f32) {
        sgemm.configure(&a, &b, nullptr, &c, 1.0, 1.0);
    } else if (precision == Precision::u8) {
        igemm.configure(&a, &b, &c);
    }

    a.allocator()->allocate();
    b.allocator()->allocate();
    c.allocator()->allocate();

    // Fill our test matrices according to specified precision
    if (precision == Precision::f32) {
        float *a_buf = (float*)a.buffer();
        float *b_buf = (float*)b.buffer();
        float *c_buf = (float*)c.buffer();

        for (int i = 0; i < a.info()->total_size() / sizeof(float); i++) {
            a_buf[i] = (float) drand48();
        }
        for (int i = 0; i < b.info()->total_size() / sizeof(float); i++) {
            b_buf[i] = (float) drand48();
        }
        for (int i = 0; i < c.info()->total_size() / sizeof(float); i++) {
            c_buf[i] = (float) drand48();
        }
    } else if ( precision == Precision::u8) {
        uint8_t *a_buf = a.buffer();
        uint8_t *b_buf = b.buffer();
        int32_t *c_buf = (int32_t*)c.buffer();
        for (int i = 0; i < a.info()->total_size() / sizeof(uint8_t); i++) {
            a_buf[i] = (uint8_t) rand();
        }
        for (int i = 0; i < b.info()->total_size() / sizeof(uint8_t); i++) {
            b_buf[i] = (uint8_t) rand();
        }
        for (int i = 0; i < c.info()->total_size() / sizeof(int32_t); i++) {
            c_buf[i] = (int32_t) rand();
        }
    }

    // do warm up
    if (precision == Precision::f32) {
        sgemm.run();
    } else if (precision == Precision::u8) {
        igemm.run();
    }

    // measure performance of xGEMM operation
    auto start = std::chrono::steady_clock::now();
    if (precision == Precision::f32 ) {
        for (int i = 0; i < num_repeats; i++) {
            sgemm.run();
        }
    } else if (precision == Precision::u8) {
        for (int i = 0; i < num_repeats; i++) {
            igemm.run();
        }
    }
    auto end = std::chrono::steady_clock::now();

    return std::chrono::duration<double, std::micro>(end - start).count() / num_repeats;
}

static void print_usage()
{
    printf(
        "Usage: <executable> [OPTIONS]\n"
        "\n"
        "Precision control:\n"
        "   --f32               32-bit floating point (default)\n"
        "   --u8                8-bit integers (v8.2a CPUs)\n"
        "Problem set control:\n"
        "   --training          Training data set (default)\n"
        "   --inference         Server inference data set\n"
        "   --device            Device inference data set\n"
        "Custom convolution definition:\n"
        "   --m                 Num rows Matrix A\n"
        "   --n                 Num columns Matrix B\n"
        "   --k                 Num columns Matrix A, rows Matrix B\n"
        "   --ta                Transpose Matrix A\n"
        "   --tb                Transpose Matrix B\n"
        "   --repeat            Number of times to test convolution (default: 10)\n"
        "   --precision         f32, u8 (default: f32)\n"
        "   --num-threads       Number of threads to spread the gemm across (default: 0 (all cpus))\n"
        "\n"
    );
    exit(-1);
}

/** Main program for xGEMM test
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments
 */
int main(int argc, char **argv)
{
    int num_repeats = 10;
    std::vector<std::tuple<int, int, int, bool, bool>> *problems = nullptr;
    int m, n, k;
    m = 128; n = 128; k = 128;
    bool ta, tb;
    ta = false; tb = false;
    int num_threads = 0; // This causes the system to use all threads
    // Default to doing a normal SGEMM
    Precision precision = Precision::f32;

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
        {"precision", required_argument, 0, 0},
        {"num-threads", required_argument, 0, 0},
        {0, 0, 0, 0}
    };

    int opt;
    do {
        int option_index = 0;
        opt = getopt_long(argc, argv, "", long_options, &option_index);
        switch (opt) {
            case -1:
                break;
            case 0:
                switch (option_index) {
                    case 0:
                        if (problems == nullptr) {
                            problems = &training_set;
                            std::cout << "Running the training benchmark set" << std::endl;
                        }
                        break;
                    case 1:
                        if (problems == nullptr) {
                            problems = &inference_server_set;
                            std::cout << "Running the inference server set" << std::endl;
                        }
                        break;
                    case 2:
                        if (problems == nullptr) {
                            problems = &inference_device_set;
                            std::cout << "Running the inference device set" << std::endl;
                        }
                        break;
                    case 3:
                        num_repeats = std::atoi(optarg);
                        if (num_repeats <= 0) {
                            std::cerr << "Invalid repeat parameter spec'ed" << std::endl;
                            return 0;
                        }
                        break;
                    case 4:
                        m = std::atoi(optarg);
                        if (m <= 0) {
                            std::cerr << "Invalid w parameter spec'ed" << std::endl;
                            return 0;
                        }
                        break;
                    case 5:
                        n = std::atoi(optarg);
                        if (n <= 0) {
                            std::cerr << "Invalid h parameter spec'ed" << std::endl;
                            return 0;
                        }
                        break;
                    case 6:
                        k = std::atoi(optarg);
                        if (k <= 0) {
                            std::cerr << "Invalid c parameter spec'ed" << std::endl;
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
                        if (!strcmp("u8", optarg)) {
                            precision = Precision::u8;
                        } else if (!strcmp("f32", optarg)) {
                            precision = Precision::f32;
                        } else {
                            std::cerr << "Invalid precision defined, must be one of [u8, f32]" << std::endl;
                        }
                        break;
                    case 10:
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
                return 0;
                break;
            default:
                print_usage();
                return 0;
                break;
        }
    } while (opt != -1);

    if (problems == nullptr) {
        problems = new std::vector<std::tuple<int, int, int, bool, bool> >();
        problems->push_back(std::tuple<int, int, int, bool, bool>(m, n, k, ta, tb));
    }

    printf("GEMMOP,TA,TB,M,N,K,USEC,GOP/s\n");
    for (const auto &problem : *problems) {

        // Matrix parameters
        int m, n, k;
        // Transpose
        bool a_t, b_t;
        std::tie(m, n, k, a_t, b_t) = problem;

        double time = time_gemm(m, n, k, a_t, b_t, num_repeats, precision, num_threads);
        double gops = (1e-3 * 2. * m * n * k) / time;
        if (precision == Precision::u8) {
            std::cout << "GEMMLOWP,";
        } else if (precision == Precision::f32) {
            std::cout << "SGEMM,";
        }
        std::cout << (a_t ? "true" : "false") << ",";
        std::cout << (b_t ? "true" : "false") << ",";
        std::cout << m << ",";
        std::cout << n << ",";
        std::cout << k << ",";
        std::cout << time << ",";
        std::cout << gops << std::endl;
    }
}
