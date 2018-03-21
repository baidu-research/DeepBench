#include <getopt.h>

#include <iomanip>
#include <memory>
#include <chrono>
#include <vector>
#include <tuple>
#include <iostream>

#include "arm_compute/runtime/NEON/NEFunctions.h"
#include "arm_compute/runtime/Scheduler.h"
#include "arm_compute/core/Types.h"

#include "conv_problems.h"

using namespace arm_compute;

struct bench_result {
    double avg_us, avg_gflops;
};

struct bench_result time_cnn(unsigned int w, unsigned int h, unsigned int n, unsigned int c, 
             unsigned int k, unsigned int filter_w, int filter_h,
             unsigned int pad_w, unsigned int pad_h, 
             unsigned int wstride, unsigned int hstride,
             int num_repeats, int num_threads) 
{
    // Setup # of threads to use for convolution
    Scheduler::set(Scheduler::Type::CPP);
    Scheduler::get().set_num_threads(num_threads);

    Tensor src;
    Tensor weights0;
    Tensor biases0;
    Tensor out_conv0;

    unsigned int out_w = (w - filter_w + 2*pad_w)/wstride + 1;
    unsigned int out_h = (h - filter_h + 2*pad_h)/hstride + 1;
    // calculate flops
    double flops = 2.0 * (1.0 * filter_w * filter_h * out_w * out_h) * c * k * n;

    //Functionfor performing ConvolutionLayer
    NEConvolutionLayer conv0;

    // Initialize source image
    const TensorShape src_shape(w, h, c, n);
    src.allocator()->init(TensorInfo(src_shape, 1, DataType::F32));

    // Initialize weights, bias and output tensors
    const TensorShape weights_shape_conv0(static_cast<long unsigned int>(filter_w), static_cast<long unsigned int>(filter_h), static_cast<unsigned int>(src_shape.z()), k);
    const TensorShape biases_shape_conv0(weights_shape_conv0[3]);
    const TensorShape out_shape_conv0(out_w, out_h, weights_shape_conv0[3]);

    weights0.allocator()->init(TensorInfo(weights_shape_conv0, 1, DataType::F32));
    biases0.allocator()->init(TensorInfo(biases_shape_conv0, 1, DataType::F32));
    out_conv0.allocator()->init(TensorInfo(out_shape_conv0, 1, DataType::F32));

    /* [Configure functions] */
    conv0.configure(&src, &weights0, &biases0, &out_conv0, PadStrideInfo(wstride, hstride, pad_w, pad_h));

    /* [Allocate tensors] */

    // Now that the padding requirements are known we can allocate the tensors:
    src.allocator()->allocate();
    weights0.allocator()->allocate();
    biases0.allocator()->allocate();
    out_conv0.allocator()->allocate();

    /* [Execute the functions] */
    conv0.run();

    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < num_repeats; i++) {
        conv0.run();
    }
    auto end = std::chrono::steady_clock::now();
    double fwd_time = std::chrono::duration<double, std::micro>(end - start).count() / (double)num_repeats;

    struct bench_result res = {fwd_time, flops / fwd_time * 1e-3};
    return res;
}

static void usage()
{
    printf(
        "Usage: <executable> [OPTIONS]\n"
        "\n"
        "Precision control:\n"
        "   --f32               32-bit floating point (default)\n"
        "   --u8s8u8            8-bit integers (AVX512VL CPUs)\n"
        "   --s16s16s32         16-bit integers with 32-bit output\n"
        "                       (AVX512_4VNNI CPUs)\n"
        "Problem set control:\n"
        "   --training          Training data set (default)\n"
        "   --inference         Server inference data set\n"
        "   --device            Device inference data set\n"
        "Custom convolution definition:\n"
        "   --w                 Width\n"
        "   --h                 Height\n"
        "   --c                 \n"
        "   --n                 \n"
        "   --k                 \n"
        "   --filter_w          \n"
        "   --filter_h          \n"
        "   --pad_w             \n"
        "   --pad_h             \n"
        "   --wstride           \n"
        "   --hstride           \n"
        "   --repeat            Number of times to test convolution (default: 50)\n"
        "   --num-threads       Number of threads to spread the convolution across (default: 0 (all cpus))\n"
        "\n"
    );
    exit(-1);
}

/** Main program for convolution test
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments ( [optional] Path to PPM image to process )
 */
int main(int argc, char **argv)
{
    int num_repeats = 50;
    int num_threads = 0;
    std::vector<std::tuple<unsigned int, unsigned int, unsigned int, unsigned int,
                       unsigned int, unsigned int, unsigned int,
                       unsigned int, unsigned int, unsigned int, unsigned int> > *problems = nullptr;
    unsigned int w, h, c, n, k, filter_w, filter_h, pad_w, pad_h, wstride, hstride;
    w = 151; h = 40; c = 1; n = 1; k = 32; filter_w = 20; 
    filter_h = 5; pad_w = 8; pad_h = 8; wstride = 8; hstride = 2; 

    // Use getopt_long here to allow for either driving the benchmark using
    // built in tests, or make it a gemm tester
    static struct option long_options[] = {
        {"training", no_argument, 0, 0},  // These will run the full tests and override customization
        {"inference", no_argument, 0, 0},
        {"device", no_argument, 0, 0},
        {"repeat", required_argument, 0, 0},
        {"w", required_argument, 0, 0},
        {"h", required_argument, 0, 0},
        {"c", required_argument, 0, 0},
        {"n", required_argument, 0, 0},
        {"k", required_argument, 0, 0},
        {"filter_w", required_argument, 0, 0},
        {"filter_h", required_argument, 0, 0},
        {"pad_w", required_argument, 0, 0},
        {"pad_h", required_argument, 0, 0},
        {"wstride", required_argument, 0, 0},
        {"hstride", required_argument, 0, 0},
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
                        w = std::atoi(optarg);
                        if (w <= 0) {
                            std::cerr << "Invalid w parameter spec'ed" << std::endl;
                            return 0;
                        }
                        break;
                    case 5:
                        h = std::atoi(optarg);
                        if (h <= 0) {
                            std::cerr << "Invalid h parameter spec'ed" << std::endl;
                            return 0;
                        }
                        break;
                    case 6:
                        c = std::atoi(optarg);
                        if (c <= 0) {
                            std::cerr << "Invalid c parameter spec'ed" << std::endl;
                            return 0;
                        }
                        break;
                    case 7:
                        n = std::atoi(optarg);
                        if (n <= 0) {
                            std::cerr << "Invalid n parameter spec'ed" << std::endl;
                            return 0;
                        }
                        break;
                    case 8:
                        k = std::atoi(optarg);
                        if (k <= 0) {
                            std::cerr << "Invalid k parameter spec'ed" << std::endl;
                            return 0;
                        }
                        break;
                    case 9:
                        filter_w = std::atoi(optarg);
                        if (filter_w <= 0) {
                            std::cerr << "Invalid filter_w paramter spec'ed" << std::endl;
                            return 0;
                        }
                        break;
                    case 10:
                        filter_h = std::atoi(optarg);
                        if (filter_h <= 0) {
                            std::cerr << "Invalid filter_h parameter spec'ed" << std::endl;
                            return 0;
                        }
                        break;
                    case 11:
                        pad_w = std::atoi(optarg);
                        if (pad_w < 0) {
                            std::cerr << "Invalid pad_w parameter spec'ed" << std::endl;
                            return 0;
                        }
                        break;
                    case 12:
                        pad_h = std::atoi(optarg);
                        if (pad_h < 0) {
                            std::cerr << "Invalid pad_h parameter spec'ed" << std::endl;
                            return 0;
                        }
                        break;
                    case 13:
                        wstride = std::atoi(optarg);
                        if (wstride <= 0) {
                            std::cerr << "Invalid wstride parameter spec'ed" << std::endl;
                            return 0;
                        }
                        break;
                    case 14:
                        hstride = std::atoi(optarg);
                        if (hstride <= 0) {
                            std::cerr << "Invalid hstride parameter spec'ed" << std::endl;
                            return 0;
                        }
                        break; 
                    case 15:
                        num_threads = std::atoi(optarg);
                        if (num_threads < 0) {
                            std::cerr << "Invalid # of threads spec'ed" << std::endl;
                            return 0;
                        }
                    default:
                        break;
                }
                break;
            case '?':
                usage();
                return 0;
                break;
            default:
                usage();
                return 0;
                break;
        }
    } while (opt != -1);

    if (problems == nullptr) {
        problems = new std::vector<std::tuple<unsigned int, unsigned int, unsigned int, unsigned int,
                       unsigned int, unsigned int, unsigned int,
                       unsigned int, unsigned int, unsigned int, unsigned int> >();
        problems->push_back(std::tuple<unsigned int, unsigned int, unsigned int, unsigned int,
                       unsigned int, unsigned int, unsigned int,
                       unsigned int, unsigned int, unsigned int, unsigned int>(w, h, c, n, k, filter_w, 
                               filter_h, pad_w, pad_h, wstride, hstride));
    }

    printf("OP,w,h,c,n,k,filter_w,filter_h,pad_w,pad_h,wstride,hstride,usecs,gops\n");
    for (const auto &problem : *problems) {

        // Filter parameters
        unsigned int k, c, filter_w, filter_h;

        // Input parameters
        unsigned int n, w, h;

        // Padding
        unsigned int pad_w, pad_h;

        // Stride
        unsigned int wstride, hstride;

        std::tie(w, h, c, n, k, filter_w, filter_h, pad_w, pad_h, wstride, hstride) = problem;

        auto res = time_cnn(w, h, c, n, k, filter_w, filter_h, pad_w, pad_h, 
                            wstride, hstride, num_repeats, num_threads);

        std::cout << "FWD,";
        std::cout << w << ",";
        std::cout << h << ",";
        std::cout << c << ",";
        std::cout << n << ",";
        std::cout << k << ",";
        std::cout << filter_w << ",";
        std::cout << filter_h << ",";
        std::cout << pad_w << ",";
        std::cout << pad_h << ",";
        std::cout << wstride << ",";
        std::cout << hstride << ",";
        std::cout << res.avg_us << ",";
        std::cout << res.avg_gflops << std::endl;
    }

}
