#include <iomanip>
#include <memory>
#include <chrono>
#include <vector>
#include <tuple>
#include <iostream>

#include "arm_compute/runtime/NEON/NEFunctions.h"
#include "arm_compute/core/Types.h"

#include "conv_problems.h"

using namespace arm_compute;

int time_cnn(unsigned int w, unsigned int h, unsigned int n, unsigned int c, 
             unsigned int k, unsigned int filter_w, int filter_h,
             unsigned int pad_w, unsigned int pad_h, 
             unsigned int wstride, unsigned int hstride,
             int num_repeats) 
{

    Tensor src;
    Tensor weights0;
    Tensor biases0;
    Tensor out_conv0;

    unsigned int out_w = (w - filter_w + 2*pad_w)/wstride + 1;
    unsigned int out_h = (h - filter_h + 2*pad_h)/hstride + 1;

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
    int fwd_time = static_cast<int>(std::chrono::duration<double, std::micro>(end - start).count() / num_repeats);

    return fwd_time;

}

/** Main program for convolution test
 *
 * @param[in] argc Number of arguments
 * @param[in] argv Arguments ( [optional] Path to PPM image to process )
 */
int main(int argc, const char **argv)
{

    int num_repeats = 50;
    std::cout << std::setw(30) << "Times" << std::endl;
    std::cout << std::setfill('-') << std::setw(115) << "-" << std::endl;
    std::cout << std::setfill(' ');
    std::cout << "   w      h      c      n      k      f_w    f_h  pad_w  pad_h    stride_w  stride_h    precision  fwd_time (usec)  " << std::endl;

    std::cout << std::setfill('-') << std::setw(115) << "-" << std::endl;
    std::cout << std::setfill(' ');

    for (const auto &problem : inference_device_set) {

        // Filter parameters
        unsigned int k, c, filter_w, filter_h;

        // Input parameters
        unsigned int n, w, h;

        // Padding
        unsigned int pad_w, pad_h;

        // Stride
        unsigned int wstride, hstride;

        std::tie(w, h, c, n, k, filter_w, filter_h, pad_w, pad_h, wstride, hstride) = problem;

        auto time = time_cnn(w, h, c, n, k, filter_w, filter_h, pad_w, pad_h, wstride, hstride, num_repeats);

        std::cout << std::setw(5) << w;
        std::cout << std::setw(7) << h;
        std::cout << std::setw(7) << c;
        std::cout << std::setw(7) << n;
        std::cout << std::setw(7) << k;
        std::cout << std::setw(7) << filter_w;
        std::cout << std::setw(7) << filter_h;
        std::cout << std::setw(7) << pad_w;
        std::cout << std::setw(8) << pad_h;
        std::cout << std::setw(10) << wstride;
        std::cout << std::setw(10) << hstride;
        std::cout << std::setw(10) << "float";
        std::cout << std::setw(14) << std::setprecision(7) << time << std::endl;;

    }

}
