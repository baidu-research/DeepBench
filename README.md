![Baidu Logo](/doc/baidu-research-logo-small.png)

# DeepBench

The primary purpose of DeepBench is to benchmark operations that are
important to deep learning on different hardware platforms. Although
the fundamental computations behind deep learning are well understood,
the way they are used in practice can be surprisingly diverse. For
example, a matrix multiplication may be compute-bound,
bandwidth-bound, or occupancy-bound, based on the size of the matrices
being multiplied and the kernel implementation. Because every deep
learning model uses these operations with different parameters, the
optimization space for hardware and software targeting deep learning
is large and underspecified.

DeepBench attempts to answer the question, "Which hardware provides
the best performance on the basic operations used for training deep
neural networks?".  We specify these operations at a low level,
suitable for use in hardware simulators for groups building new
processors targeted at deep learning.

## Where does DeepBench fit in? 

The Deep Learning eco system is consists of several different pieces. 
We wanted to highlight where DeepBench fits into this eco system. 
The diagram below describes the software and hardware components involved with deep learning.
At the very top, deep learning frameworks like Baidu's [PaddlePaddle] (https://github.com/baidu/Paddle), Theano, 
TensorFlow, Torch etc. All these frameworks allow deep learning researchers to build models. They include basic building 
blocks like layers which can be connected in different ways to create a model. In order to train the deep learning models, the frameworks work with underlying neural network libraries such as NVIDIA's cuDNN and Intel's MKL. These libraries implement operations such as matrix multiply that are important to deep learning models. Finally, the models are trained on hardware like NVIDIA GPUs or Intel's Xeon Phi processor.

<img src="doc/deepbench.png" height=300>

DeepBench uses the neural network libraries to benchmark the performance of basic operations on different hardware. It does not work with deep learning frameworks or deep learning models built for applications. We cannot measure the time required to train an entire model using DeepBench. The performance characteristics of models built for different applications are very different from each other. Therefore, we are benchmarking the underlying operations involved in training a deep learning model. Benchmarking these operations will help raise awareness amongst hardware vendors and software developers about the bottlenecks in deep learning training.

## Methodology

DeepBench consists of a set of basic operations (dense matrix
multiplies, convolutions and communication) as well as some recurrent
layer types.  There is an Excel spreadsheet (`DeepBenchKernels.xlsx`)
in this repository that describes all of the sizes.

Both forward and backward operations are tested.  This first version
of the benchmark will focus on training performance in 32-bit
floating-point arithmetic.  Future versions may expand to focus on
inference workloads as well as lower precision arithmetic.

We will use vendor supplied libraries even if faster independent
libraries exist or faster results have been published. Most users will
default to the vendor supplied libraries and as such the vendor
supplied libraries are most representative of users' experience.

## Entry

We are releasing with results for four hardware platforms, NVIDIA's
TitanX, M40, TitanX Pascal and Intel's Knights Landing.  Hardware
vendors or independent users can run the appropriate benchmarks and
input the results into the spreadsheet. We provide an overview of the
results and all results are available in the `results` folder. We will
gladly accept pull requests for new hardware platforms.


# Types of Operations

## Dense Matrix Multiplies

Dense matrix multiplies exist in almost all deep neural networks
today.  They are used to implement fully connected layers and vanilla
RNNs and are building blocks for other types of recurrent layers.
Sometimes they are also used as a quick way to implement novel layer
types for which custom code doesn't exist.

When performing the GEMM operation `A * B = C`, either or both of `A`
and `B` can be optionally transposed. Common terminology to describe a matrix problem 
is the triple (M, N, K), which describes the sizes of the matrices involved, 
and the “op” which tells us which matrices (if any) are transposed. The figure below
describes how the triple (M, N, K) correspond to the sizes of the matrices being multiplied.

<img src="/doc/gemm-diag.png" width="550" />

The variant where both matrices
are transposed is not used in neural networks.  The other three
variants *are* used, but they need not be implemented as a call to
`SGEMM` with those transpose descriptors.  Sometimes it can be faster
to perform an in-place transpose followed by the appropriate
multiplication and a transpose back.  Such optimizations should be
detailed in the spreadsheet.

The constant coefficients alpha and beta should both be 1.0 so that no
work is elided.

## Convolutions

Convolutions make up the vast majority of flops in networks that
operate on images and videos and form important parts of networks such
as speech and natural language modeling making them, perhaps, the
single most important layer from a performance perspective.

Convolutions have 4 or 5 dimensional inputs and outputs giving rise to
a large number of possible orderings for these dimensions.  For the
first version of the benchmark we are only concerned with performance
in NCHW format i.e.  data is presented in image, feature maps, rows
and columns.

There are many techniques for computing convolutions that are optimal
for different sizes of the filter and image, including:  direct, matrix multiply
based, FFT based, and Winograd based approaches.  In the first version
of this benchmark, we are not concerned about the accuracy of the
different approaches since the general consensus is that 32-bit
floating point is accurate *enough* for each of them. We have noted
the approach used for each size in the spreadsheet.

## Recurrent Layers

Recurrent layers are usually made up of some combination of the above
operations and also simpler operations such as unary or binary
operations which aren't very compute intensive and generally are a
small percentage of overall runtime.  However, the GEMM and
convolution operations are relatively small in recurrent layers, 
so the cost of these smaller operations can become significant.  This is especially true if there
is a high fixed overhead associated with starting a computation.  It
is also possible to use alternate storage formats for the recurrent
matrices because the cost of converting to a new storage format can be
amortized over the many steps of the recurrent computation.  If this
is done the time to convert to and from the custom format should be
included in the overall time.

These factors lead to many optimization possibilities both within a
time step and across a sequence of time steps such that measuring the
raw performance of the operations is not necessarily
representative of the performance of an entire recurrent layer.  In
this benchmark we focus on only one recurrent layer, even though there
are even more optimization opportunities if one considers stacks of
them.

The calculation of the inputs should not be included in the time for
the recurrent layer calculation since it can be calculated as one
large multiply and then consumed by the actual recurrent calculation.
So in: h_t = g(Wx_t + Uh_t-1) the time for the calculation of Wx_t for
all t should not be included in the time for the recurrent layer.

The backward calculation should calculate the updates with respect to
the weights but not the inputs.  All the recurrent work is done to
calculate the weight updates, so calculating the updates with respect
to the inputs as well just obscures what we are trying to measure.

The non-linearity for vanilla RNNs should be a ReLU.  The internal
non-linearities of the LSTM should be the standard operations -
sigmoid for the gates and tanh for the activations.  The LSTM should
not have peephole connections.


## All-Reduce

Neural networks are today often trained across multiple GPUs or even
multiple systems each with multiple GPUs.  There are two main categories of techniques for
doing this: synchronous and asynchronous. Synchronous techniques rely
on keeping the parameters on all instances of the model synchronized, usually by making
sure all instances of the model have the same copy of the gradients before taking an
optimization step.  The
[Message Passing Interface (MPI)](https://en.wikipedia.org/wiki/Message_Passing_Interface)
primitive usually used to perform this
operation is called All-Reduce. There are many ways to implement
All-Reduce based on the number of ranks, the size of the data, and the
topology of the network.  This benchmark places no constraints on the
implementation other than that it should be
deterministic. Asynchronous methods are quite varied and in this
version of the benchmark we will not be attempting to test these
methods.

In order to evaluate All-Reduce, we use the following libraries and benchmarks:
* [NVIDIA's NCCL](https://github.com/NVIDIA/nccl)
* [Ohio State University (OSU) Benchmarks](http://mvapich.cse.ohio-state.edu/benchmarks/)

The NCCL library contains a set of standard communication
routines. The library supports any number of GPUs in a single node and
can be run in single process or multi-process (MPI). The NCCL routines
don't support All-Reduce across multiple nodes. In order to evaluate All-Reduce
across multiple nodes, we use the benchmarks from OSU. We report the
shortest latency achieved from all three implementations (NCCL single
process, NCCL MPI, OpenMPI).

#### Topology for NVIDIA 8 GPU System
Each node has two CPU sockets, and each socket has a PCIe root complex.  For each socket there are two PLX switches that are each connected to the CPU socket via 16 lanes of PCIe v3.  There are two GPUs on each PLX switch. All pairs of GPUs communicate simultaneously over 16 lanes of PCIe v3. The two CPU sockets are connected via Intel QPI. The interconnect across nodes is InfiniBand FDR. The figure below shows a schematic diagram of one our nodes, where all devices connected by the same PCI
root complex are encapsulated in a dotted box

![Topology of NVIDIA GPU system with 8 GPUs](/doc/topology-8gpu-system.png)

#### Topology for Intel Xeon Phi and Omni-Path System
The MPI_AllReduce time is measured on Intel Xeon Phi processor 7250 on Intel’s internal Endeavor cluster with Intel® Omni-Path Architecture (Intel® OPA) series 100 fabric with fat-tree topology, using Intel MPI 5.1.3.181.

# Results
In this section, we are documenting the performance for a few operations. 
These are picked at random and are only meant to demonstrate the performance for a few applications.
__The results below only include the time and TeraFLOPS for the fastest processor for the particular operation and parameters. The full results can be found in the `results` folder__. The files in the results folder are:

* `DeepBench_IA_KNL7250.xlsx`: Performance results on Intel's Xeon Phi Processor
* `DeepBench_TitanX.xlsx`: Performance results on NVIDIA's TitanX GPUs
* `DeepBench_M40.xlsx`: Performance results on NVIDIA's M40 GPUs
* `DeepBench_TitanX_Pascal`: Performance results on NVIDIA's TitanX Pascal GPU


The software libraries (e.g. cuDNN, OpenMPI) used to benchmark performance are mentioned in each of Excel workbooks in `Specs` sheet.
Please feel free to ask us any clarifying questions.

Results on more hardware platforms will be added once they are available. We welcome contributions from all hardware vendors.


## GEMM Results

| Kernel                 | A Transpose | B Transpose | Application        | Results (ms) | TeraFLOPS | Processor     |
|------------------------|-------------|-------------|--------------------|--------------|-----------|---------------|
| M=1760, N=128, K=1760  | N           | N           | Speech Recognition | 0.17         | 4.66      | TitanX Pascal |
| M=7860, N=64, K=2560   | N           | N           | Speech Recognition | 0.42         | 5.94      | TitanX Pascal |
| M=2560, N=64, K=2560   | N           | N           | Speech Recognition | 0.28         | 3.02      | TitanX Pascal |
| M=5124, N=9124, K=2560 | T           | N           | Speech Recognition | 55.17        | 4.34      | Xeon Phi 7250 |
| M=3072, N=128, K=1024  | T           | N           | Speech Recognition | 0.28         | 2.87      | TitanX Pascal |

## Convolution Results

| Input Size                      | Filter Size   | # of Filters | Padding (h, w) | Stride (h, w) | Application        | Total Time (ms) | Fwd TeraFLOPS | Processor     |
|---------------------------------|---------------|--------------|----------------|---------------|--------------------|-----------------|---------------|---------------|
| W = 700, H = 161, C = 1, N = 32 | R = 5, S = 20 | 32           | 0, 0           | 2, 2          | Speech Recognition | 2.98            | 6.63          | TitanX Pascal |
| W = 54, H = 54, C = 64, N = 8   | R = 3, S = 3  | 64           | 1, 1           | 1, 1          | Face Recognition   | 0.63            | 10.55         | TitanX Pascal |
| W = 224, H = 224, C = 3, N = 16 | R = 3, S = 3  | 64           | 1, 1           | 1, 1          | Computer Vision    | 3.99            | 3.6           | TitanX Pascal |
| W = 7, H = 7,  C = 512, N = 16  | R = 3, S = 3  | 512          | 1, 1           | 1, 1          | Computer Vision    | 2.93            | 5.88          | TitanX Pascal |
| W = 28, H = 28, C = 192, N = 16 | R = 5, S = 5  | 32           | 2, 2           | 1, 1          | Computer Vision    | 1.57            | 6.59          | TitanX Pascal |

## Recurrent Ops Results

The recurrent op kernels are only run on NVIDIA hardware. The recurrent benchmarks will be run on Intel hardware shortly. 
Results should be available in October.

| Hidden Units | Batch Size | TimeSteps | Recurrent Type | Application         | Total Time | Fwd TeraFLOPS | Processor     |
|--------------|------------|-----------|----------------|---------------------|------------|---------------|---------------|
| 1760         | 64         | 50        | Vanilla        | Speech Recognition  | 8.48       | 1.14          | TitanX Pascal |
| 2560         | 32         | 50        | Vanilla        | Speech Recognition  | 24.69      | 1.69          | TitanX Pascal |
| 1024         | 128        | 25        | LSTM           | Machine Translation | 16.90      | 5.41          | TitanX Pascal |

## All-Reduce Results

We couldn't run the All-Reduce benchmark on NVIDIA's TitanX Pascal GPU since we only had a single Pascal GPU. 

| Size (# of floats) | Number of Processors | Application        | Result (ms) | Bandwidth (GB/s) | Processor      |
|--------------------|----------------------|--------------------|-------------|------------------|----------------|
| 16777216           | 8                    | Speech Recognition | 22.06       | 24.34            | TitanX Maxwell with InfiniBand FDR |
| 16777216           | 16                   | Speech Recognition | 53.76       | 19.97            | Xeon Phi 7250 with Intel® Omni-Path |
| 16777216           | 32                   | Speech Recognition | 55.68       | 38.57            | Xeon Phi 7250 with Intel® Omni-Path |

# Get Involved

We welcome contributions from the community to DeepBench. You can contribute in two ways:

1. Deep Learning Researchers/Engineers: If you are deep learning researcher or engineer working on a new deep learning application, you may have different operations and/or workloads involved in training your model. We are interested in learning more about the underlying operations that are adversely impacting the performance (speed) of your model. Please contribute these operations and workloads!
2. Hardware Vendors: We would gladly accept contributions from other hardware vendors. We're open to accepting benchmark results from large companies or smaller startups building hardware for training deep learning models. Please contribute benchmark results for your hardware!

# Getting the Code
To get the code, simply clone the github repo

```
git clone https://github.com/baidu-research/DeepBench
```

# NVIDIA Benchmarks
## Compiling

In order to build the benchmarks, you will need to specify the following paths:
```
MPI_PATH: Path to MPI library. The benchmarks have been tested with OpenMPI version 1.10.2.
CUDA_PATH: Path to CUDA library. The benchmarks have been tested with version 7.5.18.
CUDNN_PATH: Path to CUDNN library. The benchmarks have been tested with version 5.0.
NCCL_PATH: Path to NCCL library. NCCL library is available at https://github.com/NVIDIA/nccl. The benchmarks have been tested with commit b3a9e1333d9e2e1b8553b5843ba1ba4f7c79739d
```

To build all the benchmarks, please use the following command:
```
cd code/
make CUDA_PATH=<cuda_path> CUDNN_PATH=<cudnn_path> MPI_PATH=<mpi_path> NCCL_PATH=<nccl_path>
```

You need to build the code for the appropriate architecture. By default, the architecture version is set to 5.2. This works for the TitanX and Tesla M40 GPU. In order build the benchmark for another architecture (such as Pascal with version 6.1), please append the following variable to the `make` command:

```
ARCH=sm_61 ## Just an example for Pascal architecture
```

## Running the Benchmarks

Once compilation completes successfully, the executables will be
generated in the `bin` directory. Before executing the benchmarks, it
is important to set your `LD_LIBRARY_PATH` correctly. For bash shells,
please use:

```
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<cuda_path>:<cudnn_path>:<mpi_path>:<nccl_path>
```


The GEMM, convolution and recurrent benchmarks can be run by calling
the respective executables. Here is some of the output from the GEMM 
benchmark:

```
~/DeepBench/code$ bin/gemm_bench
                         Times
----------------------------------------------------------------------------------------
    m       n      k      a_t     b_t      time (usec)
   1760     16   1760      0      0          266
   1760     32   1760      0      0          269
   1760     64   1760      0      0          290
   1760    128   1760      0      0          431
```

To execute the NCCL single All-Reduce benchmark, you need to specify
the number of GPUs as an argument. Please note that the number of GPUs
must not be greater than the number of GPUs visible in your system.

```
bin/nccl_single_all_reduce <num_gpus>
```

The NCCL MPI All-Reduce benchmark can be run using `mpirun` as shown below:

```
mpirun -np <num_ranks> bin/nccl_mpi_all_reduce
```
`num_ranks` cannot be greater than the number of GPUs in the system.

The `osu_allreduce` benchmark can be executed using mpirun as follows:
```
mpirun -np <num_processes> bin/osu_allreduce
```

The `osu_allreduce` benchmark can be run with more processes than
GPUs. However, all our experiments were conducted with each process
running on a single GPU.

# Intel Benchmarks
# Compiling and Running the Benchmarks

Source all the Intel tools (icc, mkl, mpi) into the path

```
source <icc_installdir>/bin/compilervars.sh intel64
source <mkl_installdir>/bin/mklvars.sh intel64
source <impi_installdir>/bin/mpivars.sh intel64
```

Running the Intel GEMM benchmark (MKL 2017)

```
code/intel/sgemm/run_mkl_sgemm_ia.sh
```

Running the Intel convolution benchmark (MKL 2017 and libxsmm (open
source KNL optimized convolution implementation))

```
code/intel/convolution/run_conv_ia.sh
```

Running the Intel All-Reduce benchmark (Uses the standard OSU benchmark
compiled/running with Intel MPI)

```
code/osu_allreduce/run_allreduce_ia.sh <osu_allreduce binary> <hostfile>
```
