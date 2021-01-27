/*******************************************************************************
* Copyright 2016-2017 Intel Corporation
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

#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <assert.h>
#include <getopt.h>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <vector>
#include <cstdint>
#include <sstream>

#ifndef USE_OPENBLAS
#include <mkl.h>
#else
#include <cblas.h>
#endif

#include "gemm_problems.h"

#define FIX_LD(x)   (((((x) + 127)/128)*128) + 16)

#define MKL_MEM_ALIGNMENT (4*1024)

#ifdef IGEMM_S8U8S32
#define A_TYPE MKL_INT8
#define B_TYPE MKL_UINT8
#define C_TYPE MKL_INT32
#ifdef PACKED_API
#error "packed api is not supported for integer GEMM"
#endif
#else
#define A_TYPE float
#define B_TYPE float
#define C_TYPE float
#endif

typedef struct gemm_params {
    bool ta;
    bool tb;
#ifdef USE_OPENBLAS
    CBLAS_TRANSPOSE transa;
    CBLAS_TRANSPOSE transb;
#else
    char transa;
    char transb;
#endif
    int m;
    int n;
    int k;
    int lda;
    int ldb;
    int ldc;
} gemm_params_t;

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
  return;
}

int main(int argc, char *argv[])
{
  int i, j;
  size_t sizea, sizeb, sizec;
  size_t num_gemms;
  size_t max_sizea = 0, max_sizeb = 0, max_sizec = 0;
  int max_m = 0, max_n = 0, max_k = 0;
  A_TYPE  *A, ao = 0, bo = 0;
  B_TYPE  *B;
  C_TYPE  *C, co = 0;
  float  alpha = 1.0, beta = 1.0;
  double flops, total_flops = 0., ave_time, total_time = 0.;
#ifdef PACKED_API
  float *AP, *BP;
#endif
  // DEFAULT settings
  int REPEAT = 10;
  // Default matrix test size if we are doing a single test
  int m, n, k;
  m = 128; n = 128; k = 128;
  bool ta, tb;
  ta = false; tb = false;
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
    {0, 0, 0, 0}
  };

  int c;
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
  } while (c != -1);

  if (p_problem_set == nullptr) {
    p_problem_set = new std::vector<std::tuple<int, int, int, bool, bool> >();
    p_problem_set->push_back(std::tuple<int, int, int, bool, bool>(m, n, k, ta, tb));
  }

  num_gemms = p_problem_set->size();
  gemm_params_t* p_gemm_params = (gemm_params_t*) malloc(num_gemms*sizeof(gemm_params_t));

  i = 0;
  for (const auto &problem : *p_problem_set) {
    std::tie(p_gemm_params[i].m, p_gemm_params[i].n, p_gemm_params[i].k, 
             p_gemm_params[i].ta, p_gemm_params[i].tb) = problem;

    if (p_gemm_params[i].ta) {
      p_gemm_params[i].lda = FIX_LD(p_gemm_params[i].k);
      sizea = p_gemm_params[i].lda * p_gemm_params[i].m;
#ifdef USE_OPENBLAS
      p_gemm_params[i].transa = CblasTrans;
#else
      p_gemm_params[i].transa = 'T';
#endif
    } else {
      p_gemm_params[i].lda = FIX_LD(p_gemm_params[i].m);
      sizea = p_gemm_params[i].lda * p_gemm_params[i].k;
#ifdef USE_OPENBLAS
      p_gemm_params[i].transa = CblasNoTrans;
#else
      p_gemm_params[i].transa = 'N';
#endif
    }

    if (p_gemm_params[i].tb) {
      p_gemm_params[i].ldb = FIX_LD(p_gemm_params[i].n);
      sizeb = p_gemm_params[i].ldb * p_gemm_params[i].k;
#ifdef USE_OPENBLAS
      p_gemm_params[i].transb = CblasTrans;
#else
      p_gemm_params[i].transb = 'T';
#endif
    } else {
      p_gemm_params[i].ldb = FIX_LD(p_gemm_params[i].k);
      sizeb = p_gemm_params[i].ldb * p_gemm_params[i].n;
#ifdef USE_OPENBLAS
      p_gemm_params[i].transb = CblasNoTrans;
#else
      p_gemm_params[i].transb = 'N';
#endif
    }

    p_gemm_params[i].ldc = FIX_LD(p_gemm_params[i].m);
    sizec = p_gemm_params[i].ldc * p_gemm_params[i].n;

    max_sizea = std::max(sizea, max_sizea);
    max_sizeb = std::max(sizeb, max_sizeb);
    max_sizec = std::max(sizec, max_sizec);

    max_m     = std::max(p_gemm_params[i].m, max_m);
    max_n     = std::max(p_gemm_params[i].n, max_n);
    max_k     = std::max(p_gemm_params[i].k, max_k);
    ++i;
  }

  assert(i == num_gemms);

#ifdef USE_OPENBLAS
  A = (A_TYPE*) malloc(sizeof(A_TYPE)*max_sizea);
  B = (B_TYPE*) malloc(sizeof(B_TYPE)*max_sizeb);
  C = (C_TYPE*) malloc(sizeof(C_TYPE)*max_sizec);
#elif defined(PACKED_API)
  AP = sgemm_alloc("A", &max_m, &max_n, &max_k);
  BP = sgemm_alloc("B", &max_m, &max_n, &max_k);
#else
  A = (A_TYPE*) mkl_malloc(sizeof(A_TYPE)*max_sizea, MKL_MEM_ALIGNMENT);
  B = (B_TYPE*) mkl_malloc(sizeof(B_TYPE)*max_sizeb, MKL_MEM_ALIGNMENT);
  C = (C_TYPE*) mkl_malloc(sizeof(C_TYPE)*max_sizec, MKL_MEM_ALIGNMENT);
#endif

#ifdef IGEMM_S8U8S32
  for (i=0; i<max_sizea; ++i) A[i] = 11;
  for (i=0; i<max_sizeb; ++i) B[i] = 22;
  for (i=0; i<max_sizec; ++i) C[i] = 33;
#else
  for (i=0; i<max_sizea; ++i) A[i] = (float) drand48();
  for (i=0; i<max_sizeb; ++i) B[i] = (float) drand48();
  for (i=0; i<max_sizec; ++i) C[i] = (float) drand48();
#endif

  // Define the CSV format
  std::cout << "GEMMOP,TA,TB,M,N,K,USEC,GOP/S" << std::endl;

  for (i=0; i < num_gemms; ++i) {

#ifdef PACKED_API
    sgemm_pack("A", &p_gemm_params[i].transa, &p_gemm_params[i].m, &p_gemm_params[i].n, &p_gemm_params[i].k, &alpha, A, &p_gemm_params[i].lda, AP);
    sgemm_pack("B", &p_gemm_params[i].transb, &p_gemm_params[i].m, &p_gemm_params[i].n, &p_gemm_params[i].k, &alpha, B, &p_gemm_params[i].ldb, BP);
    // warmup
    sgemm_compute("P", "P", &p_gemm_params[i].m, &p_gemm_params[i].n, &p_gemm_params[i].k, 
            AP, &p_gemm_params[i].lda, BP, &p_gemm_params[i].ldb, &beta, C, &p_gemm_params[i].ldc);

    auto st_time = std::chrono::steady_clock::now();
    for (j = 0; j < REPEAT; ++j) {
      sgemm_compute("P", "P", &p_gemm_params[i].m, &p_gemm_params[i].n, &p_gemm_params[i].k, 
          AP, &p_gemm_params[i].lda, BP, &p_gemm_params[i].ldb, &beta, C, &p_gemm_params[i].ldc);
    }
#else
    // warmup
#ifdef IGEMM_S8U8S32
    gemm_s8u8s32(&p_gemm_params[i].transa, &p_gemm_params[i].transb, "F", &p_gemm_params[i].m, &p_gemm_params[i].n, &p_gemm_params[i].k, 
        &alpha, A, &p_gemm_params[i].lda, &ao, B, &p_gemm_params[i].ldb, &bo, &beta, C, &p_gemm_params[i].ldc, &co);
#elif defined(USE_OPENBLAS)
   cblas_sgemm(CblasColMajor, p_gemm_params[i].transa, p_gemm_params[i].transb, 
               p_gemm_params[i].m, p_gemm_params[i].n, p_gemm_params[i].k, alpha, 
	       A, p_gemm_params[i].lda, B, p_gemm_params[i].ldb, beta, C, p_gemm_params[i].ldc);
#else
    sgemm(&p_gemm_params[i].transa, &p_gemm_params[i].transb, &p_gemm_params[i].m, &p_gemm_params[i].n, &p_gemm_params[i].k, 
        &alpha, A, &p_gemm_params[i].lda, B, &p_gemm_params[i].ldb, &beta, C, &p_gemm_params[i].ldc);
#endif
    // time measurements
    auto st_time = std::chrono::steady_clock::now();
    for (j = 0; j < REPEAT; ++j) {
#ifdef IGEMM_S8U8S32
      gemm_s8u8s32(&p_gemm_params[i].transa, &p_gemm_params[i].transb, "F", &p_gemm_params[i].m, &p_gemm_params[i].n, &p_gemm_params[i].k, 
          &alpha, A, &p_gemm_params[i].lda, &ao, B, &p_gemm_params[i].ldb, &bo, &beta, C, &p_gemm_params[i].ldc, &co);
#elif defined(USE_OPENBLAS)
      cblas_sgemm(CblasColMajor, p_gemm_params[i].transa, p_gemm_params[i].transb, 
                  p_gemm_params[i].m, p_gemm_params[i].n, p_gemm_params[i].k, alpha, 
                  A, p_gemm_params[i].lda, B, p_gemm_params[i].ldb, beta, C, p_gemm_params[i].ldc);
#else
      sgemm(&p_gemm_params[i].transa, &p_gemm_params[i].transb, &p_gemm_params[i].m, &p_gemm_params[i].n, &p_gemm_params[i].k, 
          &alpha, A, &p_gemm_params[i].lda, B, &p_gemm_params[i].ldb, &beta, C, &p_gemm_params[i].ldc);
#endif
    }
#endif
    auto end_time = std::chrono::steady_clock::now();

    flops = 2.*p_gemm_params[i].m*p_gemm_params[i].n*p_gemm_params[i].k;
    total_flops += flops;

    ave_time     = std::chrono::duration<double, std::micro>(end_time - st_time).count() /REPEAT;
    total_time  += ave_time;

#ifdef IGEMM_S8U8S32
    printf("GEMM_S8U8S32,%s,%s,%d,%d,%d,%.1f,%.5f\n",
        p_gemm_params[i].ta ? "true":"false", p_gemm_params[i].tb ? "true":"false", 
        p_gemm_params[i].m, p_gemm_params[i].n, p_gemm_params[i].k,
        ave_time, 1E-3*flops/ave_time);
#else
    printf("SGEMM,%s,%s,%d,%d,%d,%.1f,%.5f\n",
        p_gemm_params[i].ta ? "true":"false", p_gemm_params[i].tb ? "true":"false", 
        p_gemm_params[i].m, p_gemm_params[i].n, p_gemm_params[i].k,
        ave_time, 1E-3*flops/ave_time);
#endif
  }

#ifdef USE_OPENBLAS
  free(A);
  free(B);
  free(C);
#elif defined(PACKED_API)
  sgemm_free(AP);
  sgemm_free(BP);
#else
  mkl_free(A);
  mkl_free(B);
  mkl_free(C);
#endif

#ifdef IGEMM_S8U8S32
  printf("Total time %.1f usec, Overall Performance: %.5f GOp/sec \n", total_time, 1E-3*total_flops/total_time);
#else
  printf("Total time %.1f usec, Overall Performance: %.5f GFlop/sec \n", total_time, 1E-3*total_flops/total_time);
#endif
  return 0;
}
