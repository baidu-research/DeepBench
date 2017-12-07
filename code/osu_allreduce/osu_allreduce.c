#define BENCHMARK "OSU MPI%s Allreduce Latency Test"
/*
 * Copyright (C) 2002-2014 the Network-Based Computing Laboratory
 * (NBCL), The Ohio State University.
 *
 * Contact: Dr. D. K. Panda (panda@cse.ohio-state.edu)
 */

/*
This program is available under BSD licensing.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

(1) Redistributions of source code must retain the above copyright
notice, this list of conditions and the following disclaimer.

(2) Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

(3) Neither the name of The Ohio State University nor the names of
their contributors may be used to endorse or promote products derived
from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

*/

#include "osu_coll.h"
#include "all_reduce_problems.h"

#ifdef ENABLE_MLSL
#include "mlsl.h"
#define MLSL_CALL(expression)                                             \
  do                                                                      \
  {                                                                       \
      int ret = expression;                                               \
      if (ret != CMLSL_SUCCESS)                                           \
      {                                                                   \
          printf("%s:%d: MLSL error: ret %d\n", __FILE__, __LINE__, ret); \
          mlsl_environment_finalize(env);                                 \
          exit(EXIT_FAILURE);                                             \
      }                                                                   \
  } while (0)
mlsl_environment env;
mlsl_distribution distribution;
#define FINALIZE()                                                        \
  do {                                                                    \
      MLSL_CALL(mlsl_environment_delete_distribution(env, distribution)); \
      MLSL_CALL(mlsl_environment_finalize(env));                          \
  } while (0)
#else
#define FINALIZE() MPI_Finalize()
#endif

int main(int argc, char *argv[])
{
    int i, j, numprocs, rank, size;
    double latency = 0.0, t_start = 0.0, t_stop = 0.0;
    double timer=0.0;
    double avg_time = 0.0, max_time = 0.0, min_time = 0.0;
    float *sendbuf, *recvbuf;
    int po_ret;
    size_t bufsize;

    int64_t* problems = all_reduce_kernels_size;
    int64_t* numRepeats = all_reduce_kernels_repeat;

    set_header(HEADER);
#ifdef ENABLE_MLSL
    mlsl_comm_req request;
    set_benchmark_name("mlsl_osu_allreduce");
#else
    set_benchmark_name("osu_allreduce");
#endif
    enable_accel_support();
    po_ret = process_options(argc, argv);

    if (po_okay == po_ret && none != options.accel) {
        if (init_accel()) {
            fprintf(stderr, "Error initializing device\n");
            exit(EXIT_FAILURE);
        }
    }

#ifdef ENABLE_MLSL
    MLSL_CALL(mlsl_environment_get_env(&env));
    MLSL_CALL(mlsl_environment_init(env, &argc, &argv));
    size_t process_idx, process_count;
    MLSL_CALL(mlsl_environment_get_process_idx(env, &process_idx));
    MLSL_CALL(mlsl_environment_get_process_count(env, &process_count));
    rank = process_idx;
    numprocs = process_count;
    MLSL_CALL(mlsl_environment_create_distribution(env, process_count, 1, &distribution));
#else
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
#endif

    switch (po_ret) {
        case po_bad_usage:
            print_bad_usage_message(rank);
            FINALIZE();
            exit(EXIT_FAILURE);
        case po_help_message:
            print_help_message(rank);
            FINALIZE();
            exit(EXIT_SUCCESS);
        case po_version_message:
            print_version_message(rank);
            FINALIZE();
            exit(EXIT_SUCCESS);
        case po_okay:
            break;
    }

    if(numprocs < 2) {
        if (rank == 0) {
            fprintf(stderr, "This test requires at least two processes\n");
        }

        FINALIZE();
        exit(EXIT_FAILURE);
    }

    if (options.max_message_size > options.max_mem_limit) {
        options.max_message_size = options.max_mem_limit;
    }

    bufsize = sizeof(float)*(options.max_message_size/sizeof(float));
    if (allocate_buffer((void**)&sendbuf, bufsize, options.accel)) {
        fprintf(stderr, "Could Not Allocate Memory [rank %d]\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    set_buffer(sendbuf, options.accel, 1, bufsize);

    bufsize = sizeof(float)*(options.max_message_size/sizeof(float));
    if (allocate_buffer((void**)&recvbuf, bufsize, options.accel)) {
        fprintf(stderr, "Could Not Allocate Memory [rank %d]\n", rank);
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }
    set_buffer(recvbuf, options.accel, 0, bufsize);

    print_preamble(rank, numprocs);

    size = options.max_message_size/sizeof(float);

    for (j = 0; j < _NUMBER_OF_KERNELS_; j++)
    {
        size = problems[j];

        options.iterations = numRepeats[j];
        MPI_Barrier(MPI_COMM_WORLD);

        timer = 0.0;
        t_start = MPI_Wtime();
        for(i=0; i < options.iterations; i++) {
#ifdef ENABLE_MLSL
            MLSL_CALL(mlsl_distribution_all_reduce(distribution, sendbuf, recvbuf, size, DT_FLOAT, RT_SUM, GT_DATA, &request));
            MLSL_CALL(mlsl_environment_wait(env, request));
#else
            MPI_Allreduce(sendbuf, recvbuf, size, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
#endif
        }

        t_stop = MPI_Wtime();
        timer = t_stop-t_start;

        latency = (double)(timer * 1e3) / options.iterations;

        MPI_Reduce(&latency, &min_time, 1, MPI_DOUBLE, MPI_MIN, 0,
                MPI_COMM_WORLD);
        MPI_Reduce(&latency, &max_time, 1, MPI_DOUBLE, MPI_MAX, 0,
                MPI_COMM_WORLD);
        MPI_Reduce(&latency, &avg_time, 1, MPI_DOUBLE, MPI_SUM, 0,
                MPI_COMM_WORLD);
        avg_time = avg_time/numprocs;

        print_stats(rank, size, avg_time, min_time, max_time);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    free_buffer(sendbuf, options.accel);
    free_buffer(recvbuf, options.accel);

    FINALIZE();

    if (none != options.accel) {
        if (cleanup_accel()) {
            fprintf(stderr, "Error cleaning up device\n");
            exit(EXIT_FAILURE);
        }
    }

    return EXIT_SUCCESS;
}
