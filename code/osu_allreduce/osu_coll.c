/*
 * Copyright (C) 2002-2014 the Network-Based Computing Laboratory
 * (NBCL), The Ohio State University.
 *
 * Contact: Dr. D. K. Panda (panda@cse.ohio-state.edu)
 *
 * For detailed copyright and licensing information, please refer to the
 * copyright file COPYRIGHT in the top level directory.
 */

/*
 * HEADER FILES
 */
#include "osu_coll.h"

#ifdef _ENABLE_OPENACC_
#include <openacc.h>
#endif

#ifdef _ENABLE_CUDA_
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#ifdef ENABLE_MLSL
#include "mlsl.h"
mlsl_environment env;
#endif

/*
 * GLOBAL VARIABLES
 */
#ifdef _ENABLE_CUDA_
static CUcontext cuContext;
#endif

static char const * benchmark_header = NULL;
static char const * benchmark_name = NULL;
static int accel_enabled = 0;

static struct {
    char const * message;
    char const * optarg;
    char opt;
} bad_usage;

static int
set_max_message_size (int value)
{
    if (0 > value) {
        return -1;
    }

    options.max_message_size = value;

    return 0;
}

static int
set_num_iterations (int value)
{
    if (1 > value) {
        return -1;
    }

    options.iterations = value;
    options.iterations_large = value;

    return 0;
}

static int
set_max_memlimit (int value)
{
    options.max_mem_limit = value;

    if (value < MAX_MEM_LOWER_LIMIT) {
        options.max_mem_limit = MAX_MEM_LOWER_LIMIT;
        fprintf(stderr,"Requested memory limit too low, using [%d] instead.",
                MAX_MEM_LOWER_LIMIT);
    }

    return 0;
}

void
set_header (const char * header)
{
    benchmark_header = header;
}

void
set_benchmark_name (const char * name)
{
    benchmark_name = name;
}

void
enable_accel_support (void)
{
    accel_enabled = (CUDA_ENABLED || OPENACC_ENABLED);
}

enum po_ret_type
process_options (int argc, char *argv[])
{
    extern char * optarg;
    extern int optind, optopt;

    char const * optstring = (accel_enabled) ? "+:d:hvfm:i:M:" : "+:hvfm:i:M:";
    char c;

    /*
     * SET DEFAULT OPTIONS
     */
    options.accel = none;
    options.show_size = 1;
    options.show_full = 0;
    options.max_message_size = DEFAULT_MAX_MESSAGE_SIZE;
    options.max_mem_limit = MAX_MEM_LIMIT;
    options.iterations = 1000;
    options.iterations_large = 100;

    while ((c = getopt(argc, argv, optstring)) != -1) {
        bad_usage.opt = c;
        bad_usage.optarg = NULL;
        bad_usage.message = NULL;

        switch (c) {
            case 'h':
                return po_help_message;
            case 'v':
                return po_version_message;
            case 'm':
                if (set_max_message_size(atoi(optarg))) {
                    bad_usage.message = "Invalid Message Size";
                    bad_usage.optarg = optarg;

                    return po_bad_usage;
                }
                break;
            case 'i':
                if (set_num_iterations(atoi(optarg))) {
                    bad_usage.message = "Invalid Number of Iterations";
                    bad_usage.optarg = optarg;

                    return po_bad_usage;
                }
                break;
            case 'f':
                options.show_full = 1;
                break;
            case 'M':
                /*
                 * This function does not error but prints a warning message if
                 * the value is too low.
                 */
                set_max_memlimit(atoll(optarg));
                break;
            case 'd':
                if (!accel_enabled) {
                    bad_usage.message = "Benchmark Does Not Support "
                        "Accelerator Transfers";
                    bad_usage.optarg = optarg;
                    return po_bad_usage;
                }

                else if (0 == strncasecmp(optarg, "cuda", 10)) {
                    if (CUDA_ENABLED) {
                        options.accel = cuda;
                    }

                    else {
                        bad_usage.message = "CUDA Support Not Enabled\n"
                            "Please recompile benchmark with CUDA support";
                        bad_usage.optarg = optarg;
                        return po_bad_usage;
                    }
                }

                else if (0 == strncasecmp(optarg, "openacc", 10)) {
                    if (OPENACC_ENABLED) {
                        options.accel = openacc;
                    }

                    else {
                        bad_usage.message = "OpenACC Support Not Enabled\n"
                            "Please recompile benchmark with OpenACC support";
                        bad_usage.optarg = optarg;
                        return po_bad_usage;
                    }
                }

                else {
                    bad_usage.message = "Invalid Accel Type Specified";
                    bad_usage.optarg = optarg;
                    return po_bad_usage;
                }
                break;
            case ':':
                bad_usage.message = "Option Missing Required Argument";
                bad_usage.opt = optopt;
                return po_bad_usage;
            default:
                bad_usage.message = "Invalid Option";
                bad_usage.opt = optopt;
                return po_bad_usage;
        }
    }

    return po_okay;
}

void
print_bad_usage_message (int rank)
{
    if (rank) return;

    if (bad_usage.optarg) {
        fprintf(stderr, "%s [-%c %s]\n\n", bad_usage.message, bad_usage.opt,
                bad_usage.optarg);
    }

    else {
        fprintf(stderr, "%s [-%c]\n\n", bad_usage.message, bad_usage.opt);
    }

    print_help_message(rank);
}

void
print_help_message (int rank)
{
    if (rank) return;

    printf("Usage: %s [options]\n", benchmark_name);
    printf("options:\n");

    if (accel_enabled) {
        printf("  -d TYPE       use accelerator device buffers which can be of TYPE `cuda' or\n");
        printf("                `openacc' (uses standard host buffers if not specified)\n");
    }

    if (options.show_size) {
        printf("  -m SIZE       set maximum message size to SIZE bytes (default 1048576)\n");
        printf("  -M SIZE       set per process maximum memory consumption to SIZE bytes\n");
        printf("                (default %d)\n", MAX_MEM_LIMIT);
    }

    printf("  -i ITER       set iterations per message size to ITER (default 1000 for small\n");
    printf("                messages, 100 for large messages)\n");
    printf("  -f            print full format listing (MIN/MAX latency and ITERATIONS\n");
    printf("                displayed in addition to AVERAGE latency)\n");
    printf("  -h            print this help\n");
    printf("  -v            print version info\n");
    printf("\n");
    fflush(stdout);
}

void
print_version_message (int rank)
{
    if (rank) return;

    switch (options.accel) {
        case cuda:
            printf(benchmark_header, "-CUDA");
            break;
        case openacc:
            printf(benchmark_header, "-OPENACC");
            break;
        default:
            printf(benchmark_header, "");
            break;
    }

    fflush(stdout);
}

void
print_preamble (int rank, int numprocs)
{
    if (rank) return;

    switch (options.accel) {
        case cuda:
            printf(benchmark_header, "-CUDA");
            break;
        case openacc:
            printf(benchmark_header, "-OPENACC");
            break;
        default:
            printf(benchmark_header, "");
            break;
    }

    fprintf(stdout, "# Number of Ranks %d\n", numprocs);

    if (options.show_size) {
        fprintf(stdout, "%-*s", 10, "# Size");
        fprintf(stdout, "%*s", FIELD_WIDTH, "Avg Latency(ms)");
    }

    else {
        fprintf(stdout, "# Avg Latency(ms)");
    }

    if (options.show_full) {
        fprintf(stdout, "%*s", FIELD_WIDTH, "Min Latency(ms)");
        fprintf(stdout, "%*s", FIELD_WIDTH, "Max Latency(ms)");
        fprintf(stdout, "%*s\n", 12, "Iterations");
    }

    else {
        fprintf(stdout, "\n");
    }

    fflush(stdout);
}

void
print_stats (int rank, int size, double avg_time, double min_time, double
        max_time)
{
    if (rank) return;

    if (options.show_size) {
        fprintf(stdout, "%-*d", 10, size);
        fprintf(stdout, "%*.*f", FIELD_WIDTH, FLOAT_PRECISION, avg_time);
    }

    else {
        fprintf(stdout, "%*.*f", 17, FLOAT_PRECISION, avg_time);
    }

    if (options.show_full) {
        fprintf(stdout, "%*.*f%*.*f%*lu\n",
                FIELD_WIDTH, FLOAT_PRECISION, min_time,
                FIELD_WIDTH, FLOAT_PRECISION, max_time,
                12, options.iterations);
    }

    else {
        fprintf(stdout, "\n");
    }

    fflush(stdout);
}

void
set_buffer (void * buffer, enum accel_type type, int data, size_t size)
{
#ifdef _ENABLE_OPENACC_
    size_t i;
    char * p = (char *)buffer;
#endif

    switch (type) {
        case none:
            memset(buffer, data, size);
            break;
        case cuda:
#ifdef _ENABLE_CUDA_
            cudaMemset(buffer, data, size);
#endif
            break;
        case openacc:
#ifdef _ENABLE_OPENACC_
#pragma acc parallel loop deviceptr(p)
            for(i = 0; i < size; i++) {
                p[i] = data;
            }
#endif
            break;
    }
}

int
allocate_buffer (void ** buffer, size_t size, enum accel_type type)
{
    size_t alignment = sysconf(_SC_PAGESIZE);
#ifdef _ENABLE_CUDA_
    cudaError_t cuerr = cudaSuccess;
#endif

    switch (type) {
        case none:
#ifdef ENABLE_MLSL
            mlsl_environment_get_env(&env);
            return mlsl_environment_alloc(env, size, alignment, buffer);
#else
            return posix_memalign(buffer, alignment, size);
#endif
#ifdef _ENABLE_CUDA_
        case cuda:
            cuerr = cudaMalloc(buffer, size);
            if (cudaSuccess != cuerr) {
                return 1;
            }

            else {
                return 0;
            }
#endif
#ifdef _ENABLE_OPENACC_
        case openacc:
            *buffer = acc_malloc(size);
            if (NULL == *buffer) {
                return 1;
            }

            else {
                return 0;
            }
#endif
        default:
            return 1;
    }
}

void
free_buffer (void * buffer, enum accel_type type)
{
    switch (type) {
        case none:
#ifdef ENABLE_MLSL
            mlsl_environment_get_env(&env);
            mlsl_environment_free(env, buffer);
#else
            free(buffer);
#endif
            break;
        case cuda:
#ifdef _ENABLE_CUDA_
            cudaFree(buffer);
#endif
            break;
        case openacc:
#ifdef _ENABLE_OPENACC_
            acc_free(buffer);
#endif
            break;
    }
}

int
init_accel (void)
{
#if defined(_ENABLE_OPENACC_) || defined(_ENABLE_CUDA_)
     char * str;
     int local_rank, dev_count;
     int dev_id = 0;
#endif
#ifdef _ENABLE_CUDA_
    CUresult curesult = CUDA_SUCCESS;
    CUdevice cuDevice;
#endif

    switch (options.accel) {
#ifdef _ENABLE_CUDA_
        case cuda:
            if ((str = getenv("SLURM_LOCALID")) != NULL) {
                cudaGetDeviceCount(&dev_count);
                local_rank = atoi(str);
                dev_id = local_rank % dev_count;
            }

            curesult = cuInit(0);
            if (curesult != CUDA_SUCCESS) {
                printf("CuInit Failed: %4d \n",curesult);
                return 1;
            }

            curesult = cuDeviceGet(&cuDevice, dev_id);
            if (curesult != CUDA_SUCCESS) {
                printf("cuDeviceGet Failed: %4d \n",curesult);
                return 1;
            }

            curesult = cuCtxCreate(&cuContext, 0, cuDevice);
            if (curesult != CUDA_SUCCESS) {
                printf("cuCtxCreate Failed: %4d \n",curesult);
                return 1;
            }
            break;
#endif
#ifdef _ENABLE_OPENACC_
        case openacc:
            if ((str = getenv("LOCAL_RANK")) != NULL) {
                dev_count = acc_get_num_devices(acc_device_not_host);
                local_rank = atoi(str);
                dev_id = local_rank % dev_count;
            }

            acc_set_device_num (dev_id, acc_device_not_host);
            break;
#endif
        default:
            fprintf(stderr, "Invalid device type, should be cuda or openacc\n");
            return 1;
    }

    return 0;
}

int
cleanup_accel (void)
{
#ifdef _ENABLE_CUDA_
    CUresult curesult = CUDA_SUCCESS;
#endif

    switch (options.accel) {
#ifdef _ENABLE_CUDA_
        case cuda:
            curesult = cuCtxDestroy(cuContext);

            if (curesult != CUDA_SUCCESS) {
                return 1;
            }
            break;
#endif
#ifdef _ENABLE_OPENACC_
        case openacc:
            acc_shutdown(acc_device_nvidia);
            break;
#endif
        default:
            fprintf(stderr, "Invalid accel type, should be cuda or openacc\n");
            return 1;
    }

    return 0;
}

/* vi:set sw=4 sts=4 tw=80: */

