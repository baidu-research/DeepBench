/*
 * Copyright (C) 2002-2014 the Network-Based Computing Laboratory
 * (NBCL), The Ohio State University.
 *
 * Contact: Dr. D. K. Panda (panda@cse.ohio-state.edu)
 *
 * For detailed copyright and licensing information, please refer to the
 * copyright file COPYRIGHT in the top level OMB directory.
 */
#ifndef OSU_COLL_H
#define OSU_COLL_H 1

#include <mpi.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <stdint.h>

#ifndef DEFAULT_MAX_MESSAGE_SIZE
//#define DEFAULT_MAX_MESSAGE_SIZE 67108864
#define DEFAULT_MAX_MESSAGE_SIZE 268435456
#endif

#define SKIP 200
#define SKIP_LARGE 10
#define LARGE_MESSAGE_SIZE 8192
#define MAX_ALIGNMENT 16384
//#define MAX_MEM_LIMIT 67108864
#define MAX_MEM_LIMIT 268435456
#define MAX_MEM_LOWER_LIMIT (1*1024*1024)

#ifdef _ENABLE_OPENACC_
#   define OPENACC_ENABLED 1
#else
#   define OPENACC_ENABLED 0
#endif

#ifdef _ENABLE_CUDA_
#   define CUDA_ENABLED 1
#else
#   define CUDA_ENABLED 0
#endif

#ifndef BENCHMARK
#   define BENCHMARK "MPI%s BENCHMARK NAME UNSET"
#endif

#ifdef PACKAGE_VERSION
#   define HEADER "# " BENCHMARK " v" PACKAGE_VERSION "\n"
#else
#   define HEADER "# " BENCHMARK "\n"
#endif

#ifndef FIELD_WIDTH
#   define FIELD_WIDTH 20
#endif

#ifndef FLOAT_PRECISION
#   define FLOAT_PRECISION 2
#endif

static int iterations = 1000;
static int iterations_large = 100;
static int print_size = 0;
static uint64_t max_mem_limit = MAX_MEM_LIMIT;
static int process_args (int argc, char *argv[], int rank, int * size, int * full) __attribute__((unused));
static void print_header (int rank, int full) __attribute__((unused));
static void print_data (int rank, int full, int size, double avg_time, double
        min_time, double max_time, int iterations) __attribute__((unused));

static void print_usage(int rank, const char * prog, int has_size)
{
    if (rank == 0) {
        if (has_size) {
            fprintf(stdout, " USAGE : %s [-m SIZE] [-i ITER] [-f] [-hv] [-M SIZE]\n", prog);
            fprintf(stdout, "  -m : Set maximum message size to SIZE.\n");
            fprintf(stdout, "       By default, the value of SIZE is 1MB.\n");
            fprintf(stdout, "  -i : Set number of iterations per message size to ITER.\n");
            fprintf(stdout, "       By default, the value of ITER is 1000 for small messages\n");
            fprintf(stdout, "       and 100 for large messages.\n");
            fprintf(stdout, "  -M : Set maximum memory consumption (per process) to SIZE. \n");
            fprintf(stdout, "       By default, the value of SIZE is 512MB.\n");
        }

        else {
            fprintf(stdout, " USAGE : %s [-i ITER] [-f] [-hv] \n", prog);
            fprintf(stdout, "  -i : Set number of iterations to ITER.\n");
            fprintf(stdout, "       By default, the value of ITER is 1000.\n");
        }

        fprintf(stdout, "  -f : Print full format listing.  With this option\n");
        fprintf(stdout, "       the MIN/MAX latency and number of ITERATIONS are\n");
        fprintf(stdout, "       printed out in addition to the AVERAGE latency.\n");

        fprintf(stdout, "  -h : Print this help.\n");
        fprintf(stdout, "  -v : Print version info.\n");
        fprintf(stdout, "\n");
        fflush(stdout);
    }
}

static void print_version()
{
        fprintf(stdout, HEADER, "");
        fflush(stdout);
}

static int process_args (int argc, char *argv[], int rank, int * size, int * full)
{
    char c;

    if (size) {
        print_size = 1;
    }

    while ((c = getopt(argc, argv, ":hvfm:i:M:")) != -1) {
        switch (c) {
            case 'h':
                print_usage(rank, argv[0], size != NULL);
                return 1;

            case 'v':
                if (rank == 0) {
                    print_version();
                }

                return 1;

            case 'm':
                if (size) {
                    *size = atoi(optarg);
                    if (*size < 0) {
                        print_usage(rank, argv[0], size != NULL);
                        return -1;
                    }
                }

                else {
                    print_usage(rank, argv[0], 0);
                    return -1;
                }
                break;

            case 'i':
                iterations_large = atoi(optarg);
                iterations = iterations_large;
                if (iterations < 1) {
                    print_usage(rank, argv[0], size != NULL);
                    return -1;
                }
                break;

            case 'f':
                *full = 1;
                break;

            case 'M':
                max_mem_limit = atoll(optarg);
                if (max_mem_limit < MAX_MEM_LOWER_LIMIT) {
                    max_mem_limit = MAX_MEM_LOWER_LIMIT;
                    if(rank == 0) fprintf(stderr,"Requested memory limit too low. ");
                    if(rank == 0) fprintf(stderr,"Reverting to default lower-limit value %d\n",
                                          MAX_MEM_LOWER_LIMIT);
                }
                break;

            default:
                if (rank == 0) {
                    print_usage(rank, argv[0], size != NULL);
                }

                return -1;
        }
    }

    return 0;
}

static void print_header (int rank, int full)
{
    if(rank == 0) {
        fprintf(stdout, HEADER, "");

        if (print_size) {
            fprintf(stdout, "%-*s", 10, "# Size");
            fprintf(stdout, "%*s", FIELD_WIDTH, "Avg Latency(ms)");
        }

        else {
            fprintf(stdout, "# Avg Latency(us)");
        }

        if (full) {
            fprintf(stdout, "%*s", FIELD_WIDTH, "Min Latency(ms)");
            fprintf(stdout, "%*s", FIELD_WIDTH, "Max Latency(ms)");
            fprintf(stdout, "%*s\n", 12, "Iterations");
        }

        else {
            fprintf(stdout, "\n");
        }

        fflush(stdout);
    }
}

static void print_data (int rank, int full, int size, double avg_time, double
        min_time, double max_time, int iterations)
{
    if(rank == 0) {
        if (print_size) {
            fprintf(stdout, "%-*d", 10, size);
            fprintf(stdout, "%*.*f", FIELD_WIDTH, FLOAT_PRECISION, avg_time);
        }

        else {
            fprintf(stdout, "%*.*f", 17, FLOAT_PRECISION, avg_time);
        }

        if (full) {
            fprintf(stdout, "%*.*f%*.*f%*d\n",
                    FIELD_WIDTH, FLOAT_PRECISION, min_time,
                    FIELD_WIDTH, FLOAT_PRECISION, max_time,
                    12, iterations);
        }

        else {
            fprintf(stdout, "\n");
        }

        fflush(stdout);
    }
}

enum po_ret_type {
    po_cuda_not_avail,
    po_openacc_not_avail,
    po_bad_usage,
    po_help_message,
    po_version_message,
    po_okay,
};

enum accel_type {
    none,
    cuda,
    openacc
};

struct {
    enum accel_type accel;
    int show_size;
    int show_full;
    size_t max_message_size;
    size_t iterations;
    size_t iterations_large;
    size_t max_mem_limit;
} options;

/*
 * Option Processing
 */
enum po_ret_type process_options (int argc, char *argv[]);

/*
 * Print Information
 */
void print_bad_usage_message (int rank);
void print_help_message (int rank);
void print_version_message (int rank);
void print_preamble (int rank, int numprocs);
void print_stats (int rank, int size, double avg, double min, double max);

/*
 * Memory Management
 */
int allocate_buffer (void ** buffer, size_t size, enum accel_type type);
void free_buffer (void * buffer, enum accel_type type);
void set_buffer (void * buffer, enum accel_type type, int data, size_t size);

/*
 * CUDA Context Management
 */
int init_accel (void);
int cleanup_accel (void);

/*
 * Set Benchmark Properties
 */
void set_header (const char * header);
void set_benchmark_name (const char * name);
void enable_accel_support (void);

#endif
