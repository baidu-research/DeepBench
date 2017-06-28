/*******************************************************************************
* Copyright 2017 Intel Corporation
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

#include <math.h>

#include <iomanip>
#include <iostream>
#include <tuple>
#include <vector>

#include "omp.h"

#include "mkl.h"

#include "sparse_gemm_problems.h"

#define NRUN 10
#define NSERIES 10
#define ALIGNMENT 1024

double sample_normal()
{
    double u = ((double)rand() / (RAND_MAX)) * 2 - 1;
    double v = ((double)rand() / (RAND_MAX)) * 2 - 1;
    double r = u * u + v * v;
    if (r == 0 || r > 1)
        return sample_normal();
    double c = sqrt(-2 * log(r) / r);
    return u * c;
}

void prepare_matrix_sparse(MKL_INT row, MKL_INT col, float sparsity,
        MKL_INT **ia_ptr, MKL_INT **ja_ptr, float **a_ptr)
{
    MKL_INT *ia = (MKL_INT *)mkl_malloc(sizeof(MKL_INT) * (row + 1), ALIGNMENT);

    double s = sqrt((sparsity - sparsity * sparsity) * col);
    for (MKL_INT k = 1; k < row + 1; k++) {
        ia[k] = round(sample_normal() * s + col * sparsity);
        if (ia[k] < 0)
            ia[k] = 0;
        if (ia[k] > col)
            ia[k] = col;
    }

    ia[0] = 0;
    for (MKL_INT k = 0; k < row; k++) {
        ia[k + 1] += ia[k];
    }

    MKL_INT nnz = ia[row];
    MKL_INT *ja = (MKL_INT *)mkl_malloc(sizeof(MKL_INT) * (nnz + 1), ALIGNMENT);
    float *a = (float *)mkl_malloc(sizeof(float) * (nnz + 1), ALIGNMENT);

    for (MKL_INT k = 0; k < nnz; k++) {
        ja[k] = round((col - 1) * ((double)rand() / (RAND_MAX)));
        a[k] = 1. / 3. * sqrt(1 + k % 79);
    }

#pragma omp parallel for
    for (MKL_INT i = 0; i < row; i++) {
        const MKL_INT startRow = ia[i];
        const MKL_INT endRow = ia[i + 1];
        MKL_INT sorted = 0;
        MKL_INT elem;
        while (!sorted) {
            sorted = 1;
            for (elem = startRow; elem < endRow - 1; elem++) {
                // sort elements in the row with bubble sort
                const MKL_INT col1 = ja[elem];
                const MKL_INT col2 = ja[elem + 1];
                // col2 should be greater than col1
                // TODO: merge
                if (col1 > col2) {
                    // exchange elements
                    sorted = 0; // next iteration required
                    ja[elem] = col2;
                    ja[elem + 1] = col1;
                } else if (col1 == col2) {
                    sorted = 0;
                    ja[elem + 1] = (col1 != row - 1) ? col1 + 1 : 0;
                }
            }
        }
    }
    *ia_ptr = ia;
    *ja_ptr = ja;
    *a_ptr = a;

    return;
}

void clean_cache(size_t size)
{
#pragma omp parallel for schedule(static, 1)
    for (int j = 0; j < omp_get_num_threads(); j++) {
        double *arr = (double *)malloc(size * sizeof(double));
        for (size_t i = 0; i < size; i++)
            arr[i] = i / 3.14;
        free(arr);
    }
}

float scal_diff(MKL_INT N, float *x, float *y)
{
    float s = 0.;
    for (MKL_INT i = 0; i < N; i++)
        s += (x[i] - y[i]) * (x[i] - y[i]);
    return s;
}

void csrgemm_loop(sparse_operation_t transa, sparse_layout_t order, MKL_INT m,
        MKL_INT n, MKL_INT k, float *val, MKL_INT *ia, MKL_INT *ja, float *x,
        float *y)
{
    float *tmp_y = (float *)malloc(sizeof(float) * (n));

    for (MKL_INT row = 0; row < m; row++) {
        for (MKL_INT col = 0; col < n; col++)
            tmp_y[col] = 0;

        for (MKL_INT ind = ia[row]; ind < ia[row + 1]; ind++, val++, ja++)
            for (MKL_INT col = 0; col < n; col++)
                tmp_y[col] += *val * x[*ja + col * k];

        for (MKL_INT col = 0; col < n; col++)
            y[row + col * m] = tmp_y[col];
    }

    free(tmp_y);
}

double csrgemv(sparse_operation_t transa, sparse_layout_t order, MKL_INT m,
        MKL_INT n, MKL_INT k, float *val, MKL_INT *ia, MKL_INT *ja, float *x,
        float *y)
{
    sparse_matrix_t csrA;
    mkl_sparse_s_create_csr(
            &csrA, SPARSE_INDEX_BASE_ZERO, m, k, ia, ia + 1, ja, val);

    struct matrix_descr descrA_matrix;
    descrA_matrix.type = SPARSE_MATRIX_TYPE_GENERAL;
    descrA_matrix.mode = SPARSE_FILL_MODE_UPPER;
    descrA_matrix.diag = SPARSE_DIAG_NON_UNIT;

    if (n == 1)
        mkl_sparse_set_mv_hint(csrA, transa, descrA_matrix, NRUN);
    else
        mkl_sparse_set_mm_hint(csrA, transa, descrA_matrix, order, n, NRUN);
    mkl_sparse_optimize(csrA);

    double time = 0;
    for (MKL_INT j = 0; j < NRUN; j++) {
        clean_cache(32768);
        double t = dsecnd();
        if (n == 1)
            mkl_sparse_s_mv(transa, 1.0, csrA, descrA_matrix, x, 0.0, y);
        else
            mkl_sparse_s_mm(transa, 1.0, csrA, descrA_matrix, order, x, n, k,
                    0.0, y, m);
        time += dsecnd() - t;
    }

    mkl_sparse_destroy(csrA);
    return time / NRUN;
}

int main(int argc, char *argv[])
{
    using namespace std;

    cout << "             m   |   n  |   k  |    nnz   | time spmv  | perf "
            "spmv  | res spmv/loop\n";

    for (const auto &problem : inference_server_set) {

        MKL_INT m, n, k;
        bool a_t, b_t;
        float sparsity;

        tie(m, n, k, a_t, b_t, sparsity) = problem;

        sparse_operation_t transa = (a_t ? SPARSE_OPERATION_TRANSPOSE :
                                           SPARSE_OPERATION_NON_TRANSPOSE);
        sparse_layout_t order
                = (b_t ? SPARSE_LAYOUT_ROW_MAJOR : SPARSE_LAYOUT_COLUMN_MAJOR);

        float *x = (float *)mkl_malloc(sizeof(float) * n * k, ALIGNMENT);
        float *y = (float *)mkl_malloc(sizeof(float) * m * n, ALIGNMENT);
        float *y_chk = (float *)mkl_malloc(sizeof(float) * m * n, ALIGNMENT);

        for (MKL_INT i = 0; i < n * k; i++)
            x[i] = -100.89 + i % 89;

        MKL_INT *ia, *ja;
        float *val;
        prepare_matrix_sparse(m, k, 1. - sparsity, &ia, &ja, &val);
        MKL_INT nnz = ia[m] - ia[0];

        double min_time = csrgemv(transa, order, m, n, k, val, ia, ja, x, y);
        for (MKL_INT i = 1; i <= NSERIES; i++) {
            double time = csrgemv(transa, order, m, n, k, val, ia, ja, x, y);
            if (min_time > time)
                min_time = time;
        }

        csrgemm_loop(transa, order, m, n, k, val, ia, ja, x, y_chk);
        double s_m = scal_diff(m * n, y_chk, y);

        cout << "sparse    [" << setw(6) << m << "|" << setw(6) << n << "|"
             << setw(6) << k << "|" << setw(10) << nnz << "|" << setw(12)
             << min_time << "|" << setw(12)
             << 2. * nnz * n / min_time / 1024 / 1024 / 1024 << "|" << setw(12)
             << s_m / n << "\n";

        mkl_free(x);
        mkl_free(y);
        mkl_free(y_chk);
        mkl_free(ia);
        mkl_free(ja);
        mkl_free(val);
    }

    return 0;
}
