/*******************************************************************************
 * Copyright 2016 Intel Corporation All Rights Reserved.
 *
 * The source code,  information  and material  ("Material") contained  herein is
 * owned by Intel Corporation or its  suppliers or licensors,  and  title to such
 * Material remains with Intel  Corporation or its  suppliers or  licensors.  The
 * Material  contains  proprietary  information  of  Intel or  its suppliers  and
 * licensors.  The Material is protected by  worldwide copyright  laws and treaty
 * provisions.  No part  of  the  Material   may  be  used,  copied,  reproduced,
 * modified, published,  uploaded, posted, transmitted,  distributed or disclosed
 * in any way without Intel's prior express written permission.  No license under
 * any patent,  copyright or other  intellectual property rights  in the Material
 * is granted to  or  conferred  upon  you,  either   expressly,  by implication,
 * inducement,  estoppel  or  otherwise.  Any  license   under such  intellectual
 * property rights must be express and approved by Intel in writing.
 *
 * Unless otherwise agreed by Intel in writing,  you may not remove or alter this
 * notice or  any  other  notice   embedded  in  Materials  by  Intel  or Intel's
 * suppliers or licensors in any way.
 *******************************************************************************/

typedef struct gemm_params {
    char transa;
    char transb;
    int m;
    int n;
    int k;
    int lda;
    int ldb;
    int ldc;
} gemm_params_t;

gemm_params_t gemm_params[] = { 
  {'N','N',1760,16,1760},
  {'N','N',1760,32,1760},
  {'N','N',1760,64,1760},
  {'N','N',1760,128,1760},
  {'N','N',1760,7000,1760},
  {'N','N',2048,16,2048},
  {'N','N',2048,32,2048},
  {'N','N',2048,64,2048},
  {'N','N',2048,128,2048},
  {'N','N',2048,7000,2048},
  {'N','N',2560,16,2560},
  {'N','N',2560,32,2560},
  {'N','N',2560,64,2560},
  {'N','N',2560,128,2560},
  {'N','N',2560,7000,2560},
  {'N','N',4096,16,4096},
  {'N','N',4096,32,4096},
  {'N','N',4096,64,4096},
  {'N','N',4096,128,4096},
  {'N','N',4096,7000,4096},
  {'T','N',1760,16,1760},
  {'T','N',1760,32,1760},
  {'T','N',1760,64,1760},
  {'T','N',1760,128,1760},
  {'T','N',1760,7000,1760},
  {'T','N',2048,16,2048},
  {'T','N',2048,32,2048},
  {'T','N',2048,64,2048},
  {'T','N',2048,128,2048},
  {'T','N',2048,7000,2048},
  {'T','N',2560,16,2560},
  {'T','N',2560,32,2560},
  {'T','N',2560,64,2560},
  {'T','N',2560,128,2560},
  {'T','N',2560,7000,2560},
  {'T','N',4096,16,4096},
  {'T','N',4096,32,4096},
  {'T','N',4096,64,4096},
  {'T','N',4096,128,4096},
  {'T','N',4096,7000,4096},
  {'N','T',1760,7133,1760},
  {'N','T',2048,7133,2048},
  {'N','T',2560,7133,2560},
  {'N','T',4096,7133,4096},
  {'N','N',5124,9124,1760},
  {'N','N',35,8457,1760},
  {'N','N',5124,9124,2048},
  {'N','N',35,8457,2048},
  {'N','N',5124,9124,2560},
  {'N','N',35,8457,2560},
  {'N','N',5124,9124,4096},
  {'N','N',35,8457,4096},
  {'T','N',5124,9124,1760},
  {'T','N',35,8457,1760},
  {'T','N',5124,9124,2048},
  {'T','N',35,8457,2048},
  {'T','N',5124,9124,2560},
  {'T','N',35,8457,2560},
  {'T','N',5124,9124,4096},
  {'T','N',35,8457,4096},
  {'N','N',7680,16,2560},
  {'N','N',7680,32,2560},
  {'N','N',7680,64,2560},
  {'N','N',7680,128,2560},
  {'T','N',7680,16,2560},
  {'T','N',7680,32,2560},
  {'T','N',7680,64,2560},
  {'T','N',7680,128,2560},
  {'N','N',3072,16,1024},
  {'N','N',3072,32,1024},
  {'N','N',3072,64,1024},
  {'N','N',3072,128,1024},
  {'T','N',3072,16,1024},
  {'T','N',3072,32,1024},
  {'T','N',3072,64,1024},
  {'T','N',3072,128,1024},
  {'N','T',3072,7435,1024},
  {'N','T',7680,5481,2560}
};
