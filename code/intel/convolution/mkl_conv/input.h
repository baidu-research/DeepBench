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

 typedef struct conv_params {
    int groups;
    int minibatch;
    int w;
    int h;
    int ic;
    int oc;
    int kw;
    int kh;
    int c_stride;
    int offset;
    int iters;
} conv_params_t;

conv_params_t conv_params[] = {
// Use libxsmm for the non-square convolutions
//  {1,4,700,161,1,32,5,20,2,0,500},
//  {1,4,700,161,1,32,5,20,2,0,500},
//  {1,8,700,161,1,32,5,20,2,0,500},
//  {1,8,700,161,1,32,5,20,2,0,500},
//  {1,16,700,161,1,32,5,20,2,0,500},
//  {1,32,700,161,1,32,5,20,2,0,500},
//  {1,4,341,79,32,32,5,10,2,0,500},
//  {1,8,341,79,32,32,5,10,2,0,500},
//  {1,16,341,79,32,32,5,10,2,0,500},
//  {1,32,341,79,32,32,5,10,2,0,500},
  {1,16,480,48,1,16,3,3,1,0,500},
  {1,16,240,24,16,32,3,3,1,0,500},
  {1,16,120,12,32,64,3,3,1,0,500},
  {1,16,60,6,64,128,3,3,1,0,500},
  {1,8,108,108,3,64,3,3,2,0,500},
  {1,8,54,54,64,64,3,3,1,0,500},
  {1,8,27,27,128,128,3,3,1,0,500},
  {1,8,14,14,128,256,3,3,1,0,500},
  {1,8,7,7,256,512,3,3,1,0,500},
  {1,8,224,224,3,64,3,3,1,0,500},
  {1,8,112,112,64,128,3,3,1,0,500},
  {1,8,56,56,128,256,3,3,1,0,500},
  {1,8,28,28,256,512,3,3,1,0,500},
  {1,8,14,14,512,512,3,3,1,0,500},
  {1,8,7,7,512,512,3,3,1,0,500},
  {1,16,224,224,3,64,3,3,1,0,500},
  {1,16,112,112,64,128,3,3,1,0,500},
  {1,16,56,56,128,256,3,3,1,0,500},
  {1,16,28,28,256,512,3,3,1,0,500},
  {1,16,14,14,512,512,3,3,1,0,500},
  {1,16,7,7,512,512,3,3,1,0,500},
  {1,16,224,224,3,64,7,7,2,0,500},
  {1,16,28,28,192,32,5,5,1,0,500},
  {1,16,28,28,192,64,1,1,1,0,500},
  {1,16,14,14,512,48,5,5,1,0,500},
  {1,16,14,14,512,192,1,1,1,0,500},
  {1,16,7,7,832,256,1,1,1,0,500},
  {1,16,7,7,832,128,5,5,1,0,500},
};
