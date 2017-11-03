#ifndef _ALL_REDUCE_KERNELS_
#define _ALL_REDUCE_KERNELS_


#define _NUMBER_OF_KERNELS_ 3

int64_t all_reduce_kernels_size[] = {131072, 2097152, 4194304};
int64_t all_reduce_kernels_repeat[] = {10, 10, 10};

// Vector saves the number of floats each rank will transfer
//std::vector<int> all_reduce_problems_size = {131072, 2097152, 4194304}; //, 8388608, 16777216, 33554432, 67108864, 134217728}; //{131072, 2097152, 4194304, 8388608, 16777216, 33554432, 67108864, 134217728}; //100000, 3097600, 4194304, 6553600}; //, 16777217, 67930000, 152220000};
//std::vector<int> all_reduce_problems_repeat = {100, 10, 10, 10, 10, 10, 10, 10}; //{1000, 100, 10, 10, 10, 10, 10, 10};

#endif
