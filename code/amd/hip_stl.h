#pragma once

namespace force {

template<typename T>
class device_ptr{
public:
  T *ptr;
  device_ptr(T *ptr) : ptr(ptr) {}
};

template<typename T, typename U>
void fill(device_ptr<T> begin, device_ptr<T> end, U val){
    
}

}
