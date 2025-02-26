//
// Created by curio on 2025/2/7.
//
#pragma once

#ifndef WAV2WAV_UTILS_H
#define WAV2WAV_UTILS_H
#include <vector>
#include "common.h"
//
//template <class T>
//class EmptyTensor
//{
//public:
//    EmptyTensor() = default;
//    auto EmptyTensor3D(std::vector<int> shape)
//    {
//        size_t dims = shape.size();
//        std::vector<T> subTensor(shape[dims-1]);
//        for(int i = dims-2; i>=0; i--)
//        {
//            std::vector<T> tensor (shape[i], subTensor);
//            subTensor = tensor;
//        }
//        return subTensor;
//    }
//};

std::vector<std::vector<float>> matmul(const std::vector<std::vector<float>>& A,
                                       const std::vector<std::vector<float>>& B);

std::vector<float> reflectPad(const std::vector<float>& input, size_t padFront, size_t padBack);

template <class T>
std::vector<std::vector<T>> transpose(const std::vector<std::vector<T>>& matrix);

template<class T>
std::vector<std::pair<T, int> > topK(const std::vector<T> &arr, size_t k);

template<class T>
std::vector<T> softmax(std::vector<T>& x);

tensor_info<float> wte_get_data(tensor_info<long> &input_ids);

#endif //WAV2WAV_UTILS_H
