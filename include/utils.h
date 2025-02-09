//
// Created by curio on 2025/2/7.
//
#pragma once

#ifndef WAV2WAV_UTILS_H
#define WAV2WAV_UTILS_H
#include <vector>

std::vector<std::vector<float>> matmul(const std::vector<std::vector<float>>& A,
                                       const std::vector<std::vector<float>>& B);

std::vector<float> reflectPad(const std::vector<float>& input, size_t padFront, size_t padBack);

template <class T>
std::vector<std::vector<T>> transpose(const std::vector<std::vector<T>>& matrix);

#endif //WAV2WAV_UTILS_H
