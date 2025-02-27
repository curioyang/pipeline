//
// Created by curio on 2025/2/7.
//
#pragma once

#ifndef WAV2WAV_UTILS_H
#define WAV2WAV_UTILS_H
#include <vector>
#include <chrono>
#include <string>
#include <fstream>
#include <iostream>
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


class ScopedTiming {
public:
    ScopedTiming(std::string info = "ScopedTiming") : m_info(info) {
        m_start = std::chrono::steady_clock::now();
    }

    ~ScopedTiming() {
        m_stop = std::chrono::steady_clock::now();
        double elapsed_ms = std::chrono::duration<double,std::milli>(m_stop - m_start).count();
        std::cout << m_info << " took " << elapsed_ms << " ms" << std::endl;
    }

private:
    std::string m_info;
    std::chrono::steady_clock::time_point m_start;
    std::chrono::steady_clock::time_point m_stop;
};

template <typename T>
void read_binary_file(const std::string &file_name, std::vector<T> &v)
{
    std::ifstream ifs(file_name, std::ios::binary);
    ifs.seekg(0, ifs.end);
    size_t len = ifs.tellg();
    v.resize(len / sizeof(T));
    ifs.seekg(0, ifs.beg);
    ifs.read(reinterpret_cast<char *>(v.data()), len);
    ifs.close();
}
void write_binary_file(const char *file_name, const char *buf, size_t size);

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

int16_t float_to_int16(float value);
#endif // WAV2WAV_UTILS_H
