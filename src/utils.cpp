//
// Created by curio on 2025/2/9.
//
#include "utils.h"
#include "common.h"
#include <algorithm>
#include <complex>
#include <cstring>
#include <memory>
#include <queue>
#include <utility> // for std::pair

void write_binary_file(const char *file_name, const char *buf, size_t size)
{
    std::ofstream ofs(file_name, std::ios::out | std::ios::binary);
    ofs.write(buf, size);
    ofs.close();
}

std::vector<std::vector<float>> matmul(const std::vector<std::vector<float>> &A,
                                       const std::vector<std::vector<float>> &B) {
    size_t A_rows = A.size();
    size_t A_cols = A[0].size();
    size_t B_rows = B.size();
    size_t B_cols = B[0].size();

    // 检查是否可以进行矩阵乘法
    if (A_cols != B_rows) {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
    }

    // 初始化结果矩阵
    std::vector<std::vector<float>> result(A_rows, std::vector<float>(B_cols, 0.0f));

    // 执行矩阵乘法
    for (size_t i = 0; i < A_rows; ++i) {
        for (size_t j = 0; j < B_cols; ++j) {
            for (size_t k = 0; k < A_cols; ++k) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    return result;
}


std::vector<float> reflectPad(const std::vector<float> &input, size_t padFront, size_t padBack) {
    // 计算填充后的新大小
    size_t newSize = input.size() + padFront + padBack;
    std::vector<float> padded(newSize);

    // 复制原始数据到中间位置
    std::copy(input.begin(), input.end(), padded.begin() + padFront);

    // 反射填充前端
    for (size_t i = 0; i < padFront; ++i) {
        padded[i] = input[padFront - i - 1];
    }

    // 反射填充后端
    for (size_t i = 0; i < padBack; ++i) {
        padded[padded.size() - 1 - i] = input[input.size() - 1 - i];
    }

    return padded;
}

template<class T>
std::vector<std::pair<T, int> > topK(const std::vector<T> &arr, size_t k) {
    std::vector<std::pair<T, int> > result;

    if (k <= 0 || arr.empty()) {
        throw std::invalid_argument("Invalid k value");
    }

    std::priority_queue<std::pair<T, int>, std::vector<std::pair<T, int> >, std::greater<std::pair<T, int> > > minHeap;

    for (int i = 0; i < arr.size(); ++i) {
        T value = arr[i];
        int index = i;

        // Push the current element into the heap
        if (minHeap.size() < k) {
            minHeap.push({value, index});
        } else {
            // Compare with the smallest element in the heap
            if (value > minHeap.top().first) {
                minHeap.pop();
                minHeap.push({value, index});
            }
        }
    }

    while (!minHeap.empty()) {
        result.push_back(minHeap.top());
        minHeap.pop();
    }

    std::reverse(result.begin(), result.end());

    return result;
}

template std::vector<std::pair<float, int>> topK<float>(const std::vector<float> &, size_t);

template<class T>
std::vector<T> softmax(const std::vector<T> &x) {
    std::vector<T> result;
    double max_x = x[0];

    for (double val: x) {
        if (val > max_x) max_x = val;
    }

    double sum = 0.0;
    for (double val: x) {
        sum += std::exp(val - max_x);
    }

    for (double val: x) {
        result.push_back((float) (std::exp(val - max_x) / sum));
    }

    return result;
}

float bf16_to_fp32(uint16_t bf16)
{
    uint32_t fp32_bits = bf16 << 16;

    float result;
    std::memcpy(&result, &fp32_bits, sizeof(result));
    return result;
}

tensor_info<float> wte_get_data(tensor_info<long> &input_ids)
{
    std::vector<float> wte_data(input_ids.data.size() * 896);
    std::string weights_path;
    size_t size = 0;
    if (WTE_F16)
    {
        weights_path = "../data/wte_weights_bf16.bin";
        size = 896 * sizeof(float)/2;
    }
    else
    {
        weights_path = "../data/wte_weights.bin";
        size = 896 * sizeof(float);
    }
    
    FILE *file = fopen(weights_path.c_str(), "rb");

    if (WTE_F16)
    {
        std::vector<uint16_t> buffer(896);
        for (size_t i = 0; i < input_ids.data.size(); i++)
        {
            fseek(file, input_ids.data[i] * size, SEEK_SET);
            size_t bytes_read = fread(buffer.data(), 1, size, file);
            (void)bytes_read;
            auto ptr = wte_data.data() + i * 896;
            for (int j = 0; j < 896; j++)
            {
                ptr[j] = bf16_to_fp32(buffer[j]);
            }
        }
        fclose(file);
        std::vector<long> new_shape = input_ids.shape;
        new_shape.emplace_back(896);
        tensor_info<float> result{.data = wte_data, .shape = new_shape};
        return std::move(result);
    }
    else
    {
        std::vector<float> buffer(896);
        for (size_t i = 0; i < input_ids.data.size(); i++)
        {
            fseek(file, input_ids.data[i] * size, SEEK_SET);
            size_t bytes_read = fread(buffer.data(), 1, size, file);
            (void)bytes_read;
            auto ptr = wte_data.data() + i * 896;
            for (int j = 0; j < 896; j++)
            {
                ptr[j] = buffer[j];
            }
        }
        fclose(file);
        std::vector<long> new_shape = input_ids.shape;
        new_shape.emplace_back(896);
        tensor_info<float> result{.data = wte_data, .shape = new_shape};
        return std::move(result);
    }
}